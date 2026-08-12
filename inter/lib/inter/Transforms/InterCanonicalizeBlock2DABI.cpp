#include "inter/Dialect/Inter/IR/XW.h"

#include "inter/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

#include <array>

namespace inter {
#define GEN_PASS_DEF_CANONICALIZEBLOCK2DABI
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

enum class Block2DBuiltin { Prefetch, Read, ReadTransform, Write };

static std::optional<Block2DBuiltin> classifyBuiltin(StringRef symbol) {
  return StringSwitch<std::optional<Block2DBuiltin>>(symbol)
      .Case("_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i",
            Block2DBuiltin::Prefetch)
      .Case("_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt",
            Block2DBuiltin::Read)
      .Case("_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj",
            Block2DBuiltin::ReadTransform)
      .Case("_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj",
            Block2DBuiltin::Write)
      .Default(std::nullopt);
}

static Value castValue(OpBuilder &builder, Location location, Type type,
                       Value value) {
  if (value.getType() == type)
    return value;
  return UnrealizedConversionCastOp::create(builder, location, type, value)
      .getResult(0);
}

static FailureOr<std::array<Value, 2>> getCoordinates(LLVM::CallOp call,
                                                       OpBuilder &builder) {
  LLVM::InsertElementOp yInsert =
      call.getArgOperands()[4].getDefiningOp<LLVM::InsertElementOp>();
  LLVM::InsertElementOp xInsert =
      yInsert ? yInsert.getVector().getDefiningOp<LLVM::InsertElementOp>()
              : LLVM::InsertElementOp();
  if (!xInsert || getConstantIntValue(xInsert.getPosition()) != 0 ||
      getConstantIntValue(yInsert.getPosition()) != 1)
    return call.emitOpError(
               "coordinates must be assembled by inserting x then y"),
           failure();
  return std::array<Value, 2>{xInsert.getValue(), yInsert.getValue()};
}

static FailureOr<LLVM::AllocaOp> getShimAllocation(LLVM::CallOp call) {
  LLVM::AllocaOp allocation =
      call.getArgOperands().back().getDefiningOp<LLVM::AllocaOp>();
  if (!allocation)
    return call.emitOpError("data pointer must be a private alloca ABI shim"),
           failure();
  return allocation;
}

static LogicalResult verifyShimUsers(LLVM::CallOp call,
                                     LLVM::AllocaOp allocation,
                                     Operation *access) {
  if (!llvm::all_of(allocation->getUsers(), [&](Operation *user) {
        return user == call || user == access;
      }))
    return call.emitOpError("private alloca ABI shim has unsupported uses");
  return success();
}

static void eraseCoordinates(LLVM::CallOp call) {
  LLVM::InsertElementOp yInsert =
      call.getArgOperands()[4].getDefiningOp<LLVM::InsertElementOp>();
  LLVM::InsertElementOp xInsert =
      yInsert ? yInsert.getVector().getDefiningOp<LLVM::InsertElementOp>()
              : LLVM::InsertElementOp();
  call.erase();
  if (yInsert && yInsert->use_empty())
    yInsert.erase();
  if (xInsert && xInsert->use_empty())
    xInsert.erase();
}

static LogicalResult canonicalizeCall(LLVM::CallOp call,
                                      Block2DBuiltin builtin) {
  bool prefetch = builtin == Block2DBuiltin::Prefetch;
  bool read = builtin == Block2DBuiltin::Read ||
              builtin == Block2DBuiltin::ReadTransform;
  bool transform = builtin == Block2DBuiltin::ReadTransform;
  if (call.getArgOperands().size() != (prefetch ? 5 : 6))
    return call.emitOpError("has an unexpected block2D builtin signature");

  OpBuilder builder(call);
  FailureOr<std::array<Value, 2>> coordinates = getCoordinates(call, builder);
  if (failed(coordinates))
    return failure();
  MLIRContext *context = call.getContext();
  FunctionOpInterface function = call->getParentOfType<FunctionOpInterface>();
  IntegerAttr simdWidth = function ? function->getAttrOfType<IntegerAttr>(
                                         xw::XWDialect::getSimdWidthAttrName())
                                   : IntegerAttr();
  if (!simdWidth)
    return call.emitOpError("requires an enclosing xw.simd_width");
  Value base = castValue(
      builder, call.getLoc(),
      xw::PtrType::get(context, xw::GlobalAddressSpaceAttr::get(context)),
      call.getArgOperands()[0]);
  std::array<Value, 3> surface = {
      call.getArgOperands()[1], call.getArgOperands()[2],
      call.getArgOperands()[3]};
  Type tokenType = xw::MemTokenType::get(context);
  constexpr int64_t blockWidth = 16;
  int64_t elementBits = builtin == Block2DBuiltin::Write ? 32 : 16;
  int64_t blockHeight = transform ? 16 : 8;

  if (prefetch) {
    xw::Block2DPrefetchOp operation = xw::Block2DPrefetchOp::create(
        builder, call.getLoc(), tokenType, base, surface[0], surface[1],
        surface[2], (*coordinates)[0], (*coordinates)[1], elementBits,
        blockWidth, blockHeight, 1, false, false, Value());
    operation->setDiscardableAttrs(call->getDiscardableAttrDictionary());
    eraseCoordinates(call);
    return success();
  }

  FailureOr<LLVM::AllocaOp> allocation = getShimAllocation(call);
  if (failed(allocation))
    return failure();
  if (read) {
    LLVM::LoadOp load;
    for (Operation *user : (*allocation)->getUsers())
      if (user != call)
        load = dyn_cast<LLVM::LoadOp>(user);
    if (!load || failed(verifyShimUsers(call, *allocation, load)))
      return call.emitOpError("read shim requires exactly one load");
    if (call->getBlock() != load->getBlock() || !call->isBeforeInBlock(load))
      return call.emitOpError("read shim load must follow the builtin call");
    Type resultType =
        xw::SimdType::get(context, load.getType(), simdWidth.getInt());
    xw::Block2DReadOp operation = xw::Block2DReadOp::create(
        builder, call.getLoc(), resultType, tokenType, base, surface[0],
        surface[1], surface[2], (*coordinates)[0], (*coordinates)[1],
        elementBits, blockWidth, blockHeight, 1, false, transform, Value());
    operation->setDiscardableAttrs(call->getDiscardableAttrDictionary());
    Value replacement = castValue(builder, load.getLoc(), load.getType(),
                                  operation.getValue());
    load.replaceAllUsesWith(replacement);
    load.erase();
  } else {
    LLVM::StoreOp store;
    for (Operation *user : (*allocation)->getUsers())
      if (user != call)
        store = dyn_cast<LLVM::StoreOp>(user);
    if (!store || failed(verifyShimUsers(call, *allocation, store)))
      return call.emitOpError("write shim requires exactly one store");
    if (call->getBlock() != store->getBlock() || !store->isBeforeInBlock(call))
      return call.emitOpError("write shim store must precede the builtin call");
    Value data = castValue(
        builder, call.getLoc(),
        xw::SimdType::get(context, store.getValue().getType(),
                          simdWidth.getInt()),
        store.getValue());
    xw::Block2DWriteOp operation = xw::Block2DWriteOp::create(
        builder, call.getLoc(), tokenType, data, base, surface[0], surface[1],
        surface[2], (*coordinates)[0], (*coordinates)[1], elementBits,
        blockWidth, blockHeight, 1, false, false, Value());
    operation->setDiscardableAttrs(call->getDiscardableAttrDictionary());
    store.erase();
  }
  eraseCoordinates(call);
  (*allocation).erase();
  return success();
}

struct CanonicalizeBlock2DABI final
    : inter::impl::CanonicalizeBlock2DABIBase<CanonicalizeBlock2DABI> {
  void runOnOperation() override {
    SmallVector<LLVM::CallOp> calls;
    getOperation().walk([&](LLVM::CallOp call) {
      if (call.getCallee() && classifyBuiltin(*call.getCallee()))
        calls.push_back(call);
    });
    for (LLVM::CallOp call : calls) {
      std::optional<Block2DBuiltin> builtin = classifyBuiltin(*call.getCallee());
      if (failed(canonicalizeCall(call, *builtin)))
        return signalPassFailure();
    }
    for (LLVM::LLVMFuncOp declaration :
         llvm::make_early_inc_range(getOperation().getOps<LLVM::LLVMFuncOp>()))
      if (declaration.isExternal() && classifyBuiltin(declaration.getName()) &&
          SymbolTable::symbolKnownUseEmpty(declaration, getOperation()))
        declaration.erase();
  }
};

} // namespace
