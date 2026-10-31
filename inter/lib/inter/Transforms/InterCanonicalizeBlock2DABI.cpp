#include "inter/Dialect/Inter/IR/XW.h"

#include "inter/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

#include <array>

namespace inter {
#define GEN_PASS_DEF_CANONICALIZEBLOCK2DABI
#define GEN_PASS_DEF_CANONICALIZEDPASBUILTIN
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
      .Case("_Z52intel_sub_group_2d_block_read_transform_16b_"
            "16r16x1cPU3AS1viiiDv2_iPj",
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

static void collectDescriptorPointers(Value value,
                                      SmallVectorImpl<Value> &pointers) {
  if (LLVM::PtrToIntOp cast = value.getDefiningOp<LLVM::PtrToIntOp>()) {
    pointers.push_back(cast.getArg());
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !isa<LLVM::IntToPtrOp, LLVM::ExtractElementOp,
                         LLVM::InsertElementOp, LLVM::BitcastOp>(operation))
    return;
  for (Value operand : operation->getOperands())
    collectDescriptorPointers(operand, pointers);
}

static Value getBlock2DBase(Value value) {
  if (!value.getDefiningOp<LLVM::IntToPtrOp>())
    return value;
  SmallVector<Value> pointers;
  collectDescriptorPointers(value, pointers);
  if (pointers.empty() || !llvm::all_of(pointers, [&](Value pointer) {
        return pointer == pointers.front();
      }))
    return value;
  return pointers.front();
}

static void eraseDeadDescriptorChain(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation || !operation->use_empty() ||
      !isa<LLVM::IntToPtrOp, LLVM::ExtractElementOp, LLVM::InsertElementOp,
           LLVM::BitcastOp, LLVM::PtrToIntOp>(operation))
    return;
  SmallVector<Value> operands(operation->getOperands());
  operation->erase();
  for (Value operand : operands)
    eraseDeadDescriptorChain(operand);
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
  Value seed = xInsert ? xInsert.getVector() : Value();
  call.erase();
  if (yInsert && yInsert->use_empty())
    yInsert.erase();
  if (xInsert && xInsert->use_empty())
    xInsert.erase();
  if (LLVM::UndefOp undef =
          seed ? seed.getDefiningOp<LLVM::UndefOp>() : LLVM::UndefOp();
      undef && undef->use_empty())
    undef.erase();
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
  Value originalBase = call.getArgOperands()[0];
  Value base = castValue(
      builder, call.getLoc(),
      xw::PtrType::get(context, xw::GlobalAddressSpaceAttr::get(context)),
      getBlock2DBase(originalBase));
  std::array<Value, 3> surface = {call.getArgOperands()[1],
                                  call.getArgOperands()[2],
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
    eraseDeadDescriptorChain(originalBase);
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
    Value replacement =
        castValue(builder, load.getLoc(), load.getType(), operation.getValue());
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
    Value data =
        castValue(builder, call.getLoc(),
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
  eraseDeadDescriptorChain(originalBase);
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
      std::optional<Block2DBuiltin> builtin =
          classifyBuiltin(*call.getCallee());
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

struct DpasBuiltin {
  xw::DpasPrecision aPrecision;
  xw::DpasPrecision bPrecision;
  int64_t k;
};

static std::optional<DpasBuiltin> classifyDpasBuiltin(StringRef symbol) {
  StringRef mangled = symbol;
  if (!mangled.consume_front("_Z"))
    return std::nullopt;
  size_t lengthDigits = mangled.find_if_not(llvm::isDigit);
  size_t nameLength;
  if (lengthDigits == 0 ||
      mangled.take_front(lengthDigits).getAsInteger(10, nameLength))
    return std::nullopt;
  mangled = mangled.drop_front(lengthDigits);
  if (nameLength > mangled.size())
    return std::nullopt;
  StringRef name = mangled.take_front(nameLength);
  StringRef prefix = "intel_sub_group_";
  StringRef matrix = "_matrix_mad_k";
  if (!name.consume_front(prefix))
    return std::nullopt;
  size_t matrixPosition = name.find(matrix);
  if (matrixPosition == StringRef::npos)
    return std::nullopt;
  StringRef precisions = name.take_front(matrixPosition);
  auto [aName, bName] = precisions.split('_');
  auto parsePrecision = [](StringRef name) -> std::optional<xw::DpasPrecision> {
    return StringSwitch<std::optional<xw::DpasPrecision>>(name)
        .Case("f16", xw::DpasPrecision::F16)
        .Case("bf16", xw::DpasPrecision::BF16)
        .Default(std::nullopt);
  };
  std::optional<xw::DpasPrecision> a = parsePrecision(aName);
  std::optional<xw::DpasPrecision> b = parsePrecision(bName);
  StringRef kText = name.drop_front(matrixPosition + matrix.size());
  int64_t k;
  if (!a || !b || kText.empty() || kText.getAsInteger(10, k) || k <= 0)
    return std::nullopt;
  return DpasBuiltin{*a, *b, k};
}

static Value stripStorageBitcasts(Value value) {
  while (true) {
    if (LLVM::BitcastOp bitcast = value.getDefiningOp<LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (UnrealizedConversionCastOp cast =
            value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 && cast->getNumResults() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    return value;
  }
}

static void eraseDeadBitcasts(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<LLVM::BitcastOp, UnrealizedConversionCastOp>(operation) ||
        !operation->use_empty())
      return;
    Value source = operation->getOperand(0);
    operation->erase();
    value = source;
  }
}

struct CanonicalizeDpasBuiltin final
    : inter::impl::CanonicalizeDpasBuiltinBase<CanonicalizeDpasBuiltin> {
  void runOnOperation() override {
    SmallVector<LLVM::CallOp> calls;
    getOperation().walk([&](LLVM::CallOp call) {
      if (call.getCallee() && classifyDpasBuiltin(*call.getCallee()))
        calls.push_back(call);
    });
    for (LLVM::CallOp call : calls) {
      if (call.getArgOperands().size() != 3 || call->getNumResults() != 1) {
        call.emitOpError("has an unexpected DPAS builtin signature");
        return signalPassFailure();
      }
      DpasBuiltin builtin = *classifyDpasBuiltin(*call.getCallee());
      FunctionOpInterface function =
          call->getParentOfType<FunctionOpInterface>();
      IntegerAttr width = function ? function->getAttrOfType<IntegerAttr>(
                                         xw::XWDialect::getSimdWidthAttrName())
                                   : IntegerAttr();
      if (!width) {
        call.emitOpError("requires xw.simd_width");
        return signalPassFailure();
      }
      OpBuilder builder(call);
      std::array<Value, 3> original = {call.getArgOperands()[0],
                                       call.getArgOperands()[1],
                                       call.getArgOperands()[2]};
      Value a = stripStorageBitcasts(original[0]);
      Value b = stripStorageBitcasts(original[1]);
      Value acc = stripStorageBitcasts(original[2]);
      auto packetType = [&](Value value) -> Type {
        if (isa<xw::SimdType>(value.getType()))
          return value.getType();
        return xw::SimdType::get(call.getContext(), value.getType(),
                                 width.getInt());
      };
      Type aType = packetType(a);
      Type bType = packetType(b);
      Type accType = packetType(acc);
      Type resultType =
          xw::SimdType::get(call.getContext(), call.getType(0), width.getInt());
      a = castValue(builder, call.getLoc(), aType, a);
      b = castValue(builder, call.getLoc(), bType, b);
      acc = castValue(builder, call.getLoc(), accType, acc);
      VectorType resultPacket = dyn_cast<VectorType>(call.getType(0));
      if (!resultPacket || resultPacket.getRank() != 1 ||
          resultPacket.isScalable()) {
        call.emitOpError("result must be a fixed 1-D vector packet");
        return signalPassFailure();
      }
      int64_t operandsPerDword = 2;
      if (builtin.k % operandsPerDword != 0) {
        call.emitOpError("K is incompatible with the source precision");
        return signalPassFailure();
      }
      xw::DpasOp dpas = xw::DpasOp::create(
          builder, call.getLoc(), resultType, a, b, acc,
          xw::DpasPrecisionAttr::get(call.getContext(), builtin.aPrecision),
          xw::DpasPrecisionAttr::get(call.getContext(), builtin.bPrecision),
          builder.getI64IntegerAttr(builtin.k),
          builder.getI64IntegerAttr(builtin.k / operandsPerDword),
          builder.getI64IntegerAttr(resultPacket.getNumElements()));
      Value replacement =
          castValue(builder, call.getLoc(), call.getType(0), dpas.getResult());
      call.getResult().replaceAllUsesWith(replacement);
      call.erase();
      for (Value value : original)
        eraseDeadBitcasts(value);
    }
    for (LLVM::LLVMFuncOp declaration :
         llvm::make_early_inc_range(getOperation().getOps<LLVM::LLVMFuncOp>()))
      if (declaration.isExternal() &&
          classifyDpasBuiltin(declaration.getName()) &&
          SymbolTable::symbolKnownUseEmpty(declaration, getOperation()))
        declaration.erase();
  }
};

} // namespace
