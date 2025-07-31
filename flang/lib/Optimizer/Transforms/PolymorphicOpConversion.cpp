//===-- PolymorphicOpConversion.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/BuiltinModules.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/derived-api.h"
#include "flang/Semantics/runtime-type-info.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

namespace fir {
#define GEN_PASS_DEF_POLYMORPHICOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;

// Reconstruct binding tables for dynamic dispatch.
using BindingTable = llvm::DenseMap<llvm::StringRef, unsigned>;
using BindingTables = llvm::DenseMap<llvm::StringRef, BindingTable>;

static std::string getTypeDescriptorTypeName() {
  llvm::SmallVector<llvm::StringRef, 1> modules = {
      Fortran::semantics::typeInfoBuiltinModule};
  return fir::NameUniquer::doType(modules, /*proc=*/{}, /*blockId=*/0,
                                  Fortran::semantics::typeDescriptorTypeName,
                                  /*kinds=*/{});
}

static std::optional<mlir::Type>
buildBindingTables(BindingTables &bindingTables, mlir::ModuleOp mod) {

  std::optional<mlir::Type> typeDescriptorType;
  std::string typeDescriptorTypeName = getTypeDescriptorTypeName();
  // The binding tables are defined in FIR after lowering inside fir.type_info
  // operations. Go through each binding tables and store the procedure name and
  // binding index for later use by the fir.dispatch conversion pattern.
  for (auto typeInfo : mod.getOps<fir::TypeInfoOp>()) {
    if (!typeDescriptorType && typeInfo.getSymName() == typeDescriptorTypeName)
      typeDescriptorType = typeInfo.getType();
    unsigned bindingIdx = 0;
    BindingTable bindings;
    if (typeInfo.getDispatchTable().empty()) {
      bindingTables[typeInfo.getSymName()] = bindings;
      continue;
    }
    for (auto dtEntry :
         typeInfo.getDispatchTable().front().getOps<fir::DTEntryOp>()) {
      bindings[dtEntry.getMethod()] = bindingIdx;
      ++bindingIdx;
    }
    bindingTables[typeInfo.getSymName()] = bindings;
  }
  return typeDescriptorType;
}

namespace {

/// SelectTypeOp converted to an if-then-else chain
///
/// This lowers the test conditions to calls into the runtime.
class SelectTypeConv : public OpConversionPattern<fir::SelectTypeOp> {
public:
  using OpConversionPattern<fir::SelectTypeOp>::OpConversionPattern;

  SelectTypeConv(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<fir::SelectTypeOp>(ctx) {}

  llvm::LogicalResult
  matchAndRewrite(fir::SelectTypeOp selectType, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

private:
  // Generate comparison of type descriptor addresses.
  mlir::Value genTypeDescCompare(mlir::Location loc, mlir::Value selector,
                                 mlir::Type ty, mlir::ModuleOp mod,
                                 mlir::PatternRewriter &rewriter) const;

  llvm::LogicalResult genTypeLadderStep(mlir::Location loc,
                                        mlir::Value selector,
                                        mlir::Attribute attr, mlir::Block *dest,
                                        std::optional<mlir::ValueRange> destOps,
                                        mlir::ModuleOp mod,
                                        mlir::PatternRewriter &rewriter,
                                        fir::KindMapping &kindMap) const;

  llvm::SmallSet<llvm::StringRef, 4> collectAncestors(fir::TypeInfoOp dt,
                                                      mlir::ModuleOp mod) const;
};

/// Lower `fir.dispatch` operation. A virtual call to a method in a dispatch
/// table.
struct DispatchOpConv : public OpConversionPattern<fir::DispatchOp> {
  using OpConversionPattern<fir::DispatchOp>::OpConversionPattern;

  DispatchOpConv(mlir::MLIRContext *ctx, const BindingTables &bindingTables,
                 std::optional<mlir::Type> typeDescriptorType)
      : mlir::OpConversionPattern<fir::DispatchOp>(ctx),
        bindingTables(bindingTables), typeDescriptorType{typeDescriptorType} {}

  llvm::LogicalResult
  matchAndRewrite(fir::DispatchOp dispatch, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = dispatch.getLoc();

    if (bindingTables.empty())
      return emitError(loc) << "no binding tables found";

    // Get derived type information.
    mlir::Type declaredType =
        fir::getDerivedType(dispatch.getObject().getType().getEleTy());
    assert(mlir::isa<fir::RecordType>(declaredType) && "expecting fir.type");
    auto recordType = mlir::dyn_cast<fir::RecordType>(declaredType);

    // Lookup for the binding table.
    auto bindingsIter = bindingTables.find(recordType.getName());
    if (bindingsIter == bindingTables.end())
      return emitError(loc)
             << "cannot find binding table for " << recordType.getName();

    // Lookup for the binding.
    const BindingTable &bindingTable = bindingsIter->second;
    auto bindingIter = bindingTable.find(dispatch.getMethod());
    if (bindingIter == bindingTable.end())
      return emitError(loc)
             << "cannot find binding for " << dispatch.getMethod();
    unsigned bindingIdx = bindingIter->second;

    mlir::Value passedObject = dispatch.getObject();

    if (!typeDescriptorType)
      return emitError(loc) << "cannot find " << getTypeDescriptorTypeName()
                            << " fir.type_info that is required to get the "
                               "related builtin type and lower fir.dispatch";
    mlir::Type typeDescTy = *typeDescriptorType;

    // clang-format off
    // Before:
    //   fir.dispatch "proc1"(%11 :
    //   !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

    // After:
    //   %12 = fir.box_tdesc %11 : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.tdesc<none>
    //   %13 = fir.convert %12 : (!fir.tdesc<none>) -> !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype>>
    //   %14 = fir.field_index binding, !fir.type<_QM__fortran_type_infoTderivedtype>
    //   %15 = fir.coordinate_of %13, %14 : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding>>>>>
    //   %bindings = fir.load %15 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding>>>>>
    //   %16 = fir.box_addr %bindings : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding>>>>) -> !fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding>>>
    //   %17 = fir.coordinate_of %16, %c0 : (!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding>>>, index) -> !fir.ref<!fir.type<_QM__fortran_type_infoTbinding>>
    //   %18 = fir.field_index proc, !fir.type<_QM__fortran_type_infoTbinding>
    //   %19 = fir.coordinate_of %17, %18 : (!fir.ref<!fir.type<_QM__fortran_type_infoTbinding>>, !fir.field) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr>>
    //   %20 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr>
    //   %21 = fir.coordinate_of %19, %20 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr>>, !fir.field) -> !fir.ref<i64>
    //   %22 = fir.load %21 : !fir.ref<i64>
    //   %23 = fir.convert %22 : (i64) -> (() -> ())
    //   fir.call %23()  : () -> ()
    // clang-format on

    // Load the descriptor.
    mlir::Type fieldTy = fir::FieldType::get(rewriter.getContext());
    mlir::Type tdescType =
        fir::TypeDescType::get(mlir::NoneType::get(rewriter.getContext()));
    mlir::Value boxDesc =
        fir::BoxTypeDescOp::create(rewriter, loc, tdescType, passedObject);
    boxDesc = fir::ConvertOp::create(
        rewriter, loc, fir::ReferenceType::get(typeDescTy), boxDesc);

    // Load the bindings descriptor.
    auto bindingsCompName = Fortran::semantics::bindingDescCompName;
    fir::RecordType typeDescRecTy = mlir::cast<fir::RecordType>(typeDescTy);
    mlir::Value field =
        fir::FieldIndexOp::create(rewriter, loc, fieldTy, bindingsCompName,
                                  typeDescRecTy, mlir::ValueRange{});
    mlir::Type coorTy =
        fir::ReferenceType::get(typeDescRecTy.getType(bindingsCompName));
    mlir::Value bindingBoxAddr =
        fir::CoordinateOp::create(rewriter, loc, coorTy, boxDesc, field);
    mlir::Value bindingBox = fir::LoadOp::create(rewriter, loc, bindingBoxAddr);

    // Load the correct binding.
    mlir::Value bindings = fir::BoxAddrOp::create(rewriter, loc, bindingBox);
    fir::RecordType bindingTy = fir::unwrapIfDerived(
        mlir::cast<fir::BaseBoxType>(bindingBox.getType()));
    mlir::Type bindingAddrTy = fir::ReferenceType::get(bindingTy);
    mlir::Value bindingIdxVal =
        mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                        rewriter.getIndexAttr(bindingIdx));
    mlir::Value bindingAddr = fir::CoordinateOp::create(
        rewriter, loc, bindingAddrTy, bindings, bindingIdxVal);

    // Get the function pointer.
    auto procCompName = Fortran::semantics::procCompName;
    mlir::Value procField = fir::FieldIndexOp::create(
        rewriter, loc, fieldTy, procCompName, bindingTy, mlir::ValueRange{});
    fir::RecordType procTy =
        mlir::cast<fir::RecordType>(bindingTy.getType(procCompName));
    mlir::Type procRefTy = fir::ReferenceType::get(procTy);
    mlir::Value procRef = fir::CoordinateOp::create(rewriter, loc, procRefTy,
                                                    bindingAddr, procField);

    auto addressFieldName = Fortran::lower::builtin::cptrFieldName;
    mlir::Value addressField = fir::FieldIndexOp::create(
        rewriter, loc, fieldTy, addressFieldName, procTy, mlir::ValueRange{});
    mlir::Type addressTy = procTy.getType(addressFieldName);
    mlir::Type addressRefTy = fir::ReferenceType::get(addressTy);
    mlir::Value addressRef = fir::CoordinateOp::create(
        rewriter, loc, addressRefTy, procRef, addressField);
    mlir::Value address = fir::LoadOp::create(rewriter, loc, addressRef);

    // Get the function type.
    llvm::SmallVector<mlir::Type> argTypes;
    for (mlir::Value operand : dispatch.getArgs())
      argTypes.push_back(operand.getType());
    llvm::SmallVector<mlir::Type> resTypes;
    if (!dispatch.getResults().empty())
      resTypes.push_back(dispatch.getResults()[0].getType());

    mlir::Type funTy =
        mlir::FunctionType::get(rewriter.getContext(), argTypes, resTypes);
    mlir::Value funcPtr = fir::ConvertOp::create(rewriter, loc, funTy, address);

    // Make the call.
    llvm::SmallVector<mlir::Value> args{funcPtr};
    args.append(dispatch.getArgs().begin(), dispatch.getArgs().end());
    rewriter.replaceOpWithNewOp<fir::CallOp>(
        dispatch, resTypes, nullptr, args, dispatch.getArgAttrsAttr(),
        dispatch.getResAttrsAttr(), dispatch.getProcedureAttrsAttr());
    return mlir::success();
  }

private:
  BindingTables bindingTables;
  std::optional<mlir::Type> typeDescriptorType;
};

/// Convert FIR structured control flow ops to CFG ops.
class PolymorphicOpConversion
    : public fir::impl::PolymorphicOpConversionBase<PolymorphicOpConversion> {
public:
  llvm::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    return mlir::success();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    mlir::RewritePatternSet patterns(context);

    BindingTables bindingTables;
    std::optional<mlir::Type> typeDescriptorType =
        buildBindingTables(bindingTables, mod);

    patterns.insert<SelectTypeConv>(context);
    patterns.insert<DispatchOpConv>(context, bindingTables, typeDescriptorType);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::affine::AffineDialect,
                           mlir::cf::ControlFlowDialect, FIROpsDialect,
                           mlir::func::FuncDialect>();

    // apply the patterns
    target.addIllegalOp<SelectTypeOp>();
    target.addIllegalOp<DispatchOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to CFG\n");
      signalPassFailure();
    }
  }
};
} // namespace

llvm::LogicalResult SelectTypeConv::matchAndRewrite(
    fir::SelectTypeOp selectType, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  auto typeGuards = selectType.getCases();
  unsigned typeGuardNum = typeGuards.size();
  auto selector = selectType.getSelector();
  auto loc = selectType.getLoc();
  auto mod = selectType.getOperation()->getParentOfType<mlir::ModuleOp>();
  fir::KindMapping kindMap = fir::getKindMapping(mod);

  // Order type guards so the condition and branches are done to respect the
  // Execution of SELECT TYPE construct as described in the Fortran 2018
  // standard 11.1.11.2 point 4.
  // 1. If a TYPE IS type guard statement matches the selector, the block
  //    following that statement is executed.
  // 2. Otherwise, if exactly one CLASS IS type guard statement matches the
  //    selector, the block following that statement is executed.
  // 3. Otherwise, if several CLASS IS type guard statements match the
  //    selector, one of these statements will inevitably specify a type that
  //    is an extension of all the types specified in the others; the block
  //    following that statement is executed.
  // 4. Otherwise, if there is a CLASS DEFAULT type guard statement, the block
  //    following that statement is executed.
  // 5. Otherwise, no block is executed.

  llvm::SmallVector<unsigned> orderedTypeGuards;
  llvm::SmallVector<unsigned> orderedClassIsGuards;
  unsigned defaultGuard = typeGuardNum - 1;

  // The following loop go through the type guards in the fir.select_type
  // operation and sort them into two lists.
  // - All the TYPE IS type guard are added in order to the orderedTypeGuards
  //   list. This list is used at the end to generate the if-then-else ladder.
  // - CLASS IS type guard are added in a separate list. If a CLASS IS type
  //   guard type extends a type already present, the type guard is inserted
  //   before in the list to respect point 3. above. Otherwise it is just
  //   added in order at the end.
  for (unsigned t = 0; t < typeGuardNum; ++t) {
    if (auto a = mlir::dyn_cast<fir::ExactTypeAttr>(typeGuards[t])) {
      orderedTypeGuards.push_back(t);
      continue;
    }

    if (auto a = mlir::dyn_cast<fir::SubclassAttr>(typeGuards[t])) {
      if (auto recTy = mlir::dyn_cast<fir::RecordType>(a.getType())) {
        auto dt = mod.lookupSymbol<fir::TypeInfoOp>(recTy.getName());
        assert(dt && "dispatch table not found");
        llvm::SmallSet<llvm::StringRef, 4> ancestors =
            collectAncestors(dt, mod);
        if (!ancestors.empty()) {
          auto it = orderedClassIsGuards.begin();
          while (it != orderedClassIsGuards.end()) {
            fir::SubclassAttr sAttr =
                mlir::dyn_cast<fir::SubclassAttr>(typeGuards[*it]);
            if (auto ty = mlir::dyn_cast<fir::RecordType>(sAttr.getType())) {
              if (ancestors.contains(ty.getName()))
                break;
            }
            ++it;
          }
          if (it != orderedClassIsGuards.end()) {
            // Parent type is present so place it before.
            orderedClassIsGuards.insert(it, t);
            continue;
          }
        }
      }
      orderedClassIsGuards.push_back(t);
    }
  }
  orderedTypeGuards.append(orderedClassIsGuards);
  orderedTypeGuards.push_back(defaultGuard);
  assert(orderedTypeGuards.size() == typeGuardNum &&
         "ordered type guard size doesn't match number of type guards");

  for (unsigned idx : orderedTypeGuards) {
    auto *dest = selectType.getSuccessor(idx);
    std::optional<mlir::ValueRange> destOps =
        selectType.getSuccessorOperands(operands, idx);
    if (mlir::dyn_cast<mlir::UnitAttr>(typeGuards[idx]))
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
          selectType, dest, destOps.value_or(mlir::ValueRange{}));
    else if (mlir::failed(genTypeLadderStep(loc, selector, typeGuards[idx],
                                            dest, destOps, mod, rewriter,
                                            kindMap)))
      return mlir::failure();
  }
  return mlir::success();
}

llvm::LogicalResult SelectTypeConv::genTypeLadderStep(
    mlir::Location loc, mlir::Value selector, mlir::Attribute attr,
    mlir::Block *dest, std::optional<mlir::ValueRange> destOps,
    mlir::ModuleOp mod, mlir::PatternRewriter &rewriter,
    fir::KindMapping &kindMap) const {
  mlir::Value cmp;
  // TYPE IS type guard comparison are all done inlined.
  if (auto a = mlir::dyn_cast<fir::ExactTypeAttr>(attr)) {
    if (fir::isa_trivial(a.getType()) ||
        mlir::isa<fir::CharacterType>(a.getType())) {
      // For type guard statement with Intrinsic type spec the type code of
      // the descriptor is compared.
      int code = fir::getTypeCode(a.getType(), kindMap);
      if (code == 0)
        return mlir::emitError(loc)
               << "type code unavailable for " << a.getType();
      mlir::Value typeCode = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI8IntegerAttr(code));
      mlir::Value selectorTypeCode = fir::BoxTypeCodeOp::create(
          rewriter, loc, rewriter.getI8Type(), selector);
      cmp = mlir::arith::CmpIOp::create(rewriter, loc,
                                        mlir::arith::CmpIPredicate::eq,
                                        selectorTypeCode, typeCode);
    } else {
      // Flang inline the kind parameter in the type descriptor so we can
      // directly check if the type descriptor addresses are identical for
      // the TYPE IS type guard statement.
      mlir::Value res =
          genTypeDescCompare(loc, selector, a.getType(), mod, rewriter);
      if (!res)
        return mlir::failure();
      cmp = res;
    }
    // CLASS IS type guard statement is done with a runtime call.
  } else if (auto a = mlir::dyn_cast<fir::SubclassAttr>(attr)) {
    // Retrieve the type descriptor from the type guard statement record type.
    assert(mlir::isa<fir::RecordType>(a.getType()) && "expect fir.record type");
    mlir::Value typeDescAddr = fir::TypeDescOp::create(
        rewriter, loc, mlir::TypeAttr::get(a.getType()));
    mlir::Type refNoneType = ReferenceType::get(rewriter.getNoneType());
    mlir::Value typeDesc =
        ConvertOp::create(rewriter, loc, refNoneType, typeDescAddr);

    // Prepare the selector descriptor for the runtime call.
    mlir::Type descNoneTy = fir::BoxType::get(rewriter.getNoneType());
    mlir::Value descSelector =
        ConvertOp::create(rewriter, loc, descNoneTy, selector);

    // Generate runtime call.
    llvm::StringRef fctName = RTNAME_STRING(ClassIs);
    mlir::func::FuncOp callee;
    {
      // Since conversion is done in parallel for each fir.select_type
      // operation, the runtime function insertion must be threadsafe.
      auto runtimeAttr =
          mlir::NamedAttribute(fir::FIROpsDialect::getFirRuntimeAttrName(),
                               mlir::UnitAttr::get(rewriter.getContext()));
      callee =
          fir::createFuncOp(rewriter.getUnknownLoc(), mod, fctName,
                            rewriter.getFunctionType({descNoneTy, refNoneType},
                                                     rewriter.getI1Type()),
                            {runtimeAttr});
    }
    cmp = rewriter
              .create<fir::CallOp>(loc, callee,
                                   mlir::ValueRange{descSelector, typeDesc})
              .getResult(0);
  }

  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock =
      rewriter.createBlock(dest->getParent(), mlir::Region::iterator(dest));
  rewriter.setInsertionPointToEnd(thisBlock);
  if (destOps.has_value())
    mlir::cf::CondBranchOp::create(rewriter, loc, cmp, dest, destOps.value(),
                                   newBlock, mlir::ValueRange{});
  else
    mlir::cf::CondBranchOp::create(rewriter, loc, cmp, dest, newBlock);
  rewriter.setInsertionPointToEnd(newBlock);
  return mlir::success();
}

// Generate comparison of type descriptor addresses.
mlir::Value
SelectTypeConv::genTypeDescCompare(mlir::Location loc, mlir::Value selector,
                                   mlir::Type ty, mlir::ModuleOp mod,
                                   mlir::PatternRewriter &rewriter) const {
  assert(mlir::isa<fir::RecordType>(ty) && "expect fir.record type");
  mlir::Value typeDescAddr =
      fir::TypeDescOp::create(rewriter, loc, mlir::TypeAttr::get(ty));
  mlir::Value selectorTdescAddr = fir::BoxTypeDescOp::create(
      rewriter, loc, typeDescAddr.getType(), selector);
  auto intPtrTy = rewriter.getIndexType();
  auto typeDescInt =
      fir::ConvertOp::create(rewriter, loc, intPtrTy, typeDescAddr);
  auto selectorTdescInt =
      fir::ConvertOp::create(rewriter, loc, intPtrTy, selectorTdescAddr);
  return mlir::arith::CmpIOp::create(rewriter, loc,
                                     mlir::arith::CmpIPredicate::eq,
                                     typeDescInt, selectorTdescInt);
}

llvm::SmallSet<llvm::StringRef, 4>
SelectTypeConv::collectAncestors(fir::TypeInfoOp dt, mlir::ModuleOp mod) const {
  llvm::SmallSet<llvm::StringRef, 4> ancestors;
  while (auto parentName = dt.getIfParentName()) {
    ancestors.insert(*parentName);
    dt = mod.lookupSymbol<fir::TypeInfoOp>(*parentName);
    assert(dt && "parent type info not generated");
  }
  return ancestors;
}
