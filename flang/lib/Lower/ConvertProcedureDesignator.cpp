//===- ConvertProcedureDesignator.cpp -- Procedure Designator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertProcedureDesignator.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

static bool areAllSymbolsInExprMapped(const Fortran::evaluate::ExtentExpr &expr,
                                      Fortran::lower::SymMap &symMap) {
  for (const auto &sym : Fortran::evaluate::CollectSymbols(expr))
    if (!symMap.lookupSymbol(sym))
      return false;
  return true;
}

fir::ExtendedValue Fortran::lower::convertProcedureDesignator(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
          proc.GetSpecificIntrinsic()) {
    mlir::FunctionType signature =
        Fortran::lower::translateSignature(proc, converter);
    // Intrinsic lowering is based on the generic name, so retrieve it here in
    // case it is different from the specific name. The type of the specific
    // intrinsic is retained in the signature.
    std::string genericName =
        converter.getFoldingContext().intrinsics().GetGenericIntrinsicName(
            intrinsic->name);
    mlir::SymbolRefAttr symbolRefAttr =
        fir::getUnrestrictedIntrinsicSymbolRefAttr(builder, loc, genericName,
                                                   signature);
    mlir::Value funcPtr =
        fir::AddrOfOp::create(builder, loc, signature, symbolRefAttr);
    return funcPtr;
  }
  const Fortran::semantics::Symbol *symbol = proc.GetSymbol();
  assert(symbol && "expected symbol in ProcedureDesignator");
  mlir::Value funcPtr;
  mlir::Value funcPtrResultLength;
  if (Fortran::semantics::IsDummy(*symbol)) {
    Fortran::lower::SymbolBox val = symMap.lookupSymbol(*symbol);
    assert(val && "Dummy procedure not in symbol map");
    funcPtr = val.getAddr();
    if (fir::isCharacterProcedureTuple(funcPtr.getType(),
                                       /*acceptRawFunc=*/false))
      std::tie(funcPtr, funcPtrResultLength) =
          fir::factory::extractCharacterProcedureTuple(builder, loc, funcPtr);
  } else {
    mlir::func::FuncOp func =
        Fortran::lower::getOrDeclareFunction(proc, converter);
    mlir::SymbolRefAttr nameAttr = builder.getSymbolRefAttr(func.getSymName());
    funcPtr =
        fir::AddrOfOp::create(builder, loc, func.getFunctionType(), nameAttr);
  }
  if (Fortran::lower::mustPassLengthWithDummyProcedure(proc, converter)) {
    // The result length, if available here, must be propagated along the
    // procedure address so that call sites where the result length is assumed
    // can retrieve the length.
    Fortran::evaluate::DynamicType resultType = proc.GetType().value();
    if (const auto &lengthExpr = resultType.GetCharLength()) {
      // The length expression may refer to dummy argument symbols that are
      // meaningless without any actual arguments. Leave the length as
      // unknown in that case, it be resolved on the call site
      // with the actual arguments.
      if (areAllSymbolsInExprMapped(*lengthExpr, symMap)) {
        mlir::Value rawLen = fir::getBase(
            converter.genExprValue(toEvExpr(*lengthExpr), stmtCtx));
        // F2018 7.4.4.2 point 5.
        funcPtrResultLength =
            fir::factory::genMaxWithZero(builder, loc, rawLen);
      }
    }
    // The caller of the function pointer will have to allocate
    // the function result with the character length specified
    // by the boxed value. If the result length cannot be
    // computed statically, set it to zero (we used to use -1,
    // but this could cause assertions in LLVM after inlining
    // exposed alloca of size -1).
    if (!funcPtrResultLength)
      funcPtrResultLength = builder.createIntegerConstant(
          loc, builder.getCharacterLengthType(), 0);
    return fir::CharBoxValue{funcPtr, funcPtrResultLength};
  }
  return funcPtr;
}

static hlfir::EntityWithAttributes designateProcedurePointerComponent(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Symbol &procComponentSym, mlir::Value base,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::FortranVariableFlagsAttr attributes =
      Fortran::lower::translateSymbolAttributes(builder.getContext(),
                                                procComponentSym);
  /// Passed argument may be a descriptor. This is a scalar reference, so the
  /// base address can be directly addressed.
  if (mlir::isa<fir::BaseBoxType>(base.getType()))
    base = fir::BoxAddrOp::create(builder, loc, base);
  std::string fieldName = converter.getRecordTypeFieldName(procComponentSym);
  auto recordType =
      mlir::cast<fir::RecordType>(hlfir::getFortranElementType(base.getType()));
  mlir::Type fieldType = recordType.getType(fieldName);
  // Note: semantics turns x%p() into x%t%p() when the procedure pointer
  // component is part of parent component t.
  if (!fieldType)
    TODO(loc, "passing type bound procedure (extension)");
  mlir::Type designatorType = fir::ReferenceType::get(fieldType);
  mlir::Value compRef = hlfir::DesignateOp::create(
      builder, loc, designatorType, base, fieldName,
      /*compShape=*/mlir::Value{}, hlfir::DesignateOp::Subscripts{},
      /*substring=*/mlir::ValueRange{},
      /*complexPart=*/std::nullopt,
      /*shape=*/mlir::Value{}, /*typeParams=*/mlir::ValueRange{}, attributes);
  return hlfir::EntityWithAttributes{compRef};
}

static hlfir::EntityWithAttributes convertProcedurePointerComponent(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Component &procComponent,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::ExtendedValue baseExv = Fortran::lower::convertDataRefToValue(
      loc, converter, procComponent.base(), symMap, stmtCtx);
  mlir::Value base = fir::getBase(baseExv);
  const Fortran::semantics::Symbol &procComponentSym =
      procComponent.GetLastSymbol();
  return designateProcedurePointerComponent(loc, converter, procComponentSym,
                                            base, symMap, stmtCtx);
}

hlfir::EntityWithAttributes Fortran::lower::convertProcedureDesignatorToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  const auto *sym = proc.GetSymbol();
  if (sym) {
    if (sym->GetUltimate().attrs().test(Fortran::semantics::Attr::INTRINSIC))
      TODO(loc, "Procedure pointer with intrinsic target.");
    if (std::optional<fir::FortranVariableOpInterface> varDef =
            symMap.lookupVariableDefinition(*sym))
      return *varDef;
  }

  if (const Fortran::evaluate::Component *procComponent = proc.GetComponent())
    return convertProcedurePointerComponent(loc, converter, *procComponent,
                                            symMap, stmtCtx);

  fir::ExtendedValue procExv =
      convertProcedureDesignator(loc, converter, proc, symMap, stmtCtx);
  // Directly package the procedure address as a fir.boxproc or
  // tuple<fir.boxbroc, len> so that it can be returned as a single mlir::Value.
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value funcAddr = fir::getBase(procExv);
  if (!mlir::isa<fir::BoxProcType>(funcAddr.getType())) {
    mlir::Type boxTy =
        Fortran::lower::getUntypedBoxProcType(&converter.getMLIRContext());
    if (auto host = Fortran::lower::argumentHostAssocs(converter, funcAddr))
      funcAddr = fir::EmboxProcOp::create(
          builder, loc, boxTy, llvm::ArrayRef<mlir::Value>{funcAddr, host});
    else
      funcAddr = fir::EmboxProcOp::create(builder, loc, boxTy, funcAddr);
  }

  mlir::Value res = procExv.match(
      [&](const fir::CharBoxValue &box) -> mlir::Value {
        mlir::Type tupleTy =
            fir::factory::getCharacterProcedureTupleType(funcAddr.getType());
        return fir::factory::createCharacterProcedureTuple(
            builder, loc, tupleTy, funcAddr, box.getLen());
      },
      [funcAddr](const auto &) { return funcAddr; });
  return hlfir::EntityWithAttributes{res};
}

mlir::Value Fortran::lower::convertProcedureDesignatorInitialTarget(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::StatementContext stmtCtx;
  Fortran::evaluate::ProcedureDesignator proc(sym);
  auto procVal{Fortran::lower::convertProcedureDesignatorToHLFIR(
      loc, converter, proc, globalOpSymMap, stmtCtx)};
  return fir::getBase(Fortran::lower::convertToAddress(
      loc, converter, procVal, stmtCtx, procVal.getType()));
}

mlir::Value Fortran::lower::derefPassProcPointerComponent(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc, mlir::Value passedArg,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  const Fortran::semantics::Symbol *procComponentSym = proc.GetSymbol();
  assert(procComponentSym &&
         "failed to retrieve pointer procedure component symbol");
  hlfir::EntityWithAttributes pointerComp = designateProcedurePointerComponent(
      loc, converter, *procComponentSym, passedArg, symMap, stmtCtx);
  return converter.getFirOpBuilder().create<fir::LoadOp>(loc, pointerComp);
}
