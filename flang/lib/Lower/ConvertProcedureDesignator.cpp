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
    aiir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
          proc.GetSpecificIntrinsic()) {
    aiir::FunctionType signature =
        Fortran::lower::translateSignature(proc, converter);
    // Intrinsic lowering is based on the generic name, so retrieve it here in
    // case it is different from the specific name. The type of the specific
    // intrinsic is retained in the signature.
    std::string genericName =
        converter.getFoldingContext().intrinsics().GetGenericIntrinsicName(
            intrinsic->name);
    aiir::SymbolRefAttr symbolRefAttr =
        fir::getUnrestrictedIntrinsicSymbolRefAttr(builder, loc, genericName,
                                                   signature);
    aiir::Value funcPtr =
        fir::AddrOfOp::create(builder, loc, signature, symbolRefAttr);
    return funcPtr;
  }
  const Fortran::semantics::Symbol *symbol = proc.GetSymbol();
  assert(symbol && "expected symbol in ProcedureDesignator");
  aiir::Value funcPtr;
  aiir::Value funcPtrResultLength;
  if (Fortran::semantics::IsDummy(*symbol)) {
    Fortran::lower::SymbolBox val = symMap.lookupSymbol(*symbol);
    assert(val && "Dummy procedure not in symbol map");
    funcPtr = val.getAddr();
    if (fir::isCharacterProcedureTuple(funcPtr.getType(),
                                       /*acceptRawFunc=*/false))
      std::tie(funcPtr, funcPtrResultLength) =
          fir::factory::extractCharacterProcedureTuple(builder, loc, funcPtr);
  } else {
    aiir::func::FuncOp func =
        Fortran::lower::getOrDeclareFunction(proc, converter);
    aiir::SymbolRefAttr nameAttr = builder.getSymbolRefAttr(func.getSymName());
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
        aiir::Value rawLen = fir::getBase(
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
    aiir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Symbol &procComponentSym, aiir::Value base,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::FortranVariableFlagsAttr attributes =
      Fortran::lower::translateSymbolAttributes(builder.getContext(),
                                                procComponentSym);
  /// Passed argument may be a descriptor. This is a scalar reference, so the
  /// base address can be directly addressed.
  if (aiir::isa<fir::BaseBoxType>(base.getType()))
    base = fir::BoxAddrOp::create(builder, loc, base);
  std::string fieldName = converter.getRecordTypeFieldName(procComponentSym);
  auto recordType =
      aiir::cast<fir::RecordType>(hlfir::getFortranElementType(base.getType()));
  aiir::Type fieldType = recordType.getType(fieldName);
  // Note: semantics turns x%p() into x%t%p() when the procedure pointer
  // component is part of parent component t.
  if (!fieldType)
    TODO(loc, "passing type bound procedure (extension)");
  aiir::Type designatorType = fir::ReferenceType::get(fieldType);
  aiir::Value compRef = hlfir::DesignateOp::create(
      builder, loc, designatorType, base, fieldName,
      /*compShape=*/aiir::Value{}, hlfir::DesignateOp::Subscripts{},
      /*substring=*/aiir::ValueRange{},
      /*complexPart=*/std::nullopt,
      /*shape=*/aiir::Value{}, /*typeParams=*/aiir::ValueRange{}, attributes);
  return hlfir::EntityWithAttributes{compRef};
}

static hlfir::EntityWithAttributes convertProcedurePointerComponent(
    aiir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::Component &procComponent,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::ExtendedValue baseExv = Fortran::lower::convertDataRefToValue(
      loc, converter, procComponent.base(), symMap, stmtCtx);
  aiir::Value base = fir::getBase(baseExv);
  const Fortran::semantics::Symbol &procComponentSym =
      procComponent.GetLastSymbol();
  return designateProcedurePointerComponent(loc, converter, procComponentSym,
                                            base, symMap, stmtCtx);
}

hlfir::EntityWithAttributes Fortran::lower::convertProcedureDesignatorToHLFIR(
    aiir::Location loc, Fortran::lower::AbstractConverter &converter,
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
  // tuple<fir.boxbroc, len> so that it can be returned as a single aiir::Value.
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  aiir::Value funcAddr = fir::getBase(procExv);
  if (!aiir::isa<fir::BoxProcType>(funcAddr.getType())) {
    aiir::Type boxTy =
        Fortran::lower::getUntypedBoxProcType(&converter.getAIIRContext());
    if (auto host = Fortran::lower::argumentHostAssocs(converter, funcAddr))
      funcAddr = fir::EmboxProcOp::create(
          builder, loc, boxTy, llvm::ArrayRef<aiir::Value>{funcAddr, host});
    else
      funcAddr = fir::EmboxProcOp::create(builder, loc, boxTy, funcAddr);
  }

  aiir::Value res = procExv.match(
      [&](const fir::CharBoxValue &box) -> aiir::Value {
        aiir::Type tupleTy =
            fir::factory::getCharacterProcedureTupleType(funcAddr.getType());
        return fir::factory::createCharacterProcedureTuple(
            builder, loc, tupleTy, funcAddr, box.getLen());
      },
      [funcAddr](const auto &) { return funcAddr; });
  return hlfir::EntityWithAttributes{res};
}

aiir::Value Fortran::lower::convertProcedureDesignatorInitialTarget(
    Fortran::lower::AbstractConverter &converter, aiir::Location loc,
    const Fortran::semantics::Symbol &sym) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::StatementContext stmtCtx;
  Fortran::evaluate::ProcedureDesignator proc(sym);
  auto procVal{Fortran::lower::convertProcedureDesignatorToHLFIR(
      loc, converter, proc, globalOpSymMap, stmtCtx)};
  return fir::getBase(Fortran::lower::convertToAddress(
      loc, converter, procVal, stmtCtx, procVal.getType()));
}

aiir::Value Fortran::lower::derefPassProcPointerComponent(
    aiir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc, aiir::Value passedArg,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  const Fortran::semantics::Symbol *procComponentSym = proc.GetSymbol();
  assert(procComponentSym &&
         "failed to retrieve pointer procedure component symbol");
  hlfir::EntityWithAttributes pointerComp = designateProcedurePointerComponent(
      loc, converter, *procComponentSym, passedArg, symMap, stmtCtx);
  return fir::LoadOp::create(converter.getFirOpBuilder(), loc, pointerComp);
}
