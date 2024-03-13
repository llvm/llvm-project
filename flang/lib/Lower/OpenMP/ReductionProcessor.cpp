//===-- ReductionProcessor.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "ReductionProcessor.h"

#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> forceByrefReduction(
    "force-byref-reduction",
    llvm::cl::desc("Pass all reduction arguments by reference"),
    llvm::cl::Hidden);

namespace Fortran {
namespace lower {
namespace omp {

ReductionProcessor::ReductionIdentifier ReductionProcessor::getReductionType(
    const Fortran::parser::ProcedureDesignator &pd) {
  auto redType = llvm::StringSwitch<std::optional<ReductionIdentifier>>(
                     ReductionProcessor::getRealName(pd).ToString())
                     .Case("max", ReductionIdentifier::MAX)
                     .Case("min", ReductionIdentifier::MIN)
                     .Case("iand", ReductionIdentifier::IAND)
                     .Case("ior", ReductionIdentifier::IOR)
                     .Case("ieor", ReductionIdentifier::IEOR)
                     .Default(std::nullopt);
  assert(redType && "Invalid Reduction");
  return *redType;
}

ReductionProcessor::ReductionIdentifier ReductionProcessor::getReductionType(
    Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp) {
  switch (intrinsicOp) {
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
    return ReductionIdentifier::ADD;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Subtract:
    return ReductionIdentifier::SUBTRACT;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
    return ReductionIdentifier::MULTIPLY;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::AND:
    return ReductionIdentifier::AND;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV:
    return ReductionIdentifier::EQV;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::OR:
    return ReductionIdentifier::OR;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV:
    return ReductionIdentifier::NEQV;
  default:
    llvm_unreachable("unexpected intrinsic operator in reduction");
  }
}

bool ReductionProcessor::supportedIntrinsicProcReduction(
    const Fortran::parser::ProcedureDesignator &pd) {
  const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(pd)};
  assert(name && "Invalid Reduction Intrinsic.");
  if (!name->symbol->GetUltimate().attrs().test(
          Fortran::semantics::Attr::INTRINSIC))
    return false;
  auto redType = llvm::StringSwitch<bool>(getRealName(name).ToString())
                     .Case("max", true)
                     .Case("min", true)
                     .Case("iand", true)
                     .Case("ior", true)
                     .Case("ieor", true)
                     .Default(false);
  return redType;
}

std::string ReductionProcessor::getReductionName(llvm::StringRef name,
                                                 mlir::Type ty, bool isByRef) {
  ty = fir::unwrapRefType(ty);

  // extra string to distinguish reduction functions for variables passed by
  // reference
  llvm::StringRef byrefAddition{""};
  if (isByRef)
    byrefAddition = "_byref";

  return (llvm::Twine(name) +
          (ty.isIntOrIndex() ? llvm::Twine("_i_") : llvm::Twine("_f_")) +
          llvm::Twine(ty.getIntOrFloatBitWidth()) + byrefAddition)
      .str();
}

std::string ReductionProcessor::getReductionName(
    Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp,
    mlir::Type ty, bool isByRef) {
  std::string reductionName;

  switch (intrinsicOp) {
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
    reductionName = "add_reduction";
    break;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
    reductionName = "multiply_reduction";
    break;
  case Fortran::parser::DefinedOperator::IntrinsicOperator::AND:
    return "and_reduction";
  case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV:
    return "eqv_reduction";
  case Fortran::parser::DefinedOperator::IntrinsicOperator::OR:
    return "or_reduction";
  case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV:
    return "neqv_reduction";
  default:
    reductionName = "other_reduction";
    break;
  }

  return getReductionName(reductionName, ty, isByRef);
}

mlir::Value
ReductionProcessor::getReductionInitValue(mlir::Location loc, mlir::Type type,
                                          ReductionIdentifier redId,
                                          fir::FirOpBuilder &builder) {
  type = fir::unwrapRefType(type);
  assert((fir::isa_integer(type) || fir::isa_real(type) ||
          type.isa<fir::LogicalType>()) &&
         "only integer, logical and real types are currently supported");
  switch (redId) {
  case ReductionIdentifier::MAX: {
    if (auto ty = type.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/true));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t minInt = llvm::APInt::getSignedMinValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, minInt);
  }
  case ReductionIdentifier::MIN: {
    if (auto ty = type.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/false));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t maxInt = llvm::APInt::getSignedMaxValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, maxInt);
  }
  case ReductionIdentifier::IOR: {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, zeroInt);
  }
  case ReductionIdentifier::IEOR: {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, zeroInt);
  }
  case ReductionIdentifier::IAND: {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t allOnInt = llvm::APInt::getAllOnes(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, allOnInt);
  }
  case ReductionIdentifier::ADD:
  case ReductionIdentifier::MULTIPLY:
  case ReductionIdentifier::AND:
  case ReductionIdentifier::OR:
  case ReductionIdentifier::EQV:
  case ReductionIdentifier::NEQV:
    if (type.isa<mlir::FloatType>())
      return builder.create<mlir::arith::ConstantOp>(
          loc, type,
          builder.getFloatAttr(type, (double)getOperationIdentity(redId, loc)));

    if (type.isa<fir::LogicalType>()) {
      mlir::Value intConst = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(),
                                 getOperationIdentity(redId, loc)));
      return builder.createConvert(loc, type, intConst);
    }

    return builder.create<mlir::arith::ConstantOp>(
        loc, type,
        builder.getIntegerAttr(type, getOperationIdentity(redId, loc)));
  case ReductionIdentifier::ID:
  case ReductionIdentifier::USER_DEF_OP:
  case ReductionIdentifier::SUBTRACT:
    TODO(loc, "Reduction of some identifier types is not supported");
  }
  llvm_unreachable("Unhandled Reduction identifier : getReductionInitValue");
}

mlir::Value ReductionProcessor::createScalarCombiner(
    fir::FirOpBuilder &builder, mlir::Location loc, ReductionIdentifier redId,
    mlir::Type type, mlir::Value op1, mlir::Value op2) {
  mlir::Value reductionOp;
  type = fir::unwrapRefType(type);
  switch (redId) {
  case ReductionIdentifier::MAX:
    reductionOp =
        getReductionOperation<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MIN:
    reductionOp =
        getReductionOperation<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::IOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = builder.create<mlir::arith::OrIOp>(loc, op1, op2);
    break;
  case ReductionIdentifier::IEOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = builder.create<mlir::arith::XOrIOp>(loc, op1, op2);
    break;
  case ReductionIdentifier::IAND:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = builder.create<mlir::arith::AndIOp>(loc, op1, op2);
    break;
  case ReductionIdentifier::ADD:
    reductionOp =
        getReductionOperation<mlir::arith::AddFOp, mlir::arith::AddIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MULTIPLY:
    reductionOp =
        getReductionOperation<mlir::arith::MulFOp, mlir::arith::MulIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::AND: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value andiOp = builder.create<mlir::arith::AndIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, andiOp);
    break;
  }
  case ReductionIdentifier::OR: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value oriOp = builder.create<mlir::arith::OrIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, oriOp);
    break;
  }
  case ReductionIdentifier::EQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  case ReductionIdentifier::NEQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  return reductionOp;
}

mlir::omp::ReductionDeclareOp ReductionProcessor::createReductionDecl(
    fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, mlir::Type type, mlir::Location loc,
    bool isByRef) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  auto decl =
      module.lookupSymbol<mlir::omp::ReductionDeclareOp>(reductionOpName);
  if (decl)
    return decl;

  mlir::OpBuilder modBuilder(module.getBodyRegion());
  mlir::Type valTy = fir::unwrapRefType(type);
  if (!isByRef)
    type = valTy;

  decl = modBuilder.create<mlir::omp::ReductionDeclareOp>(loc, reductionOpName,
                                                          type);
  builder.createBlock(&decl.getInitializerRegion(),
                      decl.getInitializerRegion().end(), {type}, {loc});
  builder.setInsertionPointToEnd(&decl.getInitializerRegion().back());

  mlir::Value init = getReductionInitValue(loc, type, redId, builder);
  if (isByRef) {
    mlir::Value alloca = builder.create<fir::AllocaOp>(loc, valTy);
    builder.createStoreWithConvert(loc, init, alloca);
    builder.create<mlir::omp::YieldOp>(loc, alloca);
  } else {
    builder.create<mlir::omp::YieldOp>(loc, init);
  }

  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});

  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);
  mlir::Value outAddr = op1;

  op1 = builder.loadIfRef(loc, op1);
  op2 = builder.loadIfRef(loc, op2);

  mlir::Value reductionOp =
      createScalarCombiner(builder, loc, redId, type, op1, op2);
  if (isByRef) {
    builder.create<fir::StoreOp>(loc, reductionOp, outAddr);
    builder.create<mlir::omp::YieldOp>(loc, outAddr);
  } else {
    builder.create<mlir::omp::YieldOp>(loc, reductionOp);
  }

  return decl;
}

// TODO: By-ref vs by-val reductions are currently toggled for the whole
//       operation (possibly effecting multiple reduction variables).
//       This could cause a problem with openmp target reductions because
//       by-ref trivial types may not be supported.
bool ReductionProcessor::doReductionByRef(
    const llvm::SmallVectorImpl<mlir::Value> &reductionVars) {
  if (reductionVars.empty())
    return false;
  if (forceByrefReduction)
    return true;

  for (mlir::Value reductionVar : reductionVars) {
    if (auto declare =
            mlir::dyn_cast<hlfir::DeclareOp>(reductionVar.getDefiningOp()))
      reductionVar = declare.getMemref();

    if (!fir::isa_trivial(fir::unwrapRefType(reductionVar.getType())))
      return true;
  }
  return false;
}

void ReductionProcessor::addReductionDecl(
    mlir::Location currentLocation,
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpReductionClause &reduction,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
        *reductionSymbols) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::omp::ReductionDeclareOp decl;
  const auto &redOperator{
      std::get<Fortran::parser::OmpReductionOperator>(reduction.t)};
  const auto &objectList{std::get<Fortran::parser::OmpObjectList>(reduction.t)};

  if (!std::holds_alternative<Fortran::parser::DefinedOperator>(
          redOperator.u)) {
    if (const auto *reductionIntrinsic =
            std::get_if<Fortran::parser::ProcedureDesignator>(&redOperator.u)) {
      if (!ReductionProcessor::supportedIntrinsicProcReduction(
              *reductionIntrinsic)) {
        return;
      }
    } else {
      return;
    }
  }

  // initial pass to collect all recuction vars so we can figure out if this
  // should happen byref
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    if (const auto *name{
            Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
      if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
        if (reductionSymbols)
          reductionSymbols->push_back(symbol);
        mlir::Value symVal = converter.getSymbolAddress(*symbol);
        if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
          symVal = declOp.getBase();
        reductionVars.push_back(symVal);
      }
    }
  }
  const bool isByRef = doReductionByRef(reductionVars);

  if (const auto &redDefinedOp =
          std::get_if<Fortran::parser::DefinedOperator>(&redOperator.u)) {
    const auto &intrinsicOp{
        std::get<Fortran::parser::DefinedOperator::IntrinsicOperator>(
            redDefinedOp->u)};
    ReductionIdentifier redId = getReductionType(intrinsicOp);
    switch (redId) {
    case ReductionIdentifier::ADD:
    case ReductionIdentifier::MULTIPLY:
    case ReductionIdentifier::AND:
    case ReductionIdentifier::EQV:
    case ReductionIdentifier::OR:
    case ReductionIdentifier::NEQV:
      break;
    default:
      TODO(currentLocation,
           "Reduction of some intrinsic operators is not supported");
      break;
    }
    for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
      if (const auto *name{
              Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
        if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
          mlir::Value symVal = converter.getSymbolAddress(*symbol);
          if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
            symVal = declOp.getBase();
          auto redType = symVal.getType().cast<fir::ReferenceType>();
          if (redType.getEleTy().isa<fir::LogicalType>())
            decl = createReductionDecl(
                firOpBuilder,
                getReductionName(intrinsicOp, firOpBuilder.getI1Type(),
                                 isByRef),
                redId, redType, currentLocation, isByRef);
          else if (redType.getEleTy().isIntOrIndexOrFloat()) {
            decl = createReductionDecl(
                firOpBuilder, getReductionName(intrinsicOp, redType, isByRef),
                redId, redType, currentLocation, isByRef);
          } else {
            TODO(currentLocation, "Reduction of some types is not supported");
          }
          reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
              firOpBuilder.getContext(), decl.getSymName()));
        }
      }
    }
  } else if (const auto *reductionIntrinsic =
                 std::get_if<Fortran::parser::ProcedureDesignator>(
                     &redOperator.u)) {
    if (ReductionProcessor::supportedIntrinsicProcReduction(
            *reductionIntrinsic)) {
      ReductionProcessor::ReductionIdentifier redId =
          ReductionProcessor::getReductionType(*reductionIntrinsic);
      for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
        if (const auto *name{
                Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
          if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
            mlir::Value symVal = converter.getSymbolAddress(*symbol);
            if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
              symVal = declOp.getBase();
            auto redType = symVal.getType().cast<fir::ReferenceType>();
            assert(redType.getEleTy().isIntOrIndexOrFloat() &&
                   "Unsupported reduction type");
            decl = createReductionDecl(
                firOpBuilder,
                getReductionName(getRealName(*reductionIntrinsic).ToString(),
                                 redType, isByRef),
                redId, redType, currentLocation, isByRef);
            reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
                firOpBuilder.getContext(), decl.getSymName()));
          }
        }
      }
    }
  }
}

const Fortran::semantics::SourceName
ReductionProcessor::getRealName(const Fortran::parser::Name *name) {
  return name->symbol->GetUltimate().name();
}

const Fortran::semantics::SourceName ReductionProcessor::getRealName(
    const Fortran::parser::ProcedureDesignator &pd) {
  const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(pd)};
  assert(name && "Invalid Reduction Intrinsic.");
  return getRealName(name);
}

int ReductionProcessor::getOperationIdentity(ReductionIdentifier redId,
                                             mlir::Location loc) {
  switch (redId) {
  case ReductionIdentifier::ADD:
  case ReductionIdentifier::OR:
  case ReductionIdentifier::NEQV:
    return 0;
  case ReductionIdentifier::MULTIPLY:
  case ReductionIdentifier::AND:
  case ReductionIdentifier::EQV:
    return 1;
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }
}

} // namespace omp
} // namespace lower
} // namespace Fortran
