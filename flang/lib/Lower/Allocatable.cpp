//===-- Allocatable.cpp -- Allocatable statements lowering ----------------===//
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

#include "flang/Lower/Allocatable.h"
#include "../runtime/allocatable.h"
#include "RTBuilder.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

/// Runtime call generators
using namespace Fortran::runtime;
static void genAllocatableInitIntrinsic(Fortran::lower::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value boxAddress,
                                        mlir::Value typeCategory,
                                        mlir::Value kind, mlir::Value rank,
                                        mlir::Value corank) {
  auto callee =
      Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableInitIntrinsic)>(
          loc, builder);
  llvm::SmallVector<mlir::Value, 5> args = {boxAddress, typeCategory, kind,
                                            rank, corank};
  llvm::SmallVector<mlir::Value, 5> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

static void genAllocatableSetBounds(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value boxAddress,
                                    mlir::Value dimIndex, mlir::Value lowerBoud,
                                    mlir::Value upperBound) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableSetBounds)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 4> args{boxAddress, dimIndex, lowerBoud,
                                         upperBound};
  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

static mlir::Value
genAllocatableAllocate(Fortran::lower::FirOpBuilder &builder,
                       mlir::Location loc, mlir::Value boxAddress,
                       mlir::Value hasStat, mlir::Value errMsgBox,
                       mlir::Value sourceFile, mlir::Value sourceLine) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableAllocate)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 5> args{boxAddress, hasStat, errMsgBox,
                                         sourceFile, sourceLine};
  llvm::SmallVector<mlir::Value, 5> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

static mlir::Value
genAllocatableDeallocate(Fortran::lower::FirOpBuilder &builder,
                         mlir::Location loc, mlir::Value boxAddress,
                         mlir::Value hasStat, mlir::Value errMsgBox,
                         mlir::Value sourceFile, mlir::Value sourceLine) {
  auto callee = Fortran::lower::getRuntimeFunc<mkRTKey(AllocatableDeallocate)>(
      loc, builder);
  llvm::SmallVector<mlir::Value, 5> args{boxAddress, hasStat, errMsgBox,
                                         sourceFile, sourceLine};
  llvm::SmallVector<mlir::Value, 5> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  return builder.create<fir::CallOp>(loc, callee, operands).getResult(0);
}

/// Helper to get symbol from AllocateObject.
static const Fortran::semantics::Symbol &
unwrapSymbol(const Fortran::parser::AllocateObject &allocObj) {
  const auto &lastName = Fortran::parser::GetLastName(allocObj);
  assert(lastName.symbol);
  return lastName.symbol->GetUltimate();
}

namespace {
// Lower ALLOCATE/DEALLOCATE stmt ERROR and STAT variable as well as the source
// file location to be passed to the runtime.
struct ErrorManagementValues {
  void lower(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
             const Fortran::lower::SomeExpr *statExpr,
             const ::Fortran::lower::SomeExpr *errMsgExpr) {
    auto builder = converter.getFirOpBuilder();
    if (statExpr) {
      TODO("lower stat expr in allocate and deallocate");
      hasStat = builder.createBool(loc, true);
    } else {
      hasStat = builder.createBool(loc, false);
    }

    if (errMsgExpr)
      TODO("errmsg in allocate and deallocate");
    else
      errMsgBoxAddr = builder.createNullConstant(loc);
    sourceFile = converter.locationToFilename(loc);
    sourceLine = converter.locationToLineNo(loc, builder.getIntegerType(32));
  }
  bool hasErrorRecovery() const { return static_cast<bool>(statAddr); }
  // Values always initialized before lowering individual allocations
  mlir::Value sourceLine;
  mlir::Value sourceFile;
  mlir::Value hasStat;
  mlir::Value errMsgBoxAddr;
  // Value created only in certain cases before lowering individual allocations
  mlir::Value statAddr;
};

class AllocateStmtHelper {
public:
  AllocateStmtHelper(Fortran::lower::AbstractConverter &converter,
                     const Fortran::parser::AllocateStmt &stmt,
                     mlir::Location loc)
      : converter{converter}, builder{converter.getFirOpBuilder()}, stmt{stmt},
        loc{loc} {}

  void lower() {
    visitAllocateOptions();
    errorManagement.lower(converter, loc, statExpr, errMsgExpr);
    // Create a landing block after all allocations so that
    // we can jump there in case of error.
    if (errorManagement.hasErrorRecovery())
      TODO("error recovery");

    // TODO lower source and mold.
    if (sourceExpr || moldExpr)
      TODO("lower MOLD/SOURCE expr in allocate");

    for (const auto &allocation :
         std::get<std::list<Fortran::parser::Allocation>>(stmt.t))
      lowerAllocation(unwrapAllocation(allocation));
  }

private:
  struct Allocation {
    const Fortran::parser::Allocation &alloc;
    const Fortran::semantics::Symbol &symbol;
    const Fortran::semantics::DeclTypeSpec &type;
    bool hasCoarraySpec() const {
      return std::get<std::optional<Fortran::parser::AllocateCoarraySpec>>(
                 alloc.t)
          .has_value();
    }
    const auto &getShapeSpecs() const {
      return std::get<std::list<Fortran::parser::AllocateShapeSpec>>(alloc.t);
    }
  };

  Allocation unwrapAllocation(const Fortran::parser::Allocation &alloc) {
    const auto &allocObj = std::get<Fortran::parser::AllocateObject>(alloc.t);
    const auto &symbol = unwrapSymbol(allocObj);
    assert(symbol.GetType());
    return Allocation{alloc, symbol, *symbol.GetType()};
  }

  void visitAllocateOptions() {
    for (const auto &allocOption :
         std::get<std::list<Fortran::parser::AllocOpt>>(stmt.t))
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::StatOrErrmsg &statOrErr) {
                std::visit(
                    Fortran::common::visitors{
                        [&](const Fortran::parser::StatVariable &statVar) {
                          statExpr = Fortran::semantics::GetExpr(statVar);
                        },
                        [&](const Fortran::parser::MsgVariable &errMsgVar) {
                          errMsgExpr = Fortran::semantics::GetExpr(errMsgVar);
                        },
                    },
                    statOrErr.u);
              },
              [&](const Fortran::parser::AllocOpt::Source &source) {
                sourceExpr = Fortran::semantics::GetExpr(source.v.value());
              },
              [&](const Fortran::parser::AllocOpt::Mold &mold) {
                moldExpr = Fortran::semantics::GetExpr(mold.v.value());
              },
          },
          allocOption.u);
  }

  void lowerAllocation(const Allocation &alloc) {
    auto boxAddr = converter.getSymbolAddress(alloc.symbol);
    if (!boxAddr)
      TODO("Allocatable type not lowered yet");
    mlir::Value backupBox;
    if (errorManagement.hasErrorRecovery())
      backupBox = genDescriptorBackup(boxAddr);

    if (sourceExpr) {
      genSourceAllocation(alloc, boxAddr);
    } else if (moldExpr) {
      genMoldAllocation(alloc, boxAddr);
    } else {
      genSimpleAllocation(alloc, boxAddr);
    }

    if (errorManagement.hasErrorRecovery())
      genDescriptorRollBack(boxAddr, backupBox);
  }

  mlir::Value genDescriptorBackup(mlir::Value boxAddr) {
    // back-up descriptors in case something goes wrong. This is to fullfill
    // Fortran 2018 9.7.4 point 6 requirements that the original descriptor is
    // unaltered in case of error when stat is present. Instead of overthinking
    // what individual fields we need to backup, which in case of polymorphism
    // can be quite a lot, just save the whole descriptor before modifying it.
    TODO("descriptor backup in allocate with stat");
  }

  void genDescriptorRollBack(mlir::Value boxAddr, mlir::Value backupBox) {
    // copy back backed-up descriptors in case something went wrong.
    TODO("descriptor rollback in allocate with stat");
  }

  void genSimpleAllocation(const Allocation &alloc, mlir::Value boxAddr) {
    if (alloc.hasCoarraySpec())
      TODO("coarray allocation");
    if (alloc.type.IsPolymorphic())
      genSetType(alloc, boxAddr);
    genSetDeferredLengthParameters(alloc, boxAddr);
    // Set bounds for arrays
    auto idxTy = builder.getIndexType();
    auto i32Ty = builder.getIntegerType(32);
    for (const auto &iter : llvm::enumerate(alloc.getShapeSpecs())) {
      mlir::Value lb;
      const auto &bounds = iter.value().t;
      if (const auto &lbExpr = std::get<0>(bounds))
        lb = fir::getBase(
            converter.genExprValue(Fortran::semantics::GetExpr(*lbExpr), loc));
      else
        lb = builder.createIntegerConstant(loc, idxTy, 1);
      auto ub = fir::getBase(converter.genExprValue(
          Fortran::semantics::GetExpr(std::get<1>(bounds)), loc));
      auto dimIndex = builder.createIntegerConstant(loc, i32Ty, iter.index());
      // Runtime call
      genAllocatableSetBounds(builder, loc, boxAddr, dimIndex, lb, ub);
    }
    // Runtime call
    auto stat = genAllocatableAllocate(builder, loc, boxAddr, getHasStat(),
                                       getErrMsgBoxAddr(), getSourceFile(),
                                       getSourceLine());
    if (auto statAddr = getStatAddr()) {
      auto castStat = builder.createConvert(
          loc, fir::dyn_cast_ptrEleTy(statAddr.getType()), stat);
      builder.create<fir::StoreOp>(loc, castStat, statAddr);
    }
  }

  void genSetDeferredLengthParameters(const Allocation &alloc,
                                      mlir::Value boxAddr) {
    // TODO: go through type parameters and set the ones that are deferred
    // according to the allocation typespec.
  }

  void genSourceAllocation(const Allocation &alloc, mlir::Value boxAddr) {
    TODO("SOURCE allocation lowering");
  }
  void genMoldAllocation(const Allocation &alloc, mlir::Value boxAddr) {
    TODO("MOLD allocation lowering");
  }
  void genSetType(const Allocation &alloc, mlir::Value boxAddr) {
    TODO("Polymorphic entity allocation lowering");
  }

  mlir::Value getSourceLine() const {
    assert(errorManagement.sourceLine && "always needs to be lowered");
    return errorManagement.sourceLine;
  }
  mlir::Value getSourceFile() const {
    assert(errorManagement.sourceFile && "always needs to be lowered");
    return errorManagement.sourceFile;
  }
  mlir::Value getHasStat() {
    assert(errorManagement.sourceFile && "always needs to be lowered");
    return errorManagement.hasStat;
  }
  mlir::Value getErrMsgBoxAddr() {
    assert(errorManagement.sourceFile && "always needs to be lowered");
    return errorManagement.errMsgBoxAddr;
  }
  mlir::Value getStatAddr() const { return errorManagement.statAddr; }

  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::FirOpBuilder &builder;
  const Fortran::parser::AllocateStmt &stmt;
  const Fortran::lower::SomeExpr *sourceExpr{nullptr};
  const Fortran::lower::SomeExpr *moldExpr{nullptr};
  const Fortran::lower::SomeExpr *statExpr{nullptr};
  const Fortran::lower::SomeExpr *errMsgExpr{nullptr};
  ErrorManagementValues errorManagement;

  mlir::Location loc;
};
} // namespace

void Fortran::lower::genAllocateStmt(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::AllocateStmt &stmt, mlir::Location loc) {
  AllocateStmtHelper{converter, stmt, loc}.lower();
  return;
}

void Fortran::lower::genDeallocateStmt(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::DeallocateStmt &stmt, mlir::Location loc) {
  const Fortran::lower::SomeExpr *statExpr{nullptr};
  const Fortran::lower::SomeExpr *errMsgExpr{nullptr};
  for (const auto &statOrErr :
       std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t))
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::StatVariable &statVar) {
                     statExpr = Fortran::semantics::GetExpr(statVar);
                   },
                   [&](const Fortran::parser::MsgVariable &errMsgVar) {
                     errMsgExpr = Fortran::semantics::GetExpr(errMsgVar);
                   },
               },
               statOrErr.u);
  if (statExpr || errMsgExpr)
    TODO("error recovery in deallocate");
  ErrorManagementValues errorManagement;
  auto &builder = converter.getFirOpBuilder();
  errorManagement.lower(converter, loc, statExpr, errMsgExpr);
  for (const auto &allocateObject :
       std::get<std::list<Fortran::parser::AllocateObject>>(stmt.t)) {
    const auto &symbol = unwrapSymbol(allocateObject);
    auto boxAddr = converter.getSymbolAddress(symbol);
    if (!boxAddr)
      TODO("Allocatable type not lowered yet");
    // TODO use return stat for error recovery
    genAllocatableDeallocate(builder, loc, boxAddr, errorManagement.hasStat,
                             errorManagement.errMsgBoxAddr,
                             errorManagement.sourceFile,
                             errorManagement.sourceLine);
  }
}

void Fortran::lower::genAllocatableInit(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var, mlir::Value boxAddress) {
  auto loc = converter.genLocation(var.getSymbol().name());
  auto &builder = converter.getFirOpBuilder();
  auto declType = var.getSymbol().GetType();
  if (!declType)
    TODO("Is this possible ?");
  if (auto intrinsic = declType->AsIntrinsic()) {
    if (intrinsic->category() == Fortran::common::TypeCategory::Character) {
      TODO("character alloctable init");
    } else {
      auto i32ty = builder.getIntegerType(32);
      int catCode = static_cast<int>(intrinsic->category());
      auto cat = builder.createIntegerConstant(loc, i32ty, catCode);
      auto kindExpr = Fortran::evaluate::AsGenericExpr(
          Fortran::common::Clone(intrinsic->kind()));
      auto kind = fir::getBase(converter.genExprValue(kindExpr));
      auto rank =
          builder.createIntegerConstant(loc, i32ty, var.getSymbol().Rank());
      auto corank = builder.createIntegerConstant(loc, i32ty, 0);
      genAllocatableInitIntrinsic(builder, loc, boxAddress, cat, kind, rank,
                                  corank);
    }
  } else {
    TODO("derived type allocatable init");
  }
}
