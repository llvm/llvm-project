//===-- Atomic.cpp -- Lowering of atomic constructs -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Atomic.h"
#include "Clauses.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/type.h"
#include "flang/Support/Fortran.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

static llvm::cl::opt<bool> DumpAtomicAnalysis("fdebug-dump-atomic-analysis");

using namespace Fortran;

// Don't import the entire Fortran::lower.
namespace omp {
using namespace Fortran::lower::omp;
}

[[maybe_unused]] static void
dumpAtomicAnalysis(const parser::OpenMPAtomicConstruct::Analysis &analysis) {
  auto whatStr = [](int k) {
    std::string txt = "?";
    switch (k & parser::OpenMPAtomicConstruct::Analysis::Action) {
    case parser::OpenMPAtomicConstruct::Analysis::None:
      txt = "None";
      break;
    case parser::OpenMPAtomicConstruct::Analysis::Read:
      txt = "Read";
      break;
    case parser::OpenMPAtomicConstruct::Analysis::Write:
      txt = "Write";
      break;
    case parser::OpenMPAtomicConstruct::Analysis::Update:
      txt = "Update";
      break;
    }
    switch (k & parser::OpenMPAtomicConstruct::Analysis::Condition) {
    case parser::OpenMPAtomicConstruct::Analysis::IfTrue:
      txt += " | IfTrue";
      break;
    case parser::OpenMPAtomicConstruct::Analysis::IfFalse:
      txt += " | IfFalse";
      break;
    }
    return txt;
  };

  auto exprStr = [&](const parser::TypedExpr &expr) {
    if (auto *maybe = expr.get()) {
      if (maybe->v)
        return maybe->v->AsFortran();
    }
    return "<null>"s;
  };
  auto assignStr = [&](const parser::AssignmentStmt::TypedAssignment &assign) {
    if (auto *maybe = assign.get(); maybe && maybe->v) {
      std::string str;
      llvm::raw_string_ostream os(str);
      maybe->v->AsFortran(os);
      return str;
    }
    return "<null>"s;
  };

  const semantics::SomeExpr &atom = *analysis.atom.get()->v;

  llvm::errs() << "Analysis {\n";
  llvm::errs() << "  atom: " << atom.AsFortran() << "\n";
  llvm::errs() << "  cond: " << exprStr(analysis.cond) << "\n";
  llvm::errs() << "  op0 {\n";
  llvm::errs() << "    what: " << whatStr(analysis.op0.what) << "\n";
  llvm::errs() << "    assign: " << assignStr(analysis.op0.assign) << "\n";
  llvm::errs() << "  }\n";
  llvm::errs() << "  op1 {\n";
  llvm::errs() << "    what: " << whatStr(analysis.op1.what) << "\n";
  llvm::errs() << "    assign: " << assignStr(analysis.op1.assign) << "\n";
  llvm::errs() << "  }\n";
  llvm::errs() << "}\n";
}

static bool isPointerAssignment(const evaluate::Assignment &assign) {
  return common::visit(
      common::visitors{
          [](const evaluate::Assignment::BoundsSpec &) { return true; },
          [](const evaluate::Assignment::BoundsRemapping &) { return true; },
          [](const auto &) { return false; },
      },
      assign.u);
}

static fir::FirOpBuilder::InsertPoint
getInsertionPointBefore(mlir::Operation *op) {
  return fir::FirOpBuilder::InsertPoint(op->getBlock(),
                                        mlir::Block::iterator(op));
}

static fir::FirOpBuilder::InsertPoint
getInsertionPointAfter(mlir::Operation *op) {
  return fir::FirOpBuilder::InsertPoint(op->getBlock(),
                                        ++mlir::Block::iterator(op));
}

static mlir::IntegerAttr getAtomicHint(lower::AbstractConverter &converter,
                                       const omp::List<omp::Clause> &clauses) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const omp::Clause &clause : clauses) {
    if (clause.id != llvm::omp::Clause::OMPC_hint)
      continue;
    auto &hint = std::get<omp::clause::Hint>(clause.u);
    auto maybeVal = evaluate::ToInt64(hint.v);
    CHECK(maybeVal);
    return builder.getI64IntegerAttr(*maybeVal);
  }
  return nullptr;
}

static mlir::omp::ClauseMemoryOrderKind
getMemoryOrderKind(common::OmpMemoryOrderType kind) {
  switch (kind) {
  case common::OmpMemoryOrderType::Acq_Rel:
    return mlir::omp::ClauseMemoryOrderKind::Acq_rel;
  case common::OmpMemoryOrderType::Acquire:
    return mlir::omp::ClauseMemoryOrderKind::Acquire;
  case common::OmpMemoryOrderType::Relaxed:
    return mlir::omp::ClauseMemoryOrderKind::Relaxed;
  case common::OmpMemoryOrderType::Release:
    return mlir::omp::ClauseMemoryOrderKind::Release;
  case common::OmpMemoryOrderType::Seq_Cst:
    return mlir::omp::ClauseMemoryOrderKind::Seq_cst;
  }
  llvm_unreachable("Unexpected kind");
}

static std::optional<mlir::omp::ClauseMemoryOrderKind>
getMemoryOrderKind(llvm::omp::Clause clauseId) {
  switch (clauseId) {
  case llvm::omp::Clause::OMPC_acq_rel:
    return mlir::omp::ClauseMemoryOrderKind::Acq_rel;
  case llvm::omp::Clause::OMPC_acquire:
    return mlir::omp::ClauseMemoryOrderKind::Acquire;
  case llvm::omp::Clause::OMPC_relaxed:
    return mlir::omp::ClauseMemoryOrderKind::Relaxed;
  case llvm::omp::Clause::OMPC_release:
    return mlir::omp::ClauseMemoryOrderKind::Release;
  case llvm::omp::Clause::OMPC_seq_cst:
    return mlir::omp::ClauseMemoryOrderKind::Seq_cst;
  default:
    return std::nullopt;
  }
}

static std::optional<mlir::omp::ClauseMemoryOrderKind>
getMemoryOrderFromRequires(const semantics::Scope &scope) {
  // The REQUIRES construct is only allowed in the main program scope
  // and module scope, but seems like we also accept it in a subprogram
  // scope.
  // For safety, traverse all enclosing scopes and check if their symbol
  // contains REQUIRES.
  for (const auto *sc{&scope}; sc->kind() != semantics::Scope::Kind::Global;
       sc = &sc->parent()) {
    const semantics::Symbol *sym = sc->symbol();
    if (!sym)
      continue;

    const common::OmpMemoryOrderType *admo = common::visit(
        [](auto &&s) {
          using WithOmpDeclarative = semantics::WithOmpDeclarative;
          if constexpr (std::is_convertible_v<decltype(s),
                                              const WithOmpDeclarative &>) {
            return s.ompAtomicDefaultMemOrder();
          }
          return static_cast<const common::OmpMemoryOrderType *>(nullptr);
        },
        sym->details());
    if (admo)
      return getMemoryOrderKind(*admo);
  }

  return std::nullopt;
}

static std::optional<mlir::omp::ClauseMemoryOrderKind>
getDefaultAtomicMemOrder(semantics::SemanticsContext &semaCtx) {
  unsigned version = semaCtx.langOptions().OpenMPVersion;
  if (version > 50)
    return mlir::omp::ClauseMemoryOrderKind::Relaxed;
  return std::nullopt;
}

static std::optional<mlir::omp::ClauseMemoryOrderKind>
getAtomicMemoryOrder(semantics::SemanticsContext &semaCtx,
                     const omp::List<omp::Clause> &clauses,
                     const semantics::Scope &scope) {
  for (const omp::Clause &clause : clauses) {
    if (auto maybeKind = getMemoryOrderKind(clause.id))
      return *maybeKind;
  }

  if (auto maybeKind = getMemoryOrderFromRequires(scope))
    return *maybeKind;

  return getDefaultAtomicMemOrder(semaCtx);
}

static mlir::omp::ClauseMemoryOrderKindAttr
makeMemOrderAttr(lower::AbstractConverter &converter,
                 std::optional<mlir::omp::ClauseMemoryOrderKind> maybeKind) {
  if (maybeKind) {
    return mlir::omp::ClauseMemoryOrderKindAttr::get(
        converter.getFirOpBuilder().getContext(), *maybeKind);
  }
  return nullptr;
}

static mlir::Operation * //
genAtomicRead(lower::AbstractConverter &converter,
              semantics::SemanticsContext &semaCtx, mlir::Location loc,
              lower::StatementContext &stmtCtx, mlir::Value atomAddr,
              const semantics::SomeExpr &atom,
              const evaluate::Assignment &assign, mlir::IntegerAttr hint,
              std::optional<mlir::omp::ClauseMemoryOrderKind> memOrder,
              fir::FirOpBuilder::InsertPoint preAt,
              fir::FirOpBuilder::InsertPoint atomicAt,
              fir::FirOpBuilder::InsertPoint postAt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  builder.restoreInsertionPoint(preAt);

  // If the atomic clause is read then the memory-order clause must
  // not be release.
  if (memOrder) {
    if (*memOrder == mlir::omp::ClauseMemoryOrderKind::Release) {
      // Reset it back to the default.
      memOrder = getDefaultAtomicMemOrder(semaCtx);
    } else if (*memOrder == mlir::omp::ClauseMemoryOrderKind::Acq_rel) {
      // The MLIR verifier doesn't like acq_rel either.
      memOrder = mlir::omp::ClauseMemoryOrderKind::Acquire;
    }
  }

  mlir::Value storeAddr =
      fir::getBase(converter.genExprAddr(assign.lhs, stmtCtx, &loc));
  mlir::Type atomType = fir::unwrapRefType(atomAddr.getType());
  mlir::Type storeType = fir::unwrapRefType(storeAddr.getType());

  mlir::Value toAddr = [&]() {
    if (atomType == storeType)
      return storeAddr;
    return builder.createTemporary(loc, atomType, ".tmp.atomval");
  }();

  builder.restoreInsertionPoint(atomicAt);
  mlir::Operation *op = builder.create<mlir::omp::AtomicReadOp>(
      loc, atomAddr, toAddr, mlir::TypeAttr::get(atomType), hint,
      makeMemOrderAttr(converter, memOrder));

  if (atomType != storeType) {
    lower::ExprToValueMap overrides;
    // The READ operation could be a part of UPDATE CAPTURE, so make sure
    // we don't emit extra code into the body of the atomic op.
    builder.restoreInsertionPoint(postAt);
    mlir::Value load = builder.create<fir::LoadOp>(loc, toAddr);
    overrides.try_emplace(&atom, load);

    converter.overrideExprValues(&overrides);
    mlir::Value value =
        fir::getBase(converter.genExprValue(assign.rhs, stmtCtx, &loc));
    converter.resetExprOverrides();

    builder.create<fir::StoreOp>(loc, value, storeAddr);
  }
  return op;
}

static mlir::Operation * //
genAtomicWrite(lower::AbstractConverter &converter,
               semantics::SemanticsContext &semaCtx, mlir::Location loc,
               lower::StatementContext &stmtCtx, mlir::Value atomAddr,
               const semantics::SomeExpr &atom,
               const evaluate::Assignment &assign, mlir::IntegerAttr hint,
               std::optional<mlir::omp::ClauseMemoryOrderKind> memOrder,
               fir::FirOpBuilder::InsertPoint preAt,
               fir::FirOpBuilder::InsertPoint atomicAt,
               fir::FirOpBuilder::InsertPoint postAt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  builder.restoreInsertionPoint(preAt);

  // If the atomic clause is write then the memory-order clause must
  // not be acquire.
  if (memOrder) {
    if (*memOrder == mlir::omp::ClauseMemoryOrderKind::Acquire) {
      // Reset it back to the default.
      memOrder = getDefaultAtomicMemOrder(semaCtx);
    } else if (*memOrder == mlir::omp::ClauseMemoryOrderKind::Acq_rel) {
      // The MLIR verifier doesn't like acq_rel either.
      memOrder = mlir::omp::ClauseMemoryOrderKind::Release;
    }
  }

  mlir::Value value =
      fir::getBase(converter.genExprValue(assign.rhs, stmtCtx, &loc));
  mlir::Type atomType = fir::unwrapRefType(atomAddr.getType());
  mlir::Value converted = builder.createConvert(loc, atomType, value);

  builder.restoreInsertionPoint(atomicAt);
  mlir::Operation *op = builder.create<mlir::omp::AtomicWriteOp>(
      loc, atomAddr, converted, hint, makeMemOrderAttr(converter, memOrder));
  return op;
}

static mlir::Operation *
genAtomicUpdate(lower::AbstractConverter &converter,
                semantics::SemanticsContext &semaCtx, mlir::Location loc,
                lower::StatementContext &stmtCtx, mlir::Value atomAddr,
                const semantics::SomeExpr &atom,
                const evaluate::Assignment &assign, mlir::IntegerAttr hint,
                std::optional<mlir::omp::ClauseMemoryOrderKind> memOrder,
                fir::FirOpBuilder::InsertPoint preAt,
                fir::FirOpBuilder::InsertPoint atomicAt,
                fir::FirOpBuilder::InsertPoint postAt) {
  lower::ExprToValueMap overrides;
  lower::StatementContext naCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  builder.restoreInsertionPoint(preAt);

  mlir::Type atomType = fir::unwrapRefType(atomAddr.getType());

  // This must exist by now.
  semantics::SomeExpr input = *evaluate::GetConvertInput(assign.rhs);
  std::vector<semantics::SomeExpr> args =
      evaluate::GetTopLevelOperation(input).second;
  assert(!args.empty() && "Update operation without arguments");
  for (auto &arg : args) {
    if (!evaluate::IsSameOrConvertOf(arg, atom)) {
      mlir::Value val = fir::getBase(converter.genExprValue(arg, naCtx, &loc));
      overrides.try_emplace(&arg, val);
    }
  }

  builder.restoreInsertionPoint(atomicAt);
  auto updateOp = builder.create<mlir::omp::AtomicUpdateOp>(
      loc, atomAddr, hint, makeMemOrderAttr(converter, memOrder));

  mlir::Region &region = updateOp->getRegion(0);
  mlir::Block *block = builder.createBlock(&region, {}, {atomType}, {loc});
  mlir::Value localAtom = fir::getBase(block->getArgument(0));
  overrides.try_emplace(&atom, localAtom);

  converter.overrideExprValues(&overrides);
  mlir::Value updated =
      fir::getBase(converter.genExprValue(assign.rhs, stmtCtx, &loc));
  mlir::Value converted = builder.createConvert(loc, atomType, updated);
  builder.create<mlir::omp::YieldOp>(loc, converted);
  converter.resetExprOverrides();

  builder.restoreInsertionPoint(postAt); // For naCtx cleanups
  return updateOp;
}

static mlir::Operation *
genAtomicOperation(lower::AbstractConverter &converter,
                   semantics::SemanticsContext &semaCtx, mlir::Location loc,
                   lower::StatementContext &stmtCtx, int action,
                   mlir::Value atomAddr, const semantics::SomeExpr &atom,
                   const evaluate::Assignment &assign, mlir::IntegerAttr hint,
                   std::optional<mlir::omp::ClauseMemoryOrderKind> memOrder,
                   fir::FirOpBuilder::InsertPoint preAt,
                   fir::FirOpBuilder::InsertPoint atomicAt,
                   fir::FirOpBuilder::InsertPoint postAt) {
  if (isPointerAssignment(assign)) {
    TODO(loc, "Code generation for pointer assignment is not implemented yet");
  }

  // This function and the functions called here do not preserve the
  // builder's insertion point, or set it to anything specific.
  switch (action) {
  case parser::OpenMPAtomicConstruct::Analysis::Read:
    return genAtomicRead(converter, semaCtx, loc, stmtCtx, atomAddr, atom,
                         assign, hint, memOrder, preAt, atomicAt, postAt);
  case parser::OpenMPAtomicConstruct::Analysis::Write:
    return genAtomicWrite(converter, semaCtx, loc, stmtCtx, atomAddr, atom,
                          assign, hint, memOrder, preAt, atomicAt, postAt);
  case parser::OpenMPAtomicConstruct::Analysis::Update:
    return genAtomicUpdate(converter, semaCtx, loc, stmtCtx, atomAddr, atom,
                           assign, hint, memOrder, preAt, atomicAt, postAt);
  default:
    return nullptr;
  }
}

void Fortran::lower::omp::lowerAtomic(
    AbstractConverter &converter, SymMap &symTable,
    semantics::SemanticsContext &semaCtx, pft::Evaluation &eval,
    const parser::OpenMPAtomicConstruct &construct) {
  auto get = [](auto &&typedWrapper) -> decltype(&*typedWrapper.get()->v) {
    if (auto *maybe = typedWrapper.get(); maybe && maybe->v) {
      return &*maybe->v;
    } else {
      return nullptr;
    }
  };

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto &dirSpec = std::get<parser::OmpDirectiveSpecification>(construct.t);
  omp::List<omp::Clause> clauses = makeClauses(dirSpec.Clauses(), semaCtx);
  lower::StatementContext stmtCtx;

  const parser::OpenMPAtomicConstruct::Analysis &analysis = construct.analysis;
  if (DumpAtomicAnalysis)
    dumpAtomicAnalysis(analysis);

  const semantics::SomeExpr &atom = *get(analysis.atom);
  mlir::Location loc = converter.genLocation(construct.source);
  mlir::Value atomAddr =
      fir::getBase(converter.genExprAddr(atom, stmtCtx, &loc));
  mlir::IntegerAttr hint = getAtomicHint(converter, clauses);
  std::optional<mlir::omp::ClauseMemoryOrderKind> memOrder =
      getAtomicMemoryOrder(semaCtx, clauses,
                           semaCtx.FindScope(construct.source));

  if (auto *cond = get(analysis.cond)) {
    (void)cond;
    TODO(loc, "OpenMP ATOMIC COMPARE");
  } else {
    int action0 = analysis.op0.what & analysis.Action;
    int action1 = analysis.op1.what & analysis.Action;
    mlir::Operation *captureOp = nullptr;
    fir::FirOpBuilder::InsertPoint preAt = builder.saveInsertionPoint();
    fir::FirOpBuilder::InsertPoint atomicAt, postAt;

    if (construct.IsCapture()) {
      // Capturing operation.
      assert(action0 != analysis.None && action1 != analysis.None &&
             "Expexcing two actions");
      (void)action0;
      (void)action1;
      captureOp = builder.create<mlir::omp::AtomicCaptureOp>(
          loc, hint, makeMemOrderAttr(converter, memOrder));
      // Set the non-atomic insertion point to before the atomic.capture.
      preAt = getInsertionPointBefore(captureOp);

      mlir::Block *block = builder.createBlock(&captureOp->getRegion(0));
      builder.setInsertionPointToEnd(block);
      // Set the atomic insertion point to before the terminator inside
      // atomic.capture.
      mlir::Operation *term = builder.create<mlir::omp::TerminatorOp>(loc);
      atomicAt = getInsertionPointBefore(term);
      postAt = getInsertionPointAfter(captureOp);
      hint = nullptr;
      memOrder = std::nullopt;
    } else {
      // Non-capturing operation.
      assert(action0 != analysis.None && action1 == analysis.None &&
             "Expexcing single action");
      assert(!(analysis.op0.what & analysis.Condition));
      postAt = atomicAt = preAt;
    }

    // The builder's insertion point needs to be specifically set before
    // each call to `genAtomicOperation`.
    mlir::Operation *firstOp = genAtomicOperation(
        converter, semaCtx, loc, stmtCtx, analysis.op0.what, atomAddr, atom,
        *get(analysis.op0.assign), hint, memOrder, preAt, atomicAt, postAt);
    assert(firstOp && "Should have created an atomic operation");
    atomicAt = getInsertionPointAfter(firstOp);

    mlir::Operation *secondOp = nullptr;
    if (analysis.op1.what != analysis.None) {
      secondOp = genAtomicOperation(
          converter, semaCtx, loc, stmtCtx, analysis.op1.what, atomAddr, atom,
          *get(analysis.op1.assign), hint, memOrder, preAt, atomicAt, postAt);
    }

    if (construct.IsCapture()) {
      // If this is a capture operation, the first/second ops will be inside
      // of it. Set the insertion point to past the capture op itself.
      builder.restoreInsertionPoint(postAt);
    } else {
      if (secondOp) {
        builder.setInsertionPointAfter(secondOp);
      } else {
        builder.setInsertionPointAfter(firstOp);
      }
    }
  }
}
