//===-- Atomic.cpp -- Lowering of atomic constructs -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Atomic.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/type.h"
#include "flang/Support/Fortran.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
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
  auto assignStr = [&](const parser::TypedAssignment &assign) {
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
getInsertionPointBefore(aiir::Operation *op) {
  return fir::FirOpBuilder::InsertPoint(op->getBlock(),
                                        aiir::Block::iterator(op));
}

static fir::FirOpBuilder::InsertPoint
getInsertionPointAfter(aiir::Operation *op) {
  return fir::FirOpBuilder::InsertPoint(op->getBlock(),
                                        ++aiir::Block::iterator(op));
}

static aiir::IntegerAttr getAtomicHint(lower::AbstractConverter &converter,
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

static aiir::omp::ClauseMemoryOrderKind
getMemoryOrderKind(common::OmpMemoryOrderType kind) {
  switch (kind) {
  case common::OmpMemoryOrderType::Acq_Rel:
    return aiir::omp::ClauseMemoryOrderKind::Acq_rel;
  case common::OmpMemoryOrderType::Acquire:
    return aiir::omp::ClauseMemoryOrderKind::Acquire;
  case common::OmpMemoryOrderType::Relaxed:
    return aiir::omp::ClauseMemoryOrderKind::Relaxed;
  case common::OmpMemoryOrderType::Release:
    return aiir::omp::ClauseMemoryOrderKind::Release;
  case common::OmpMemoryOrderType::Seq_Cst:
    return aiir::omp::ClauseMemoryOrderKind::Seq_cst;
  }
  llvm_unreachable("Unexpected kind");
}

static std::optional<aiir::omp::ClauseMemoryOrderKind>
getMemoryOrderKind(llvm::omp::Clause clauseId) {
  switch (clauseId) {
  case llvm::omp::Clause::OMPC_acq_rel:
    return aiir::omp::ClauseMemoryOrderKind::Acq_rel;
  case llvm::omp::Clause::OMPC_acquire:
    return aiir::omp::ClauseMemoryOrderKind::Acquire;
  case llvm::omp::Clause::OMPC_relaxed:
    return aiir::omp::ClauseMemoryOrderKind::Relaxed;
  case llvm::omp::Clause::OMPC_release:
    return aiir::omp::ClauseMemoryOrderKind::Release;
  case llvm::omp::Clause::OMPC_seq_cst:
    return aiir::omp::ClauseMemoryOrderKind::Seq_cst;
  default:
    return std::nullopt;
  }
}

static std::optional<aiir::omp::ClauseMemoryOrderKind>
getMemoryOrderFromRequires(const semantics::Scope &scope) {
  // The REQUIRES construct is only allowed in the main program scope
  // and module scope, but seems like we also accept it in a subprogram
  // scope.
  // For safety, traverse all enclosing scopes and check if their symbol
  // contains REQUIRES.
  const semantics::Scope &unitScope = semantics::omp::GetProgramUnit(scope);
  if (auto *symbol = unitScope.symbol()) {
    const common::OmpMemoryOrderType *admo = common::visit(
        [](auto &&s) {
          using WithOmpDeclarative = semantics::WithOmpDeclarative;
          if constexpr (std::is_convertible_v<decltype(s),
                                              const WithOmpDeclarative &>) {
            return s.ompAtomicDefaultMemOrder();
          }
          return static_cast<const common::OmpMemoryOrderType *>(nullptr);
        },
        symbol->details());

    if (admo)
      return getMemoryOrderKind(*admo);
  }

  return std::nullopt;
}

static std::optional<aiir::omp::ClauseMemoryOrderKind>
getDefaultAtomicMemOrder(semantics::SemanticsContext &semaCtx) {
  unsigned version = semaCtx.langOptions().OpenMPVersion;
  if (version > 50)
    return aiir::omp::ClauseMemoryOrderKind::Relaxed;
  return std::nullopt;
}

static std::pair<std::optional<aiir::omp::ClauseMemoryOrderKind>, bool>
getAtomicMemoryOrder(semantics::SemanticsContext &semaCtx,
                     const omp::List<omp::Clause> &clauses,
                     const semantics::Scope &scope) {
  for (const omp::Clause &clause : clauses) {
    if (auto maybeKind = getMemoryOrderKind(clause.id))
      return std::make_pair(*maybeKind, /*canOverride=*/false);
  }

  if (auto maybeKind = getMemoryOrderFromRequires(scope))
    return std::make_pair(*maybeKind, /*canOverride=*/true);

  return std::make_pair(getDefaultAtomicMemOrder(semaCtx),
                        /*canOverride=*/false);
}

static std::optional<aiir::omp::ClauseMemoryOrderKind>
makeValidForAction(std::optional<aiir::omp::ClauseMemoryOrderKind> memOrder,
                   int action0, int action1, unsigned version) {
  // When the atomic default memory order specified on a REQUIRES directive is
  // disallowed on a given ATOMIC operation, and it's not ACQ_REL, the order
  // reverts to RELAXED. ACQ_REL decays to either ACQUIRE or RELEASE, depending
  // on the operation.

  if (!memOrder) {
    return memOrder;
  }

  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  // Figure out the main action (i.e. disregard a potential capture operation)
  int action = action0;
  if (action1 != Analysis::None)
    action = action0 == Analysis::Read ? action1 : action0;

  // Avaliable orderings: acquire, acq_rel, relaxed, release, seq_cst

  if (action == Analysis::Read) {
    // "acq_rel" decays to "acquire"
    if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acq_rel)
      return aiir::omp::ClauseMemoryOrderKind::Acquire;
  } else if (action == Analysis::Write) {
    // "acq_rel" decays to "release"
    if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acq_rel)
      return aiir::omp::ClauseMemoryOrderKind::Release;
  }

  if (version > 50) {
    if (action == Analysis::Read) {
      // "release" prohibited
      if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Release)
        return aiir::omp::ClauseMemoryOrderKind::Relaxed;
    }
    if (action == Analysis::Write) {
      // "acquire" prohibited
      if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acquire)
        return aiir::omp::ClauseMemoryOrderKind::Relaxed;
    }
  } else {
    if (action == Analysis::Read) {
      // "release" prohibited
      if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Release)
        return aiir::omp::ClauseMemoryOrderKind::Relaxed;
    } else {
      if (action & Analysis::Write) { // include "update"
        // "acquire" prohibited
        if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acquire)
          return aiir::omp::ClauseMemoryOrderKind::Relaxed;
        if (action == Analysis::Update) {
          // "acq_rel" prohibited
          if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acq_rel)
            return aiir::omp::ClauseMemoryOrderKind::Relaxed;
        }
      }
    }
  }

  return memOrder;
}

static aiir::omp::ClauseMemoryOrderKindAttr
makeMemOrderAttr(lower::AbstractConverter &converter,
                 std::optional<aiir::omp::ClauseMemoryOrderKind> maybeKind) {
  if (maybeKind) {
    return aiir::omp::ClauseMemoryOrderKindAttr::get(
        converter.getFirOpBuilder().getContext(), *maybeKind);
  }
  return nullptr;
}

static aiir::Operation * //
genAtomicRead(lower::AbstractConverter &converter,
              semantics::SemanticsContext &semaCtx, aiir::Location loc,
              lower::StatementContext &stmtCtx, aiir::Value atomAddr,
              const semantics::SomeExpr &atom,
              const evaluate::Assignment &assign, aiir::IntegerAttr hint,
              std::optional<aiir::omp::ClauseMemoryOrderKind> memOrder,
              fir::FirOpBuilder::InsertPoint preAt,
              fir::FirOpBuilder::InsertPoint atomicAt,
              fir::FirOpBuilder::InsertPoint postAt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  builder.restoreInsertionPoint(preAt);

  // If the atomic clause is read then the memory-order clause must
  // not be release.
  if (memOrder) {
    if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Release) {
      // Reset it back to the default.
      memOrder = getDefaultAtomicMemOrder(semaCtx);
    } else if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acq_rel) {
      // The AIIR verifier doesn't like acq_rel either.
      memOrder = aiir::omp::ClauseMemoryOrderKind::Acquire;
    }
  }

  aiir::Value storeAddr =
      fir::getBase(converter.genExprAddr(assign.lhs, stmtCtx, &loc));
  aiir::Type atomType = fir::unwrapRefType(atomAddr.getType());
  aiir::Type storeType = fir::unwrapRefType(storeAddr.getType());

  aiir::Value toAddr = [&]() {
    if (atomType == storeType)
      return storeAddr;
    return builder.createTemporary(loc, atomType, ".tmp.atomval");
  }();

  builder.restoreInsertionPoint(atomicAt);
  aiir::Operation *op = aiir::omp::AtomicReadOp::create(
      builder, loc, atomAddr, toAddr, aiir::TypeAttr::get(atomType), hint,
      makeMemOrderAttr(converter, memOrder));

  if (atomType != storeType) {
    lower::ExprToValueMap overrides;
    // The READ operation could be a part of UPDATE CAPTURE, so make sure
    // we don't emit extra code into the body of the atomic op.
    builder.restoreInsertionPoint(postAt);
    aiir::Value load = fir::LoadOp::create(builder, loc, toAddr);
    overrides.try_emplace(&atom, load);

    converter.overrideExprValues(&overrides);
    aiir::Value value =
        fir::getBase(converter.genExprValue(assign.rhs, stmtCtx, &loc));
    converter.resetExprOverrides();

    fir::StoreOp::create(builder, loc, value, storeAddr);
  }
  return op;
}

static aiir::Operation * //
genAtomicWrite(lower::AbstractConverter &converter,
               semantics::SemanticsContext &semaCtx, aiir::Location loc,
               lower::StatementContext &stmtCtx, aiir::Value atomAddr,
               const semantics::SomeExpr &atom,
               const evaluate::Assignment &assign, aiir::IntegerAttr hint,
               std::optional<aiir::omp::ClauseMemoryOrderKind> memOrder,
               fir::FirOpBuilder::InsertPoint preAt,
               fir::FirOpBuilder::InsertPoint atomicAt,
               fir::FirOpBuilder::InsertPoint postAt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  builder.restoreInsertionPoint(preAt);

  // If the atomic clause is write then the memory-order clause must
  // not be acquire.
  if (memOrder) {
    if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acquire) {
      // Reset it back to the default.
      memOrder = getDefaultAtomicMemOrder(semaCtx);
    } else if (*memOrder == aiir::omp::ClauseMemoryOrderKind::Acq_rel) {
      // The AIIR verifier doesn't like acq_rel either.
      memOrder = aiir::omp::ClauseMemoryOrderKind::Release;
    }
  }

  aiir::Value value =
      fir::getBase(converter.genExprValue(assign.rhs, stmtCtx, &loc));
  aiir::Type atomType = fir::unwrapRefType(atomAddr.getType());
  aiir::Value converted = builder.createConvert(loc, atomType, value);

  builder.restoreInsertionPoint(atomicAt);
  aiir::Operation *op =
      aiir::omp::AtomicWriteOp::create(builder, loc, atomAddr, converted, hint,
                                       makeMemOrderAttr(converter, memOrder));
  return op;
}

static aiir::Operation *
genAtomicUpdate(lower::AbstractConverter &converter,
                semantics::SemanticsContext &semaCtx, aiir::Location loc,
                lower::StatementContext &stmtCtx, aiir::Value atomAddr,
                const semantics::SomeExpr &atom,
                const evaluate::Assignment &assign, aiir::IntegerAttr hint,
                std::optional<aiir::omp::ClauseMemoryOrderKind> memOrder,
                fir::FirOpBuilder::InsertPoint preAt,
                fir::FirOpBuilder::InsertPoint atomicAt,
                fir::FirOpBuilder::InsertPoint postAt) {
  lower::ExprToValueMap overrides;
  lower::StatementContext naCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  builder.restoreInsertionPoint(preAt);

  aiir::Type atomType = fir::unwrapRefType(atomAddr.getType());

  // This must exist by now.
  semantics::SomeExpr rhs = assign.rhs;
  semantics::SomeExpr input = *evaluate::GetConvertInput(rhs);
  auto [opcode, args] = evaluate::GetTopLevelOperationIgnoreResizing(input);
  assert(!args.empty() && "Update operation without arguments");

  for (auto &arg : args) {
    if (!evaluate::IsSameOrConvertOf(arg, atom)) {
      aiir::Value val = fir::getBase(converter.genExprValue(arg, naCtx, &loc));
      overrides.try_emplace(&arg, val);
    }
  }

  aiir::ModuleOp module = builder.getModule();
  aiir::omp::AtomicControlAttr atomicControlAttr =
      aiir::omp::AtomicControlAttr::get(
          builder.getContext(), fir::getAtomicIgnoreDenormalMode(module),
          fir::getAtomicFineGrainedMemory(module),
          fir::getAtomicRemoteMemory(module));
  builder.restoreInsertionPoint(atomicAt);
  auto updateOp = aiir::omp::AtomicUpdateOp::create(
      builder, loc, atomAddr, atomicControlAttr, hint,
      makeMemOrderAttr(converter, memOrder));

  aiir::Region &region = updateOp->getRegion(0);
  aiir::Block *block = builder.createBlock(&region, {}, {atomType}, {loc});
  aiir::Value localAtom = fir::getBase(block->getArgument(0));
  overrides.try_emplace(&atom, localAtom);

  converter.overrideExprValues(&overrides);
  aiir::Value updated =
      fir::getBase(converter.genExprValue(rhs, stmtCtx, &loc));
  aiir::Value converted = builder.createConvert(loc, atomType, updated);
  aiir::omp::YieldOp::create(builder, loc, converted);
  converter.resetExprOverrides();

  builder.restoreInsertionPoint(postAt); // For naCtx cleanups
  return updateOp;
}

static aiir::Operation *
genAtomicOperation(lower::AbstractConverter &converter,
                   semantics::SemanticsContext &semaCtx, aiir::Location loc,
                   lower::StatementContext &stmtCtx, int action,
                   aiir::Value atomAddr, const semantics::SomeExpr &atom,
                   const evaluate::Assignment &assign, aiir::IntegerAttr hint,
                   std::optional<aiir::omp::ClauseMemoryOrderKind> memOrder,
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
  const parser::OmpDirectiveSpecification &dirSpec = construct.BeginDir();
  omp::List<omp::Clause> clauses = makeClauses(dirSpec.Clauses(), semaCtx);
  lower::StatementContext stmtCtx;

  const parser::OpenMPAtomicConstruct::Analysis &analysis = construct.analysis;
  if (DumpAtomicAnalysis)
    dumpAtomicAnalysis(analysis);

  const semantics::SomeExpr &atom = *get(analysis.atom);
  aiir::Location loc = converter.genLocation(construct.source);
  aiir::Value atomAddr =
      fir::getBase(converter.genExprAddr(atom, stmtCtx, &loc));
  aiir::IntegerAttr hint = getAtomicHint(converter, clauses);
  auto [memOrder, canOverride] = getAtomicMemoryOrder(
      semaCtx, clauses, semaCtx.FindScope(construct.source));

  unsigned version = semaCtx.langOptions().OpenMPVersion;
  int action0 = analysis.op0.what & analysis.Action;
  int action1 = analysis.op1.what & analysis.Action;
  if (canOverride)
    memOrder = makeValidForAction(memOrder, action0, action1, version);

  if (auto *cond = get(analysis.cond)) {
    (void)cond;
    TODO(loc, "OpenMP ATOMIC COMPARE");
  } else {
    aiir::Operation *captureOp = nullptr;
    fir::FirOpBuilder::InsertPoint preAt = builder.saveInsertionPoint();
    fir::FirOpBuilder::InsertPoint atomicAt, postAt;

    if (construct.IsCapture()) {
      // Capturing operation.
      assert(action0 != analysis.None && action1 != analysis.None &&
             "Expexcing two actions");
      (void)action0;
      (void)action1;
      captureOp = aiir::omp::AtomicCaptureOp::create(
          builder, loc, hint, makeMemOrderAttr(converter, memOrder));
      // Set the non-atomic insertion point to before the atomic.capture.
      preAt = getInsertionPointBefore(captureOp);

      aiir::Block *block = builder.createBlock(&captureOp->getRegion(0));
      builder.setInsertionPointToEnd(block);
      // Set the atomic insertion point to before the terminator inside
      // atomic.capture.
      aiir::Operation *term = aiir::omp::TerminatorOp::create(builder, loc);
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
    aiir::Operation *firstOp = genAtomicOperation(
        converter, semaCtx, loc, stmtCtx, analysis.op0.what, atomAddr, atom,
        *get(analysis.op0.assign), hint, memOrder, preAt, atomicAt, postAt);
    assert(firstOp && "Should have created an atomic operation");
    atomicAt = getInsertionPointAfter(firstOp);

    aiir::Operation *secondOp = nullptr;
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
