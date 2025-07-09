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
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
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

namespace {
// An example of a type that can be used to get the return value from
// the visitor:
//   visitor(type_identity<Xyz>) -> result_type
using SomeArgType = evaluate::Type<common::TypeCategory::Integer, 4>;

struct GetProc
    : public evaluate::Traverse<GetProc, const evaluate::ProcedureDesignator *,
                                false> {
  using Result = const evaluate::ProcedureDesignator *;
  using Base = evaluate::Traverse<GetProc, Result, false>;
  GetProc() : Base(*this) {}

  using Base::operator();

  static Result Default() { return nullptr; }

  Result operator()(const evaluate::ProcedureDesignator &p) const { return &p; }
  static Result Combine(Result a, Result b) { return a != nullptr ? a : b; }
};

struct WithType {
  WithType(const evaluate::DynamicType &t) : type(t) {
    assert(type.category() != common::TypeCategory::Derived &&
           "Type cannot be a derived type");
  }

  template <typename VisitorTy> //
  auto visit(VisitorTy &&visitor) const
      -> std::invoke_result_t<VisitorTy, SomeArgType> {
    switch (type.category()) {
    case common::TypeCategory::Integer:
      switch (type.kind()) {
      case 1:
        return visitor(llvm::type_identity<evaluate::Type<Integer, 1>>{});
      case 2:
        return visitor(llvm::type_identity<evaluate::Type<Integer, 2>>{});
      case 4:
        return visitor(llvm::type_identity<evaluate::Type<Integer, 4>>{});
      case 8:
        return visitor(llvm::type_identity<evaluate::Type<Integer, 8>>{});
      case 16:
        return visitor(llvm::type_identity<evaluate::Type<Integer, 16>>{});
      }
      break;
    case common::TypeCategory::Unsigned:
      switch (type.kind()) {
      case 1:
        return visitor(llvm::type_identity<evaluate::Type<Unsigned, 1>>{});
      case 2:
        return visitor(llvm::type_identity<evaluate::Type<Unsigned, 2>>{});
      case 4:
        return visitor(llvm::type_identity<evaluate::Type<Unsigned, 4>>{});
      case 8:
        return visitor(llvm::type_identity<evaluate::Type<Unsigned, 8>>{});
      case 16:
        return visitor(llvm::type_identity<evaluate::Type<Unsigned, 16>>{});
      }
      break;
    case common::TypeCategory::Real:
      switch (type.kind()) {
      case 2:
        return visitor(llvm::type_identity<evaluate::Type<Real, 2>>{});
      case 3:
        return visitor(llvm::type_identity<evaluate::Type<Real, 3>>{});
      case 4:
        return visitor(llvm::type_identity<evaluate::Type<Real, 4>>{});
      case 8:
        return visitor(llvm::type_identity<evaluate::Type<Real, 8>>{});
      case 10:
        return visitor(llvm::type_identity<evaluate::Type<Real, 10>>{});
      case 16:
        return visitor(llvm::type_identity<evaluate::Type<Real, 16>>{});
      }
      break;
    case common::TypeCategory::Complex:
      switch (type.kind()) {
      case 2:
        return visitor(llvm::type_identity<evaluate::Type<Complex, 2>>{});
      case 3:
        return visitor(llvm::type_identity<evaluate::Type<Complex, 3>>{});
      case 4:
        return visitor(llvm::type_identity<evaluate::Type<Complex, 4>>{});
      case 8:
        return visitor(llvm::type_identity<evaluate::Type<Complex, 8>>{});
      case 10:
        return visitor(llvm::type_identity<evaluate::Type<Complex, 10>>{});
      case 16:
        return visitor(llvm::type_identity<evaluate::Type<Complex, 16>>{});
      }
      break;
    case common::TypeCategory::Logical:
      switch (type.kind()) {
      case 1:
        return visitor(llvm::type_identity<evaluate::Type<Logical, 1>>{});
      case 2:
        return visitor(llvm::type_identity<evaluate::Type<Logical, 2>>{});
      case 4:
        return visitor(llvm::type_identity<evaluate::Type<Logical, 4>>{});
      case 8:
        return visitor(llvm::type_identity<evaluate::Type<Logical, 8>>{});
      }
      break;
    case common::TypeCategory::Character:
      switch (type.kind()) {
      case 1:
        return visitor(llvm::type_identity<evaluate::Type<Character, 1>>{});
      case 2:
        return visitor(llvm::type_identity<evaluate::Type<Character, 2>>{});
      case 4:
        return visitor(llvm::type_identity<evaluate::Type<Character, 4>>{});
      }
      break;
    case common::TypeCategory::Derived:
      (void)Derived;
      break;
    }
    llvm_unreachable("Unhandled type");
  }

  const evaluate::DynamicType &type;

private:
  // Shorter names.
  static constexpr auto Character = common::TypeCategory::Character;
  static constexpr auto Complex = common::TypeCategory::Complex;
  static constexpr auto Derived = common::TypeCategory::Derived;
  static constexpr auto Integer = common::TypeCategory::Integer;
  static constexpr auto Logical = common::TypeCategory::Logical;
  static constexpr auto Real = common::TypeCategory::Real;
  static constexpr auto Unsigned = common::TypeCategory::Unsigned;
};

template <typename T, typename U = std::remove_const_t<T>>
U AsRvalue(T &t) {
  U copy{t};
  return std::move(copy);
}

template <typename T>
T &&AsRvalue(T &&t) {
  return std::move(t);
}

struct ArgumentReplacer
    : public evaluate::Traverse<ArgumentReplacer, bool, false> {
  using Base = evaluate::Traverse<ArgumentReplacer, bool, false>;
  using Result = bool;

  Result Default() const { return false; }

  ArgumentReplacer(evaluate::ActualArguments &&newArgs)
      : Base(*this), args_(std::move(newArgs)) {}

  using Base::operator();

  template <typename T>
  Result operator()(const evaluate::FunctionRef<T> &x) {
    assert(!done_);
    auto &mut = const_cast<evaluate::FunctionRef<T> &>(x);
    mut.arguments() = args_;
    done_ = true;
    return true;
  }

  Result Combine(Result &&a, Result &&b) { return a || b; }

private:
  bool done_{false};
  evaluate::ActualArguments &&args_;
};
} // namespace

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

static bool replaceArgs(semantics::SomeExpr &expr,
                        evaluate::ActualArguments &&newArgs) {
  return ArgumentReplacer(std::move(newArgs))(expr);
}

static semantics::SomeExpr makeCall(const evaluate::DynamicType &type,
                                    const evaluate::ProcedureDesignator &proc,
                                    const evaluate::ActualArguments &args) {
  return WithType(type).visit([&](auto &&s) -> semantics::SomeExpr {
    using Type = typename llvm::remove_cvref_t<decltype(s)>::type;
    return evaluate::AsGenericExpr(
        evaluate::FunctionRef<Type>(AsRvalue(proc), AsRvalue(args)));
  });
}

static const evaluate::ProcedureDesignator &
getProcedureDesignator(const semantics::SomeExpr &call) {
  const evaluate::ProcedureDesignator *proc = GetProc{}(call);
  assert(proc && "Call has no procedure designator");
  return *proc;
}

static semantics::SomeExpr //
genReducedMinMax(const semantics::SomeExpr &orig,
                 const semantics::SomeExpr *atomArg,
                 const std::vector<semantics::SomeExpr> &args) {
  // Take a list of arguments to a min/max operation, e.g. [a0, a1, ...]
  // One of the a_i's, say a_t, must be atomArg.
  // Generate tmp = min/max(a0, a1, ... [except a_t]). Then generate
  // call = min/max(a_t, tmp).
  // Return "call".

  // The min/max intrinsics have 2 mandatory arguments, the rest is optional.
  // Make sure that the "tmp = min/max(...)" doesn't promote an optional
  // argument to a non-optional position. This could happen if a_t is at
  // position 0 or 1.
  if (args.size() <= 2)
    return orig;

  evaluate::ActualArguments nonAtoms;

  auto AsActual = [](const semantics::SomeExpr &x) {
    semantics::SomeExpr copy = x;
    return evaluate::ActualArgument(std::move(copy));
  };
  // Semantic checks guarantee that the "atom" shows exactly once in the
  // argument list (with potential conversions around it).
  // For the first two (non-optional) arguments, if "atom" is among them,
  // replace it with another occurrence of the other non-optional argument.
  if (atomArg == &args[0]) {
    // (atom, x, y...) -> (x, x, y...)
    nonAtoms.push_back(AsActual(args[1]));
    nonAtoms.push_back(AsActual(args[1]));
  } else if (atomArg == &args[1]) {
    // (x, atom, y...) -> (x, x, y...)
    nonAtoms.push_back(AsActual(args[0]));
    nonAtoms.push_back(AsActual(args[0]));
  } else {
    // (x, y, z...) -> unchanged
    nonAtoms.push_back(AsActual(args[0]));
    nonAtoms.push_back(AsActual(args[1]));
  }

  // The rest of arguments are optional, so we can just skip "atom".
  for (size_t i = 2, e = args.size(); i != e; ++i) {
    if (atomArg != &args[i])
      nonAtoms.push_back(AsActual(args[i]));
  }

  // The type of the intermediate min/max is the same as the type of its
  // arguments, which may be different from the type of the original
  // expression. The original expression may have additional coverts.
  auto tmp =
      makeCall(*atomArg->GetType(), getProcedureDesignator(orig), nonAtoms);
  semantics::SomeExpr call = orig;
  replaceArgs(call, {AsActual(*atomArg), AsActual(tmp)});
  return call;
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
  semantics::SomeExpr rhs = assign.rhs;
  semantics::SomeExpr input = *evaluate::GetConvertInput(rhs);
  auto [opcode, args] = evaluate::GetTopLevelOperation(input);
  assert(!args.empty() && "Update operation without arguments");

  // Pass args as an argument to avoid capturing a structured binding.
  const semantics::SomeExpr *atomArg = [&](auto &args) {
    for (const semantics::SomeExpr &e : args) {
      if (evaluate::IsSameOrConvertOf(e, atom))
        return &e;
    }
    llvm_unreachable("Atomic variable not in argument list");
  }(args);

  if (opcode == evaluate::operation::Operator::Min ||
      opcode == evaluate::operation::Operator::Max) {
    // Min and max operations are expanded inline, so reduce them to
    // operations with exactly two (non-optional) arguments.
    rhs = genReducedMinMax(rhs, atomArg, args);
    input = *evaluate::GetConvertInput(rhs);
    std::tie(opcode, args) = evaluate::GetTopLevelOperation(input);
    atomArg = nullptr; // No longer valid.
  }
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
      fir::getBase(converter.genExprValue(rhs, stmtCtx, &loc));
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
