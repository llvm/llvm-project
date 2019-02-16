//== Z3ConstraintManager.cpp --------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTExpr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTSolver.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTSort.h"

#include "clang/Config/config.h"

using namespace clang;
using namespace ento;

#if CLANG_ANALYZER_WITH_Z3

#include <z3.h>

namespace {

/// Configuration class for Z3
class Z3Config {
  friend class Z3Context;

  Z3_config Config;

public:
  Z3Config() : Config(Z3_mk_config()) {
    // Enable model finding
    Z3_set_param_value(Config, "model", "true");
    // Disable proof generation
    Z3_set_param_value(Config, "proof", "false");
    // Set timeout to 15000ms = 15s
    Z3_set_param_value(Config, "timeout", "15000");
  }

  ~Z3Config() { Z3_del_config(Config); }
}; // end class Z3Config

// Function used to report errors
void Z3ErrorHandler(Z3_context Context, Z3_error_code Error) {
  llvm::report_fatal_error("Z3 error: " +
                           llvm::Twine(Z3_get_error_msg_ex(Context, Error)));
}

/// Wrapper for Z3 context
class Z3Context : public SMTContext {
public:
  Z3_context Context;

  Z3Context() : SMTContext() {
    Context = Z3_mk_context_rc(Z3Config().Config);
    // The error function is set here because the context is the first object
    // created by the backend
    Z3_set_error_handler(Context, Z3ErrorHandler);
  }

  virtual ~Z3Context() {
    Z3_del_context(Context);
    Context = nullptr;
  }
}; // end class Z3Context

/// Wrapper for Z3 Sort
class Z3Sort : public SMTSort {
  friend class Z3Solver;

  Z3Context &Context;

  Z3_sort Sort;

public:
  /// Default constructor, mainly used by make_shared
  Z3Sort(Z3Context &C, Z3_sort ZS) : SMTSort(), Context(C), Sort(ZS) {
    Z3_inc_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
  }

  /// Override implicit copy constructor for correct reference counting.
  Z3Sort(const Z3Sort &Copy)
      : SMTSort(), Context(Copy.Context), Sort(Copy.Sort) {
    Z3_inc_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
  }

  /// Provide move constructor
  Z3Sort(Z3Sort &&Move) : SMTSort(), Context(Move.Context), Sort(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Sort &operator=(Z3Sort &&Move) {
    if (this != &Move) {
      if (Sort)
        Z3_dec_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
      Sort = Move.Sort;
      Move.Sort = nullptr;
    }
    return *this;
  }

  ~Z3Sort() {
    if (Sort)
      Z3_dec_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
  }

  bool isBitvectorSortImpl() const override {
    return (Z3_get_sort_kind(Context.Context, Sort) == Z3_BV_SORT);
  }

  bool isFloatSortImpl() const override {
    return (Z3_get_sort_kind(Context.Context, Sort) == Z3_FLOATING_POINT_SORT);
  }

  bool isBooleanSortImpl() const override {
    return (Z3_get_sort_kind(Context.Context, Sort) == Z3_BOOL_SORT);
  }

  unsigned getBitvectorSortSizeImpl() const override {
    return Z3_get_bv_sort_size(Context.Context, Sort);
  }

  unsigned getFloatSortSizeImpl() const override {
    return Z3_fpa_get_ebits(Context.Context, Sort) +
           Z3_fpa_get_sbits(Context.Context, Sort);
  }

  bool equal_to(SMTSort const &Other) const override {
    return Z3_is_eq_sort(Context.Context, Sort,
                         static_cast<const Z3Sort &>(Other).Sort);
  }

  Z3Sort &operator=(const Z3Sort &Move) {
    Z3_inc_ref(Context.Context, reinterpret_cast<Z3_ast>(Move.Sort));
    Z3_dec_ref(Context.Context, reinterpret_cast<Z3_ast>(Sort));
    Sort = Move.Sort;
    return *this;
  }

  void print(raw_ostream &OS) const override {
    OS << Z3_sort_to_string(Context.Context, Sort);
  }
}; // end class Z3Sort

static const Z3Sort &toZ3Sort(const SMTSort &S) {
  return static_cast<const Z3Sort &>(S);
}

class Z3Expr : public SMTExpr {
  friend class Z3Solver;

  Z3Context &Context;

  Z3_ast AST;

public:
  Z3Expr(Z3Context &C, Z3_ast ZA) : SMTExpr(), Context(C), AST(ZA) {
    Z3_inc_ref(Context.Context, AST);
  }

  /// Override implicit copy constructor for correct reference counting.
  Z3Expr(const Z3Expr &Copy) : SMTExpr(), Context(Copy.Context), AST(Copy.AST) {
    Z3_inc_ref(Context.Context, AST);
  }

  /// Provide move constructor
  Z3Expr(Z3Expr &&Move) : SMTExpr(), Context(Move.Context), AST(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Expr &operator=(Z3Expr &&Move) {
    if (this != &Move) {
      if (AST)
        Z3_dec_ref(Context.Context, AST);
      AST = Move.AST;
      Move.AST = nullptr;
    }
    return *this;
  }

  ~Z3Expr() {
    if (AST)
      Z3_dec_ref(Context.Context, AST);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddInteger(Z3_get_ast_hash(Context.Context, AST));
  }

  /// Comparison of AST equality, not model equivalence.
  bool equal_to(SMTExpr const &Other) const override {
    assert(Z3_is_eq_sort(Context.Context, Z3_get_sort(Context.Context, AST),
                         Z3_get_sort(Context.Context,
                                     static_cast<const Z3Expr &>(Other).AST)) &&
           "AST's must have the same sort");
    return Z3_is_eq_ast(Context.Context, AST,
                        static_cast<const Z3Expr &>(Other).AST);
  }

  /// Override implicit move constructor for correct reference counting.
  Z3Expr &operator=(const Z3Expr &Move) {
    Z3_inc_ref(Context.Context, Move.AST);
    Z3_dec_ref(Context.Context, AST);
    AST = Move.AST;
    return *this;
  }

  void print(raw_ostream &OS) const override {
    OS << Z3_ast_to_string(Context.Context, AST);
  }
}; // end class Z3Expr

static const Z3Expr &toZ3Expr(const SMTExpr &E) {
  return static_cast<const Z3Expr &>(E);
}

class Z3Model {
  friend class Z3Solver;

  Z3Context &Context;

  Z3_model Model;

public:
  Z3Model(Z3Context &C, Z3_model ZM) : Context(C), Model(ZM) {
    assert(C.Context != nullptr);
    Z3_model_inc_ref(Context.Context, Model);
  }

  /// Override implicit copy constructor for correct reference counting.
  Z3Model(const Z3Model &Copy) : Context(Copy.Context), Model(Copy.Model) {
    Z3_model_inc_ref(Context.Context, Model);
  }

  /// Provide move constructor
  Z3Model(Z3Model &&Move) : Context(Move.Context), Model(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Model &operator=(Z3Model &&Move) {
    if (this != &Move) {
      if (Model)
        Z3_model_dec_ref(Context.Context, Model);
      Model = Move.Model;
      Move.Model = nullptr;
    }
    return *this;
  }

  ~Z3Model() {
    if (Model)
      Z3_model_dec_ref(Context.Context, Model);
  }

  void print(raw_ostream &OS) const {
    OS << Z3_model_to_string(Context.Context, Model);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Model

/// Get the corresponding IEEE floating-point type for a given bitwidth.
static const llvm::fltSemantics &getFloatSemantics(unsigned BitWidth) {
  switch (BitWidth) {
  default:
    llvm_unreachable("Unsupported floating-point semantics!");
    break;
  case 16:
    return llvm::APFloat::IEEEhalf();
  case 32:
    return llvm::APFloat::IEEEsingle();
  case 64:
    return llvm::APFloat::IEEEdouble();
  case 128:
    return llvm::APFloat::IEEEquad();
  }
}

// Determine whether two float semantics are equivalent
static bool areEquivalent(const llvm::fltSemantics &LHS,
                          const llvm::fltSemantics &RHS) {
  return (llvm::APFloat::semanticsPrecision(LHS) ==
          llvm::APFloat::semanticsPrecision(RHS)) &&
         (llvm::APFloat::semanticsMinExponent(LHS) ==
          llvm::APFloat::semanticsMinExponent(RHS)) &&
         (llvm::APFloat::semanticsMaxExponent(LHS) ==
          llvm::APFloat::semanticsMaxExponent(RHS)) &&
         (llvm::APFloat::semanticsSizeInBits(LHS) ==
          llvm::APFloat::semanticsSizeInBits(RHS));
}

} // end anonymous namespace

typedef llvm::ImmutableSet<std::pair<SymbolRef, Z3Expr>> ConstraintZ3Ty;
REGISTER_TRAIT_WITH_PROGRAMSTATE(ConstraintZ3, ConstraintZ3Ty)

namespace {

class Z3Solver : public SMTSolver {
  friend class Z3ConstraintManager;

  Z3Context Context;

  Z3_solver Solver;

public:
  Z3Solver() : SMTSolver(), Solver(Z3_mk_simple_solver(Context.Context)) {
    Z3_solver_inc_ref(Context.Context, Solver);
  }

  /// Override implicit copy constructor for correct reference counting.
  Z3Solver(const Z3Solver &Copy)
      : SMTSolver(), Context(Copy.Context), Solver(Copy.Solver) {
    Z3_solver_inc_ref(Context.Context, Solver);
  }

  /// Provide move constructor
  Z3Solver(Z3Solver &&Move)
      : SMTSolver(), Context(Move.Context), Solver(nullptr) {
    *this = std::move(Move);
  }

  /// Provide move assignment constructor
  Z3Solver &operator=(Z3Solver &&Move) {
    if (this != &Move) {
      if (Solver)
        Z3_solver_dec_ref(Context.Context, Solver);
      Solver = Move.Solver;
      Move.Solver = nullptr;
    }
    return *this;
  }

  ~Z3Solver() {
    if (Solver)
      Z3_solver_dec_ref(Context.Context, Solver);
  }

  void addConstraint(const SMTExprRef &Exp) const override {
    Z3_solver_assert(Context.Context, Solver, toZ3Expr(*Exp).AST);
  }

  SMTSortRef getBoolSort() override {
    return std::make_shared<Z3Sort>(Context, Z3_mk_bool_sort(Context.Context));
  }

  SMTSortRef getBitvectorSort(unsigned BitWidth) override {
    return std::make_shared<Z3Sort>(Context,
                                    Z3_mk_bv_sort(Context.Context, BitWidth));
  }

  SMTSortRef getSort(const SMTExprRef &Exp) override {
    return std::make_shared<Z3Sort>(
        Context, Z3_get_sort(Context.Context, toZ3Expr(*Exp).AST));
  }

  SMTSortRef getFloat16Sort() override {
    return std::make_shared<Z3Sort>(Context,
                                    Z3_mk_fpa_sort_16(Context.Context));
  }

  SMTSortRef getFloat32Sort() override {
    return std::make_shared<Z3Sort>(Context,
                                    Z3_mk_fpa_sort_32(Context.Context));
  }

  SMTSortRef getFloat64Sort() override {
    return std::make_shared<Z3Sort>(Context,
                                    Z3_mk_fpa_sort_64(Context.Context));
  }

  SMTSortRef getFloat128Sort() override {
    return std::make_shared<Z3Sort>(Context,
                                    Z3_mk_fpa_sort_128(Context.Context));
  }

  SMTExprRef newExprRef(const SMTExpr &E) const override {
    return std::make_shared<Z3Expr>(toZ3Expr(E));
  }

  SMTExprRef mkBVNeg(const SMTExprRef &Exp) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvneg(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkBVNot(const SMTExprRef &Exp) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvnot(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkNot(const SMTExprRef &Exp) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_not(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkBVAdd(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvadd(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSub(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvsub(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVMul(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvmul(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSRem(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvsrem(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVURem(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvurem(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvsdiv(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVUDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvudiv(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVShl(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvshl(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVAshr(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvashr(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVLshr(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvlshr(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVXor(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvxor(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVOr(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvor(Context.Context, toZ3Expr(*LHS).AST,
                                   toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVAnd(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvand(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVUlt(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvult(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSlt(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvslt(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVUgt(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvugt(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSgt(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvsgt(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVUle(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvule(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSle(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvsle(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVUge(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvuge(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkBVSge(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_bvsge(Context.Context, toZ3Expr(*LHS).AST,
                                    toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkAnd(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    Z3_ast Args[2] = {toZ3Expr(*LHS).AST, toZ3Expr(*RHS).AST};
    return newExprRef(Z3Expr(Context, Z3_mk_and(Context.Context, 2, Args)));
  }

  SMTExprRef mkOr(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    Z3_ast Args[2] = {toZ3Expr(*LHS).AST, toZ3Expr(*RHS).AST};
    return newExprRef(Z3Expr(Context, Z3_mk_or(Context.Context, 2, Args)));
  }

  SMTExprRef mkEqual(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_eq(Context.Context, toZ3Expr(*LHS).AST,
                                 toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPNeg(const SMTExprRef &Exp) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_neg(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkFPIsInfinite(const SMTExprRef &Exp) override {
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_is_infinite(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkFPIsNaN(const SMTExprRef &Exp) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_is_nan(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkFPIsNormal(const SMTExprRef &Exp) override {
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_is_normal(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkFPIsZero(const SMTExprRef &Exp) override {
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_is_zero(Context.Context, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkFPMul(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(
        Z3Expr(Context,
               Z3_mk_fpa_mul(Context.Context, toZ3Expr(*LHS).AST,
                             toZ3Expr(*RHS).AST, toZ3Expr(*RoundingMode).AST)));
  }

  SMTExprRef mkFPDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(
        Z3Expr(Context,
               Z3_mk_fpa_div(Context.Context, toZ3Expr(*LHS).AST,
                             toZ3Expr(*RHS).AST, toZ3Expr(*RoundingMode).AST)));
  }

  SMTExprRef mkFPRem(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_rem(Context.Context, toZ3Expr(*LHS).AST,
                                      toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPAdd(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(
        Z3Expr(Context,
               Z3_mk_fpa_add(Context.Context, toZ3Expr(*LHS).AST,
                             toZ3Expr(*RHS).AST, toZ3Expr(*RoundingMode).AST)));
  }

  SMTExprRef mkFPSub(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(
        Z3Expr(Context,
               Z3_mk_fpa_sub(Context.Context, toZ3Expr(*LHS).AST,
                             toZ3Expr(*RHS).AST, toZ3Expr(*RoundingMode).AST)));
  }

  SMTExprRef mkFPLt(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_lt(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPGt(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_gt(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPLe(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_leq(Context.Context, toZ3Expr(*LHS).AST,
                                      toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPGe(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_geq(Context.Context, toZ3Expr(*LHS).AST,
                                      toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPEqual(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_fpa_eq(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkIte(const SMTExprRef &Cond, const SMTExprRef &T,
                   const SMTExprRef &F) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_ite(Context.Context, toZ3Expr(*Cond).AST,
                                  toZ3Expr(*T).AST, toZ3Expr(*F).AST)));
  }

  SMTExprRef mkBVSignExt(unsigned i, const SMTExprRef &Exp) override {
    return newExprRef(Z3Expr(
        Context, Z3_mk_sign_ext(Context.Context, i, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkBVZeroExt(unsigned i, const SMTExprRef &Exp) override {
    return newExprRef(Z3Expr(
        Context, Z3_mk_zero_ext(Context.Context, i, toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkBVExtract(unsigned High, unsigned Low,
                         const SMTExprRef &Exp) override {
    return newExprRef(Z3Expr(Context, Z3_mk_extract(Context.Context, High, Low,
                                                    toZ3Expr(*Exp).AST)));
  }

  SMTExprRef mkBVConcat(const SMTExprRef &LHS, const SMTExprRef &RHS) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_concat(Context.Context, toZ3Expr(*LHS).AST,
                                     toZ3Expr(*RHS).AST)));
  }

  SMTExprRef mkFPtoFP(const SMTExprRef &From, const SMTSortRef &To) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(Z3Expr(
        Context,
        Z3_mk_fpa_to_fp_float(Context.Context, toZ3Expr(*RoundingMode).AST,
                              toZ3Expr(*From).AST, toZ3Sort(*To).Sort)));
  }

  SMTExprRef mkFPtoSBV(const SMTExprRef &From, const SMTSortRef &To) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(Z3Expr(
        Context,
        Z3_mk_fpa_to_fp_signed(Context.Context, toZ3Expr(*RoundingMode).AST,
                               toZ3Expr(*From).AST, toZ3Sort(*To).Sort)));
  }

  SMTExprRef mkFPtoUBV(const SMTExprRef &From, const SMTSortRef &To) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(Z3Expr(
        Context,
        Z3_mk_fpa_to_fp_unsigned(Context.Context, toZ3Expr(*RoundingMode).AST,
                                 toZ3Expr(*From).AST, toZ3Sort(*To).Sort)));
  }

  SMTExprRef mkSBVtoFP(const SMTExprRef &From, unsigned ToWidth) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_to_sbv(Context.Context, toZ3Expr(*RoundingMode).AST,
                                  toZ3Expr(*From).AST, ToWidth)));
  }

  SMTExprRef mkUBVtoFP(const SMTExprRef &From, unsigned ToWidth) override {
    SMTExprRef RoundingMode = getFloatRoundingMode();
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_to_ubv(Context.Context, toZ3Expr(*RoundingMode).AST,
                                  toZ3Expr(*From).AST, ToWidth)));
  }

  SMTExprRef mkBoolean(const bool b) override {
    return newExprRef(Z3Expr(Context, b ? Z3_mk_true(Context.Context)
                                        : Z3_mk_false(Context.Context)));
  }

  SMTExprRef mkBitvector(const llvm::APSInt Int, unsigned BitWidth) override {
    const SMTSortRef Sort = getBitvectorSort(BitWidth);
    return newExprRef(
        Z3Expr(Context, Z3_mk_numeral(Context.Context, Int.toString(10).c_str(),
                                      toZ3Sort(*Sort).Sort)));
  }

  SMTExprRef mkFloat(const llvm::APFloat Float) override {
    SMTSortRef Sort =
        getFloatSort(llvm::APFloat::semanticsSizeInBits(Float.getSemantics()));

    llvm::APSInt Int = llvm::APSInt(Float.bitcastToAPInt(), false);
    SMTExprRef Z3Int = mkBitvector(Int, Int.getBitWidth());
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_to_fp_bv(Context.Context, toZ3Expr(*Z3Int).AST,
                                    toZ3Sort(*Sort).Sort)));
  }

  SMTExprRef mkSymbol(const char *Name, SMTSortRef Sort) override {
    return newExprRef(
        Z3Expr(Context, Z3_mk_const(Context.Context,
                                    Z3_mk_string_symbol(Context.Context, Name),
                                    toZ3Sort(*Sort).Sort)));
  }

  llvm::APSInt getBitvector(const SMTExprRef &Exp, unsigned BitWidth,
                            bool isUnsigned) override {
    return llvm::APSInt(llvm::APInt(
        BitWidth, Z3_get_numeral_string(Context.Context, toZ3Expr(*Exp).AST),
        10));
  }

  bool getBoolean(const SMTExprRef &Exp) override {
    return Z3_get_bool_value(Context.Context, toZ3Expr(*Exp).AST) == Z3_L_TRUE;
  }

  SMTExprRef getFloatRoundingMode() override {
    // TODO: Don't assume nearest ties to even rounding mode
    return newExprRef(Z3Expr(Context, Z3_mk_fpa_rne(Context.Context)));
  }

  SMTExprRef fromData(const SymbolID ID, const QualType &Ty,
                      uint64_t BitWidth) override {
    llvm::Twine Name = "$" + llvm::Twine(ID);
    return mkSymbol(Name.str().c_str(), mkSort(Ty, BitWidth));
  }

  SMTExprRef fromBoolean(const bool Bool) override {
    Z3_ast AST =
        Bool ? Z3_mk_true(Context.Context) : Z3_mk_false(Context.Context);
    return newExprRef(Z3Expr(Context, AST));
  }

  SMTExprRef fromAPFloat(const llvm::APFloat &Float) override {
    SMTSortRef Sort =
        getFloatSort(llvm::APFloat::semanticsSizeInBits(Float.getSemantics()));

    llvm::APSInt Int = llvm::APSInt(Float.bitcastToAPInt(), false);
    SMTExprRef Z3Int = fromAPSInt(Int);
    return newExprRef(Z3Expr(
        Context, Z3_mk_fpa_to_fp_bv(Context.Context, toZ3Expr(*Z3Int).AST,
                                    toZ3Sort(*Sort).Sort)));
  }

  SMTExprRef fromAPSInt(const llvm::APSInt &Int) override {
    SMTSortRef Sort = getBitvectorSort(Int.getBitWidth());
    Z3_ast AST = Z3_mk_numeral(Context.Context, Int.toString(10).c_str(),
                               toZ3Sort(*Sort).Sort);
    return newExprRef(Z3Expr(Context, AST));
  }

  SMTExprRef fromInt(const char *Int, uint64_t BitWidth) override {
    SMTSortRef Sort = getBitvectorSort(BitWidth);
    Z3_ast AST = Z3_mk_numeral(Context.Context, Int, toZ3Sort(*Sort).Sort);
    return newExprRef(Z3Expr(Context, AST));
  }

  bool toAPFloat(const SMTSortRef &Sort, const SMTExprRef &AST,
                 llvm::APFloat &Float, bool useSemantics) {
    assert(Sort->isFloatSort() && "Unsupported sort to floating-point!");

    llvm::APSInt Int(Sort->getFloatSortSize(), true);
    const llvm::fltSemantics &Semantics =
        getFloatSemantics(Sort->getFloatSortSize());
    SMTSortRef BVSort = getBitvectorSort(Sort->getFloatSortSize());
    if (!toAPSInt(BVSort, AST, Int, true)) {
      return false;
    }

    if (useSemantics && !areEquivalent(Float.getSemantics(), Semantics)) {
      assert(false && "Floating-point types don't match!");
      return false;
    }

    Float = llvm::APFloat(Semantics, Int);
    return true;
  }

  bool toAPSInt(const SMTSortRef &Sort, const SMTExprRef &AST,
                llvm::APSInt &Int, bool useSemantics) {
    if (Sort->isBitvectorSort()) {
      if (useSemantics && Int.getBitWidth() != Sort->getBitvectorSortSize()) {
        assert(false && "Bitvector types don't match!");
        return false;
      }

      // FIXME: This function is also used to retrieve floating-point values,
      // which can be 16, 32, 64 or 128 bits long. Bitvectors can be anything
      // between 1 and 64 bits long, which is the reason we have this weird
      // guard. In the future, we need proper calls in the backend to retrieve
      // floating-points and its special values (NaN, +/-infinity, +/-zero),
      // then we can drop this weird condition.
      if (Sort->getBitvectorSortSize() <= 64 ||
          Sort->getBitvectorSortSize() == 128) {
        Int = getBitvector(AST, Int.getBitWidth(), Int.isUnsigned());
        return true;
      }

      assert(false && "Bitwidth not supported!");
      return false;
    }

    if (Sort->isBooleanSort()) {
      if (useSemantics && Int.getBitWidth() < 1) {
        assert(false && "Boolean type doesn't match!");
        return false;
      }

      Int = llvm::APSInt(llvm::APInt(Int.getBitWidth(), getBoolean(AST)),
                         Int.isUnsigned());
      return true;
    }

    llvm_unreachable("Unsupported sort to integer!");
  }

  bool getInterpretation(const SMTExprRef &Exp, llvm::APSInt &Int) override {
    Z3Model Model = getModel();
    Z3_func_decl Func = Z3_get_app_decl(
        Context.Context, Z3_to_app(Context.Context, toZ3Expr(*Exp).AST));
    if (Z3_model_has_interp(Context.Context, Model.Model, Func) != Z3_L_TRUE)
      return false;

    SMTExprRef Assign = newExprRef(
        Z3Expr(Context,
               Z3_model_get_const_interp(Context.Context, Model.Model, Func)));
    SMTSortRef Sort = getSort(Assign);
    return toAPSInt(Sort, Assign, Int, true);
  }

  bool getInterpretation(const SMTExprRef &Exp, llvm::APFloat &Float) override {
    Z3Model Model = getModel();
    Z3_func_decl Func = Z3_get_app_decl(
        Context.Context, Z3_to_app(Context.Context, toZ3Expr(*Exp).AST));
    if (Z3_model_has_interp(Context.Context, Model.Model, Func) != Z3_L_TRUE)
      return false;

    SMTExprRef Assign = newExprRef(
        Z3Expr(Context,
               Z3_model_get_const_interp(Context.Context, Model.Model, Func)));
    SMTSortRef Sort = getSort(Assign);
    return toAPFloat(Sort, Assign, Float, true);
  }

  ConditionTruthVal check() const override {
    Z3_lbool res = Z3_solver_check(Context.Context, Solver);
    if (res == Z3_L_TRUE)
      return true;

    if (res == Z3_L_FALSE)
      return false;

    return ConditionTruthVal();
  }

  void push() override { return Z3_solver_push(Context.Context, Solver); }

  void pop(unsigned NumStates = 1) override {
    assert(Z3_solver_get_num_scopes(Context.Context, Solver) >= NumStates);
    return Z3_solver_pop(Context.Context, Solver, NumStates);
  }

  /// Get a model from the solver. Caller should check the model is
  /// satisfiable.
  Z3Model getModel() {
    return Z3Model(Context, Z3_solver_get_model(Context.Context, Solver));
  }

  /// Reset the solver and remove all constraints.
  void reset() const override { Z3_solver_reset(Context.Context, Solver); }

  void print(raw_ostream &OS) const override {
    OS << Z3_solver_to_string(Context.Context, Solver);
  }
}; // end class Z3Solver

class Z3ConstraintManager : public SMTConstraintManager {
  SMTSolverRef Solver = CreateZ3Solver();

public:
  Z3ConstraintManager(SubEngine *SE, SValBuilder &SB)
      : SMTConstraintManager(SE, SB, Solver) {}

  void addStateConstraints(ProgramStateRef State) const override {
    // TODO: Don't add all the constraints, only the relevant ones
    ConstraintZ3Ty CZ = State->get<ConstraintZ3>();
    ConstraintZ3Ty::iterator I = CZ.begin(), IE = CZ.end();

    // Construct the logical AND of all the constraints
    if (I != IE) {
      std::vector<SMTExprRef> ASTs;

      SMTExprRef Constraint = Solver->newExprRef(I++->second);
      while (I != IE) {
        Constraint = Solver->mkAnd(Constraint, Solver->newExprRef(I++->second));
      }

      Solver->addConstraint(Constraint);
    }
  }

  bool canReasonAbout(SVal X) const override {
    const TargetInfo &TI = getBasicVals().getContext().getTargetInfo();

    Optional<nonloc::SymbolVal> SymVal = X.getAs<nonloc::SymbolVal>();
    if (!SymVal)
      return true;

    const SymExpr *Sym = SymVal->getSymbol();
    QualType Ty = Sym->getType();

    // Complex types are not modeled
    if (Ty->isComplexType() || Ty->isComplexIntegerType())
      return false;

    // Non-IEEE 754 floating-point types are not modeled
    if ((Ty->isSpecificBuiltinType(BuiltinType::LongDouble) &&
         (&TI.getLongDoubleFormat() == &llvm::APFloat::x87DoubleExtended() ||
          &TI.getLongDoubleFormat() == &llvm::APFloat::PPCDoubleDouble())))
      return false;

    if (isa<SymbolData>(Sym))
      return true;

    SValBuilder &SVB = getSValBuilder();

    if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym))
      return canReasonAbout(SVB.makeSymbolVal(SC->getOperand()));

    if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
      if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE))
        return canReasonAbout(SVB.makeSymbolVal(SIE->getLHS()));

      if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE))
        return canReasonAbout(SVB.makeSymbolVal(ISE->getRHS()));

      if (const SymSymExpr *SSE = dyn_cast<SymSymExpr>(BSE))
        return canReasonAbout(SVB.makeSymbolVal(SSE->getLHS())) &&
               canReasonAbout(SVB.makeSymbolVal(SSE->getRHS()));
    }

    llvm_unreachable("Unsupported expression to reason about!");
  }

  ProgramStateRef removeDeadBindings(ProgramStateRef State,
                                     SymbolReaper &SymReaper) override {
    ConstraintZ3Ty CZ = State->get<ConstraintZ3>();
    ConstraintZ3Ty::Factory &CZFactory = State->get_context<ConstraintZ3>();

    for (ConstraintZ3Ty::iterator I = CZ.begin(), E = CZ.end(); I != E; ++I) {
      if (SymReaper.maybeDead(I->first))
        CZ = CZFactory.remove(CZ, *I);
    }

    return State->set<ConstraintZ3>(CZ);
  }

  ProgramStateRef assumeExpr(ProgramStateRef State, SymbolRef Sym,
                             const SMTExprRef &Exp) override {
    // Check the model, avoid simplifying AST to save time
    if (checkModel(State, Exp).isConstrainedTrue())
      return State->add<ConstraintZ3>(std::make_pair(Sym, toZ3Expr(*Exp)));

    return nullptr;
  }

  //==------------------------------------------------------------------------==/
  // Pretty-printing.
  //==------------------------------------------------------------------------==/

  void print(ProgramStateRef St, raw_ostream &OS, const char *nl,
             const char *sep) override {

    ConstraintZ3Ty CZ = St->get<ConstraintZ3>();

    OS << nl << sep << "Constraints:";
    for (ConstraintZ3Ty::iterator I = CZ.begin(), E = CZ.end(); I != E; ++I) {
      OS << nl << ' ' << I->first << " : ";
      I->second.print(OS);
    }
    OS << nl;
  }
}; // end class Z3ConstraintManager

} // end anonymous namespace

#endif

std::unique_ptr<SMTSolver> clang::ento::CreateZ3Solver() {
#if CLANG_ANALYZER_WITH_Z3
  return llvm::make_unique<Z3Solver>();
#else
  llvm::report_fatal_error("Clang was not compiled with Z3 support, rebuild "
                           "with -DCLANG_ANALYZER_BUILD_Z3=ON",
                           false);
  return nullptr;
#endif
}

std::unique_ptr<ConstraintManager>
ento::CreateZ3ConstraintManager(ProgramStateManager &StMgr, SubEngine *Eng) {
#if CLANG_ANALYZER_WITH_Z3
  return llvm::make_unique<Z3ConstraintManager>(Eng, StMgr.getSValBuilder());
#else
  llvm::report_fatal_error("Clang was not compiled with Z3 support, rebuild "
                           "with -DCLANG_ANALYZER_BUILD_Z3=ON",
                           false);
  return nullptr;
#endif
}
