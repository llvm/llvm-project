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

#include "clang/Config/config.h"

using namespace clang;
using namespace ento;

#if CLANG_ANALYZER_WITH_Z3

#include <z3.h>

// Forward declarations
namespace {
class Z3Expr;
class ConstraintZ3 {};
} // end anonymous namespace

typedef llvm::ImmutableSet<std::pair<SymbolRef, Z3Expr>> ConstraintZ3Ty;

// Expansion of REGISTER_TRAIT_WITH_PROGRAMSTATE(ConstraintZ3, Z3SetPair)
namespace clang {
namespace ento {
template <>
struct ProgramStateTrait<ConstraintZ3>
    : public ProgramStatePartialTrait<ConstraintZ3Ty> {
  static void *GDMIndex() {
    static int Index;
    return &Index;
  }
};
} // end namespace ento
} // end namespace clang

namespace {

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

class Z3Context {
  Z3_context ZC_P;

public:
  static Z3_context ZC;

  Z3Context() : ZC_P(Z3_mk_context_rc(Z3Config().Config)) { ZC = ZC_P; }

  ~Z3Context() {
    Z3_del_context(ZC);
    Z3_finalize_memory();
    ZC_P = nullptr;
  }
}; // end class Z3Context

class Z3Sort {
  friend class Z3Expr;

  Z3_sort Sort;

  Z3Sort() : Sort(nullptr) {}
  Z3Sort(Z3_sort ZS) : Sort(ZS) {
    Z3_inc_ref(Z3Context::ZC, reinterpret_cast<Z3_ast>(Sort));
  }

public:
  /// Override implicit copy constructor for correct reference counting.
  Z3Sort(const Z3Sort &Copy) : Sort(Copy.Sort) {
    Z3_inc_ref(Z3Context::ZC, reinterpret_cast<Z3_ast>(Sort));
  }

  /// Provide move constructor
  Z3Sort(Z3Sort &&Move) : Sort(nullptr) { *this = std::move(Move); }

  /// Provide move assignment constructor
  Z3Sort &operator=(Z3Sort &&Move) {
    if (this != &Move) {
      if (Sort)
        Z3_dec_ref(Z3Context::ZC, reinterpret_cast<Z3_ast>(Sort));
      Sort = Move.Sort;
      Move.Sort = nullptr;
    }
    return *this;
  }

  ~Z3Sort() {
    if (Sort)
      Z3_dec_ref(Z3Context::ZC, reinterpret_cast<Z3_ast>(Sort));
  }

  // Return a boolean sort.
  static Z3Sort getBoolSort() { return Z3Sort(Z3_mk_bool_sort(Z3Context::ZC)); }

  // Return an appropriate bitvector sort for the given bitwidth.
  static Z3Sort getBitvectorSort(unsigned BitWidth) {
    return Z3Sort(Z3_mk_bv_sort(Z3Context::ZC, BitWidth));
  }

  // Return an appropriate floating-point sort for the given bitwidth.
  static Z3Sort getFloatSort(unsigned BitWidth) {
    Z3_sort Sort;

    switch (BitWidth) {
    default:
      llvm_unreachable("Unsupported floating-point bitwidth!");
      break;
    case 16:
      Sort = Z3_mk_fpa_sort_16(Z3Context::ZC);
      break;
    case 32:
      Sort = Z3_mk_fpa_sort_32(Z3Context::ZC);
      break;
    case 64:
      Sort = Z3_mk_fpa_sort_64(Z3Context::ZC);
      break;
    case 128:
      Sort = Z3_mk_fpa_sort_128(Z3Context::ZC);
      break;
    }
    return Z3Sort(Sort);
  }

  // Return an appropriate sort for the given AST.
  static Z3Sort getSort(Z3_ast AST) {
    return Z3Sort(Z3_get_sort(Z3Context::ZC, AST));
  }

  Z3_sort_kind getSortKind() const {
    return Z3_get_sort_kind(Z3Context::ZC, Sort);
  }

  unsigned getBitvectorSortSize() const {
    assert(getSortKind() == Z3_BV_SORT && "Not a bitvector sort!");
    return Z3_get_bv_sort_size(Z3Context::ZC, Sort);
  }

  unsigned getFloatSortSize() const {
    assert(getSortKind() == Z3_FLOATING_POINT_SORT &&
           "Not a floating-point sort!");
    return Z3_fpa_get_ebits(Z3Context::ZC, Sort) +
           Z3_fpa_get_sbits(Z3Context::ZC, Sort);
  }

  bool operator==(const Z3Sort &Other) const {
    return Z3_is_eq_sort(Z3Context::ZC, Sort, Other.Sort);
  }

  Z3Sort &operator=(const Z3Sort &Move) {
    Z3_inc_ref(Z3Context::ZC, reinterpret_cast<Z3_ast>(Move.Sort));
    Z3_dec_ref(Z3Context::ZC, reinterpret_cast<Z3_ast>(Sort));
    Sort = Move.Sort;
    return *this;
  }

  void print(raw_ostream &OS) const {
    OS << Z3_sort_to_string(Z3Context::ZC, Sort);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Sort

class Z3Expr {
  friend class Z3Model;
  friend class Z3Solver;

  Z3_ast AST;

  Z3Expr(Z3_ast ZA) : AST(ZA) { Z3_inc_ref(Z3Context::ZC, AST); }

  // Return an appropriate floating-point rounding mode.
  static Z3Expr getFloatRoundingMode() {
    // TODO: Don't assume nearest ties to even rounding mode
    return Z3Expr(Z3_mk_fpa_rne(Z3Context::ZC));
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

public:
  /// Override implicit copy constructor for correct reference counting.
  Z3Expr(const Z3Expr &Copy) : AST(Copy.AST) { Z3_inc_ref(Z3Context::ZC, AST); }

  /// Provide move constructor
  Z3Expr(Z3Expr &&Move) : AST(nullptr) { *this = std::move(Move); }

  /// Provide move assignment constructor
  Z3Expr &operator=(Z3Expr &&Move) {
    if (this != &Move) {
      if (AST)
        Z3_dec_ref(Z3Context::ZC, AST);
      AST = Move.AST;
      Move.AST = nullptr;
    }
    return *this;
  }

  ~Z3Expr() {
    if (AST)
      Z3_dec_ref(Z3Context::ZC, AST);
  }

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

  /// Construct a Z3Expr from a unary operator, given a Z3_context.
  static Z3Expr fromUnOp(const UnaryOperator::Opcode Op, const Z3Expr &Exp) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    case UO_Minus:
      AST = Z3_mk_bvneg(Z3Context::ZC, Exp.AST);
      break;

    case UO_Not:
      AST = Z3_mk_bvnot(Z3Context::ZC, Exp.AST);
      break;

    case UO_LNot:
      AST = Z3_mk_not(Z3Context::ZC, Exp.AST);
      break;
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a floating-point unary operator, given a
  /// Z3_context.
  static Z3Expr fromFloatUnOp(const UnaryOperator::Opcode Op,
                              const Z3Expr &Exp) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    case UO_Minus:
      AST = Z3_mk_fpa_neg(Z3Context::ZC, Exp.AST);
      break;

    case UO_LNot:
      return Z3Expr::fromUnOp(Op, Exp);
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a n-ary binary operator.
  static Z3Expr fromNBinOp(const BinaryOperator::Opcode Op,
                           const std::vector<Z3_ast> &ASTs) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    case BO_LAnd:
      AST = Z3_mk_and(Z3Context::ZC, ASTs.size(), ASTs.data());
      break;

    case BO_LOr:
      AST = Z3_mk_or(Z3Context::ZC, ASTs.size(), ASTs.data());
      break;
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a binary operator, given a Z3_context.
  static Z3Expr fromBinOp(const Z3Expr &LHS, const BinaryOperator::Opcode Op,
                          const Z3Expr &RHS, bool isSigned) {
    Z3_ast AST;

    assert(Z3Sort::getSort(LHS.AST) == Z3Sort::getSort(RHS.AST) &&
           "AST's must have the same sort!");

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    // Multiplicative operators
    case BO_Mul:
      AST = Z3_mk_bvmul(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_Div:
      AST = isSigned ? Z3_mk_bvsdiv(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvudiv(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_Rem:
      AST = isSigned ? Z3_mk_bvsrem(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvurem(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Additive operators
    case BO_Add:
      AST = Z3_mk_bvadd(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_Sub:
      AST = Z3_mk_bvsub(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Bitwise shift operators
    case BO_Shl:
      AST = Z3_mk_bvshl(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_Shr:
      AST = isSigned ? Z3_mk_bvashr(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvlshr(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Relational operators
    case BO_LT:
      AST = isSigned ? Z3_mk_bvslt(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvult(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_GT:
      AST = isSigned ? Z3_mk_bvsgt(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvugt(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_LE:
      AST = isSigned ? Z3_mk_bvsle(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvule(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_GE:
      AST = isSigned ? Z3_mk_bvsge(Z3Context::ZC, LHS.AST, RHS.AST)
                     : Z3_mk_bvuge(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Equality operators
    case BO_EQ:
      AST = Z3_mk_eq(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_NE:
      return Z3Expr::fromUnOp(UO_LNot,
                              Z3Expr::fromBinOp(LHS, BO_EQ, RHS, isSigned));
      break;

    // Bitwise operators
    case BO_And:
      AST = Z3_mk_bvand(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_Xor:
      AST = Z3_mk_bvxor(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_Or:
      AST = Z3_mk_bvor(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Logical operators
    case BO_LAnd:
    case BO_LOr: {
      std::vector<Z3_ast> Args = {LHS.AST, RHS.AST};
      return Z3Expr::fromNBinOp(Op, Args);
    }
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a special floating-point binary operator, given
  /// a Z3_context.
  static Z3Expr fromFloatSpecialBinOp(const Z3Expr &LHS,
                                      const BinaryOperator::Opcode Op,
                                      const llvm::APFloat::fltCategory &RHS) {
    Z3_ast AST;

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    // Equality operators
    case BO_EQ:
      switch (RHS) {
      case llvm::APFloat::fcInfinity:
        AST = Z3_mk_fpa_is_infinite(Z3Context::ZC, LHS.AST);
        break;
      case llvm::APFloat::fcNaN:
        AST = Z3_mk_fpa_is_nan(Z3Context::ZC, LHS.AST);
        break;
      case llvm::APFloat::fcNormal:
        AST = Z3_mk_fpa_is_normal(Z3Context::ZC, LHS.AST);
        break;
      case llvm::APFloat::fcZero:
        AST = Z3_mk_fpa_is_zero(Z3Context::ZC, LHS.AST);
        break;
      }
      break;
    case BO_NE:
      return Z3Expr::fromFloatUnOp(
          UO_LNot, Z3Expr::fromFloatSpecialBinOp(LHS, BO_EQ, RHS));
      break;
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a floating-point binary operator, given a
  /// Z3_context.
  static Z3Expr fromFloatBinOp(const Z3Expr &LHS,
                               const BinaryOperator::Opcode Op,
                               const Z3Expr &RHS) {
    Z3_ast AST;

    assert(Z3Sort::getSort(LHS.AST) == Z3Sort::getSort(RHS.AST) &&
           "AST's must have the same sort!");

    switch (Op) {
    default:
      llvm_unreachable("Unimplemented opcode");
      break;

    // Multiplicative operators
    case BO_Mul: {
      Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
      AST = Z3_mk_fpa_mul(Z3Context::ZC, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }
    case BO_Div: {
      Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
      AST = Z3_mk_fpa_div(Z3Context::ZC, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }
    case BO_Rem:
      AST = Z3_mk_fpa_rem(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Additive operators
    case BO_Add: {
      Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
      AST = Z3_mk_fpa_add(Z3Context::ZC, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }
    case BO_Sub: {
      Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
      AST = Z3_mk_fpa_sub(Z3Context::ZC, RoundingMode.AST, LHS.AST, RHS.AST);
      break;
    }

    // Relational operators
    case BO_LT:
      AST = Z3_mk_fpa_lt(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_GT:
      AST = Z3_mk_fpa_gt(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_LE:
      AST = Z3_mk_fpa_leq(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_GE:
      AST = Z3_mk_fpa_geq(Z3Context::ZC, LHS.AST, RHS.AST);
      break;

    // Equality operators
    case BO_EQ:
      AST = Z3_mk_fpa_eq(Z3Context::ZC, LHS.AST, RHS.AST);
      break;
    case BO_NE:
      return Z3Expr::fromFloatUnOp(UO_LNot,
                                   Z3Expr::fromFloatBinOp(LHS, BO_EQ, RHS));
      break;

    // Logical operators
    case BO_LAnd:
    case BO_LOr:
      return Z3Expr::fromBinOp(LHS, Op, RHS, false);
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a SymbolData, given a Z3_context.
  static Z3Expr fromData(const SymbolID ID, bool isBool, bool isFloat,
                         uint64_t BitWidth) {
    llvm::Twine Name = "$" + llvm::Twine(ID);

    Z3Sort Sort;
    if (isBool)
      Sort = Z3Sort::getBoolSort();
    else if (isFloat)
      Sort = Z3Sort::getFloatSort(BitWidth);
    else
      Sort = Z3Sort::getBitvectorSort(BitWidth);

    Z3_symbol Symbol = Z3_mk_string_symbol(Z3Context::ZC, Name.str().c_str());
    Z3_ast AST = Z3_mk_const(Z3Context::ZC, Symbol, Sort.Sort);
    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a SymbolCast, given a Z3_context.
  static Z3Expr fromCast(const Z3Expr &Exp, QualType ToTy, uint64_t ToBitWidth,
                         QualType FromTy, uint64_t FromBitWidth) {
    Z3_ast AST;

    if ((FromTy->isIntegralOrEnumerationType() &&
         ToTy->isIntegralOrEnumerationType()) ||
        (FromTy->isAnyPointerType() ^ ToTy->isAnyPointerType()) ||
        (FromTy->isBlockPointerType() ^ ToTy->isBlockPointerType()) ||
        (FromTy->isReferenceType() ^ ToTy->isReferenceType())) {
      // Special case: Z3 boolean type is distinct from bitvector type, so
      // must use if-then-else expression instead of direct cast
      if (FromTy->isBooleanType()) {
        assert(ToBitWidth > 0 && "BitWidth must be positive!");
        Z3Expr Zero = Z3Expr::fromInt("0", ToBitWidth);
        Z3Expr One = Z3Expr::fromInt("1", ToBitWidth);
        AST = Z3_mk_ite(Z3Context::ZC, Exp.AST, One.AST, Zero.AST);
      } else if (ToBitWidth > FromBitWidth) {
        AST = FromTy->isSignedIntegerOrEnumerationType()
                  ? Z3_mk_sign_ext(Z3Context::ZC, ToBitWidth - FromBitWidth,
                                   Exp.AST)
                  : Z3_mk_zero_ext(Z3Context::ZC, ToBitWidth - FromBitWidth,
                                   Exp.AST);
      } else if (ToBitWidth < FromBitWidth) {
        AST = Z3_mk_extract(Z3Context::ZC, ToBitWidth - 1, 0, Exp.AST);
      } else {
        // Both are bitvectors with the same width, ignore the type cast
        return Exp;
      }
    } else if (FromTy->isRealFloatingType() && ToTy->isRealFloatingType()) {
      if (ToBitWidth != FromBitWidth) {
        Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
        Z3Sort Sort = Z3Sort::getFloatSort(ToBitWidth);
        AST = Z3_mk_fpa_to_fp_float(Z3Context::ZC, RoundingMode.AST, Exp.AST,
                                    Sort.Sort);
      } else {
        return Exp;
      }
    } else if (FromTy->isIntegralOrEnumerationType() &&
               ToTy->isRealFloatingType()) {
      Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
      Z3Sort Sort = Z3Sort::getFloatSort(ToBitWidth);
      AST = FromTy->isSignedIntegerOrEnumerationType()
                ? Z3_mk_fpa_to_fp_signed(Z3Context::ZC, RoundingMode.AST,
                                         Exp.AST, Sort.Sort)
                : Z3_mk_fpa_to_fp_unsigned(Z3Context::ZC, RoundingMode.AST,
                                           Exp.AST, Sort.Sort);
    } else if (FromTy->isRealFloatingType() &&
               ToTy->isIntegralOrEnumerationType()) {
      Z3Expr RoundingMode = Z3Expr::getFloatRoundingMode();
      AST = ToTy->isSignedIntegerOrEnumerationType()
                ? Z3_mk_fpa_to_sbv(Z3Context::ZC, RoundingMode.AST, Exp.AST,
                                   ToBitWidth)
                : Z3_mk_fpa_to_ubv(Z3Context::ZC, RoundingMode.AST, Exp.AST,
                                   ToBitWidth);
    } else {
      llvm_unreachable("Unsupported explicit type cast!");
    }

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a boolean, given a Z3_context.
  static Z3Expr fromBoolean(const bool Bool) {
    Z3_ast AST = Bool ? Z3_mk_true(Z3Context::ZC) : Z3_mk_false(Z3Context::ZC);
    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from a finite APFloat, given a Z3_context.
  static Z3Expr fromAPFloat(const llvm::APFloat &Float) {
    Z3_ast AST;
    Z3Sort Sort = Z3Sort::getFloatSort(
        llvm::APFloat::semanticsSizeInBits(Float.getSemantics()));

    llvm::APSInt Int = llvm::APSInt(Float.bitcastToAPInt(), true);
    Z3Expr Z3Int = Z3Expr::fromAPSInt(Int);
    AST = Z3_mk_fpa_to_fp_bv(Z3Context::ZC, Z3Int.AST, Sort.Sort);

    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from an APSInt, given a Z3_context.
  static Z3Expr fromAPSInt(const llvm::APSInt &Int) {
    Z3Sort Sort = Z3Sort::getBitvectorSort(Int.getBitWidth());
    Z3_ast AST =
        Z3_mk_numeral(Z3Context::ZC, Int.toString(10).c_str(), Sort.Sort);
    return Z3Expr(AST);
  }

  /// Construct a Z3Expr from an integer, given a Z3_context.
  static Z3Expr fromInt(const char *Int, uint64_t BitWidth) {
    Z3Sort Sort = Z3Sort::getBitvectorSort(BitWidth);
    Z3_ast AST = Z3_mk_numeral(Z3Context::ZC, Int, Sort.Sort);
    return Z3Expr(AST);
  }

  /// Construct an APFloat from a Z3Expr, given the AST representation
  static bool toAPFloat(const Z3Sort &Sort, const Z3_ast &AST,
                        llvm::APFloat &Float, bool useSemantics = true) {
    assert(Sort.getSortKind() == Z3_FLOATING_POINT_SORT &&
           "Unsupported sort to floating-point!");

    llvm::APSInt Int(Sort.getFloatSortSize(), true);
    const llvm::fltSemantics &Semantics =
        Z3Expr::getFloatSemantics(Sort.getFloatSortSize());
    Z3Sort BVSort = Z3Sort::getBitvectorSort(Sort.getFloatSortSize());
    if (!Z3Expr::toAPSInt(BVSort, AST, Int, true)) {
      return false;
    }

    if (useSemantics &&
        !Z3Expr::areEquivalent(Float.getSemantics(), Semantics)) {
      assert(false && "Floating-point types don't match!");
      return false;
    }

    Float = llvm::APFloat(Semantics, Int);
    return true;
  }

  /// Construct an APSInt from a Z3Expr, given the AST representation
  static bool toAPSInt(const Z3Sort &Sort, const Z3_ast &AST, llvm::APSInt &Int,
                       bool useSemantics = true) {
    switch (Sort.getSortKind()) {
    default:
      llvm_unreachable("Unsupported sort to integer!");
    case Z3_BV_SORT: {
      if (useSemantics && Int.getBitWidth() != Sort.getBitvectorSortSize()) {
        assert(false && "Bitvector types don't match!");
        return false;
      }

      uint64_t Value[2];
      // Force cast because Z3 defines __uint64 to be a unsigned long long
      // type, which isn't compatible with a unsigned long type, even if they
      // are the same size.
      Z3_get_numeral_uint64(Z3Context::ZC, AST,
                            reinterpret_cast<__uint64 *>(&Value[0]));
      if (Sort.getBitvectorSortSize() <= 64) {
        Int = llvm::APSInt(llvm::APInt(Int.getBitWidth(), Value[0]), true);
      } else if (Sort.getBitvectorSortSize() == 128) {
        Z3Expr ASTHigh = Z3Expr(Z3_mk_extract(Z3Context::ZC, 127, 64, AST));
        Z3_get_numeral_uint64(Z3Context::ZC, AST,
                              reinterpret_cast<__uint64 *>(&Value[1]));
        Int = llvm::APSInt(llvm::APInt(Int.getBitWidth(), Value), true);
      } else {
        assert(false && "Bitwidth not supported!");
        return false;
      }
      return true;
    }
    case Z3_BOOL_SORT:
      if (useSemantics && Int.getBitWidth() < 1) {
        assert(false && "Boolean type doesn't match!");
        return false;
      }
      Int = llvm::APSInt(
          llvm::APInt(Int.getBitWidth(),
                      Z3_get_bool_value(Z3Context::ZC, AST) == Z3_L_TRUE ? 1
                                                                         : 0),
          true);
      return true;
    }
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Z3_get_ast_hash(Z3Context::ZC, AST));
  }

  bool operator<(const Z3Expr &Other) const {
    llvm::FoldingSetNodeID ID1, ID2;
    Profile(ID1);
    Other.Profile(ID2);
    return ID1 < ID2;
  }

  /// Comparison of AST equality, not model equivalence.
  bool operator==(const Z3Expr &Other) const {
    assert(Z3_is_eq_sort(Z3Context::ZC, Z3_get_sort(Z3Context::ZC, AST),
                         Z3_get_sort(Z3Context::ZC, Other.AST)) &&
           "AST's must have the same sort");
    return Z3_is_eq_ast(Z3Context::ZC, AST, Other.AST);
  }

  /// Override implicit move constructor for correct reference counting.
  Z3Expr &operator=(const Z3Expr &Move) {
    Z3_inc_ref(Z3Context::ZC, Move.AST);
    Z3_dec_ref(Z3Context::ZC, AST);
    AST = Move.AST;
    return *this;
  }

  void print(raw_ostream &OS) const {
    OS << Z3_ast_to_string(Z3Context::ZC, AST);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Expr

class Z3Model {
  Z3_model Model;

public:
  Z3Model(Z3_model ZM) : Model(ZM) { Z3_model_inc_ref(Z3Context::ZC, Model); }

  /// Override implicit copy constructor for correct reference counting.
  Z3Model(const Z3Model &Copy) : Model(Copy.Model) {
    Z3_model_inc_ref(Z3Context::ZC, Model);
  }

  /// Provide move constructor
  Z3Model(Z3Model &&Move) : Model(nullptr) { *this = std::move(Move); }

  /// Provide move assignment constructor
  Z3Model &operator=(Z3Model &&Move) {
    if (this != &Move) {
      if (Model)
        Z3_model_dec_ref(Z3Context::ZC, Model);
      Model = Move.Model;
      Move.Model = nullptr;
    }
    return *this;
  }

  ~Z3Model() {
    if (Model)
      Z3_model_dec_ref(Z3Context::ZC, Model);
  }

  /// Given an expression, extract the value of this operand in the model.
  bool getInterpretation(const Z3Expr &Exp, llvm::APSInt &Int) const {
    Z3_func_decl Func =
        Z3_get_app_decl(Z3Context::ZC, Z3_to_app(Z3Context::ZC, Exp.AST));
    if (Z3_model_has_interp(Z3Context::ZC, Model, Func) != Z3_L_TRUE)
      return false;

    Z3_ast Assign = Z3_model_get_const_interp(Z3Context::ZC, Model, Func);
    Z3Sort Sort = Z3Sort::getSort(Assign);
    return Z3Expr::toAPSInt(Sort, Assign, Int, true);
  }

  /// Given an expression, extract the value of this operand in the model.
  bool getInterpretation(const Z3Expr &Exp, llvm::APFloat &Float) const {
    Z3_func_decl Func =
        Z3_get_app_decl(Z3Context::ZC, Z3_to_app(Z3Context::ZC, Exp.AST));
    if (Z3_model_has_interp(Z3Context::ZC, Model, Func) != Z3_L_TRUE)
      return false;

    Z3_ast Assign = Z3_model_get_const_interp(Z3Context::ZC, Model, Func);
    Z3Sort Sort = Z3Sort::getSort(Assign);
    return Z3Expr::toAPFloat(Sort, Assign, Float, true);
  }

  void print(raw_ostream &OS) const {
    OS << Z3_model_to_string(Z3Context::ZC, Model);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Model

class Z3Solver {
  friend class Z3ConstraintManager;

  Z3_solver Solver;

  Z3Solver(Z3_solver ZS) : Solver(ZS) {
    Z3_solver_inc_ref(Z3Context::ZC, Solver);
  }

public:
  /// Override implicit copy constructor for correct reference counting.
  Z3Solver(const Z3Solver &Copy) : Solver(Copy.Solver) {
    Z3_solver_inc_ref(Z3Context::ZC, Solver);
  }

  /// Provide move constructor
  Z3Solver(Z3Solver &&Move) : Solver(nullptr) { *this = std::move(Move); }

  /// Provide move assignment constructor
  Z3Solver &operator=(Z3Solver &&Move) {
    if (this != &Move) {
      if (Solver)
        Z3_solver_dec_ref(Z3Context::ZC, Solver);
      Solver = Move.Solver;
      Move.Solver = nullptr;
    }
    return *this;
  }

  ~Z3Solver() {
    if (Solver)
      Z3_solver_dec_ref(Z3Context::ZC, Solver);
  }

  /// Given a constraint, add it to the solver
  void addConstraint(const Z3Expr &Exp) {
    Z3_solver_assert(Z3Context::ZC, Solver, Exp.AST);
  }

  /// Given a program state, construct the logical conjunction and add it to
  /// the solver
  void addStateConstraints(ProgramStateRef State) {
    // TODO: Don't add all the constraints, only the relevant ones
    ConstraintZ3Ty CZ = State->get<ConstraintZ3>();
    ConstraintZ3Ty::iterator I = CZ.begin(), IE = CZ.end();

    // Construct the logical AND of all the constraints
    if (I != IE) {
      std::vector<Z3_ast> ASTs;

      while (I != IE)
        ASTs.push_back(I++->second.AST);

      Z3Expr Conj = Z3Expr::fromNBinOp(BO_LAnd, ASTs);
      addConstraint(Conj);
    }
  }

  /// Check if the constraints are satisfiable
  Z3_lbool check() { return Z3_solver_check(Z3Context::ZC, Solver); }

  /// Push the current solver state
  void push() { return Z3_solver_push(Z3Context::ZC, Solver); }

  /// Pop the previous solver state
  void pop(unsigned NumStates = 1) {
    assert(Z3_solver_get_num_scopes(Z3Context::ZC, Solver) >= NumStates);
    return Z3_solver_pop(Z3Context::ZC, Solver, NumStates);
  }

  /// Get a model from the solver. Caller should check the model is
  /// satisfiable.
  Z3Model getModel() {
    return Z3Model(Z3_solver_get_model(Z3Context::ZC, Solver));
  }

  /// Reset the solver and remove all constraints.
  void reset() { Z3_solver_reset(Z3Context::ZC, Solver); }

  void print(raw_ostream &OS) const {
    OS << Z3_solver_to_string(Z3Context::ZC, Solver);
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }
}; // end class Z3Solver

void Z3ErrorHandler(Z3_context Context, Z3_error_code Error) {
  llvm::report_fatal_error("Z3 error: " +
                           llvm::Twine(Z3_get_error_msg_ex(Context, Error)));
}

class Z3ConstraintManager : public SMTConstraintManager {
  Z3Context Context;
  mutable Z3Solver Solver;

public:
  Z3ConstraintManager(SubEngine *SE, SValBuilder &SB)
      : SMTConstraintManager(SE, SB),
        Solver(Z3_mk_simple_solver(Z3Context::ZC)) {
    Z3_set_error_handler(Z3Context::ZC, Z3ErrorHandler);
  }
  //===------------------------------------------------------------------===//
  // Implementation for Refutation.
  //===------------------------------------------------------------------===//

  void addRangeConstraints(clang::ento::ConstraintRangeTy CR) override;

  ConditionTruthVal isModelFeasible() override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from ConstraintManager.
  //===------------------------------------------------------------------===//

  bool canReasonAbout(SVal X) const override;

  ConditionTruthVal checkNull(ProgramStateRef State, SymbolRef Sym) override;

  const llvm::APSInt *getSymVal(ProgramStateRef State,
                                SymbolRef Sym) const override;

  ProgramStateRef removeDeadBindings(ProgramStateRef St,
                                     SymbolReaper &SymReaper) override;

  void print(ProgramStateRef St, raw_ostream &Out, const char *nl,
             const char *sep) override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from SimpleConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSym(ProgramStateRef state, SymbolRef Sym,
                            bool Assumption) override;

  ProgramStateRef assumeSymInclusiveRange(ProgramStateRef State, SymbolRef Sym,
                                          const llvm::APSInt &From,
                                          const llvm::APSInt &To,
                                          bool InRange) override;

  ProgramStateRef assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                       bool Assumption) override;

private:
  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//

  // Check whether a new model is satisfiable, and update the program state.
  ProgramStateRef assumeZ3Expr(ProgramStateRef State, SymbolRef Sym,
                               const Z3Expr &Exp);

  // Generate and check a Z3 model, using the given constraint.
  Z3_lbool checkZ3Model(ProgramStateRef State, const Z3Expr &Exp) const;

  // Generate a Z3Expr that represents the given symbolic expression.
  // Sets the hasComparison parameter if the expression has a comparison
  // operator.
  // Sets the RetTy parameter to the final return type after promotions and
  // casts.
  Z3Expr getZ3Expr(SymbolRef Sym, QualType *RetTy = nullptr,
                   bool *hasComparison = nullptr) const;

  // Generate a Z3Expr that takes the logical not of an expression.
  Z3Expr getZ3NotExpr(const Z3Expr &Exp) const;

  // Generate a Z3Expr that compares the expression to zero.
  Z3Expr getZ3ZeroExpr(const Z3Expr &Exp, QualType RetTy,
                       bool Assumption) const;

  // Recursive implementation to unpack and generate symbolic expression.
  // Sets the hasComparison and RetTy parameters. See getZ3Expr().
  Z3Expr getZ3SymExpr(SymbolRef Sym, QualType *RetTy,
                      bool *hasComparison) const;

  // Wrapper to generate Z3Expr from SymbolData.
  Z3Expr getZ3DataExpr(const SymbolID ID, QualType Ty) const;

  // Wrapper to generate Z3Expr from SymbolCast.
  Z3Expr getZ3CastExpr(const Z3Expr &Exp, QualType FromTy, QualType Ty) const;

  // Wrapper to generate Z3Expr from BinarySymExpr.
  // Sets the hasComparison and RetTy parameters. See getZ3Expr().
  Z3Expr getZ3SymBinExpr(const BinarySymExpr *BSE, bool *hasComparison,
                         QualType *RetTy) const;

  // Wrapper to generate Z3Expr from unpacked binary symbolic expression.
  // Sets the RetTy parameter. See getZ3Expr().
  Z3Expr getZ3BinExpr(const Z3Expr &LHS, QualType LTy,
                      BinaryOperator::Opcode Op, const Z3Expr &RHS,
                      QualType RTy, QualType *RetTy) const;

  //===------------------------------------------------------------------===//
  // Helper functions.
  //===------------------------------------------------------------------===//

  // Recover the QualType of an APSInt.
  // TODO: Refactor to put elsewhere
  QualType getAPSIntType(const llvm::APSInt &Int) const;

  // Get the QualTy for the input APSInt, and fix it if it has a bitwidth of 1.
  std::pair<llvm::APSInt, QualType> fixAPSInt(const llvm::APSInt &Int) const;

  // Perform implicit type conversion on binary symbolic expressions.
  // May modify all input parameters.
  // TODO: Refactor to use built-in conversion functions
  void doTypeConversion(Z3Expr &LHS, Z3Expr &RHS, QualType &LTy,
                        QualType &RTy) const;

  // Perform implicit integer type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleIntegerConversion()
  template <typename T,
            T(doCast)(const T &, QualType, uint64_t, QualType, uint64_t)>
  void doIntTypeConversion(T &LHS, QualType &LTy, T &RHS, QualType &RTy) const;

  // Perform implicit floating-point type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleFloatConversion()
  template <typename T,
            T(doCast)(const T &, QualType, uint64_t, QualType, uint64_t)>
  void doFloatTypeConversion(T &LHS, QualType &LTy, T &RHS,
                             QualType &RTy) const;

  // Callback function for doCast parameter on APSInt type.
  static llvm::APSInt castAPSInt(const llvm::APSInt &V, QualType ToTy,
                                 uint64_t ToWidth, QualType FromTy,
                                 uint64_t FromWidth);
}; // end class Z3ConstraintManager

Z3_context Z3Context::ZC;

} // end anonymous namespace

ProgramStateRef Z3ConstraintManager::assumeSym(ProgramStateRef State,
                                               SymbolRef Sym, bool Assumption) {
  QualType RetTy;
  bool hasComparison;

  Z3Expr Exp = getZ3Expr(Sym, &RetTy, &hasComparison);
  // Create zero comparison for implicit boolean cast, with reversed assumption
  if (!hasComparison && !RetTy->isBooleanType())
    return assumeZ3Expr(State, Sym, getZ3ZeroExpr(Exp, RetTy, !Assumption));

  return assumeZ3Expr(State, Sym, Assumption ? Exp : getZ3NotExpr(Exp));
}

ProgramStateRef Z3ConstraintManager::assumeSymInclusiveRange(
    ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
    const llvm::APSInt &To, bool InRange) {
  QualType RetTy;
  // The expression may be casted, so we cannot call getZ3DataExpr() directly
  Z3Expr Exp = getZ3Expr(Sym, &RetTy);

  QualType FromTy;
  llvm::APSInt NewFromInt;
  std::tie(NewFromInt, FromTy) = fixAPSInt(From);
  Z3Expr FromExp = Z3Expr::fromAPSInt(NewFromInt);

  // Construct single (in)equality
  if (From == To)
    return assumeZ3Expr(State, Sym,
                        getZ3BinExpr(Exp, RetTy, InRange ? BO_EQ : BO_NE,
                                     FromExp, FromTy, nullptr));

  QualType ToTy;
  llvm::APSInt NewToInt;
  std::tie(NewToInt, ToTy) = fixAPSInt(To);
  Z3Expr ToExp = Z3Expr::fromAPSInt(NewToInt);
  assert(FromTy == ToTy && "Range values have different types!");
  // Construct two (in)equalities, and a logical and/or
  Z3Expr LHS = getZ3BinExpr(Exp, RetTy, InRange ? BO_GE : BO_LT, FromExp,
                            FromTy, nullptr);
  Z3Expr RHS =
      getZ3BinExpr(Exp, RetTy, InRange ? BO_LE : BO_GT, ToExp, ToTy, nullptr);
  return assumeZ3Expr(
      State, Sym,
      Z3Expr::fromBinOp(LHS, InRange ? BO_LAnd : BO_LOr, RHS,
                        RetTy->isSignedIntegerOrEnumerationType()));
}

ProgramStateRef Z3ConstraintManager::assumeSymUnsupported(ProgramStateRef State,
                                                          SymbolRef Sym,
                                                          bool Assumption) {
  // Skip anything that is unsupported
  return State;
}

bool Z3ConstraintManager::canReasonAbout(SVal X) const {
  const TargetInfo &TI = getBasicVals().getContext().getTargetInfo();

  Optional<nonloc::SymbolVal> SymVal = X.getAs<nonloc::SymbolVal>();
  if (!SymVal)
    return true;

  const SymExpr *Sym = SymVal->getSymbol();
  do {
    QualType Ty = Sym->getType();

    // Complex types are not modeled
    if (Ty->isComplexType() || Ty->isComplexIntegerType())
      return false;

    // Non-IEEE 754 floating-point types are not modeled
    if ((Ty->isSpecificBuiltinType(BuiltinType::LongDouble) &&
         (&TI.getLongDoubleFormat() == &llvm::APFloat::x87DoubleExtended() ||
          &TI.getLongDoubleFormat() == &llvm::APFloat::PPCDoubleDouble())))
      return false;

    if (isa<SymbolData>(Sym)) {
      break;
    } else if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
      Sym = SC->getOperand();
    } else if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
      if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
        Sym = SIE->getLHS();
      } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
        Sym = ISE->getRHS();
      } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
        return canReasonAbout(nonloc::SymbolVal(SSM->getLHS())) &&
               canReasonAbout(nonloc::SymbolVal(SSM->getRHS()));
      } else {
        llvm_unreachable("Unsupported binary expression to reason about!");
      }
    } else {
      llvm_unreachable("Unsupported expression to reason about!");
    }
  } while (Sym);

  return true;
}

ConditionTruthVal Z3ConstraintManager::checkNull(ProgramStateRef State,
                                                 SymbolRef Sym) {
  QualType RetTy;
  // The expression may be casted, so we cannot call getZ3DataExpr() directly
  Z3Expr VarExp = getZ3Expr(Sym, &RetTy);
  Z3Expr Exp = getZ3ZeroExpr(VarExp, RetTy, true);
  // Negate the constraint
  Z3Expr NotExp = getZ3ZeroExpr(VarExp, RetTy, false);

  Solver.reset();
  Solver.addStateConstraints(State);

  Solver.push();
  Solver.addConstraint(Exp);
  Z3_lbool isSat = Solver.check();

  Solver.pop();
  Solver.addConstraint(NotExp);
  Z3_lbool isNotSat = Solver.check();

  // Zero is the only possible solution
  if (isSat == Z3_L_TRUE && isNotSat == Z3_L_FALSE)
    return true;
  // Zero is not a solution
  else if (isSat == Z3_L_FALSE && isNotSat == Z3_L_TRUE)
    return false;

  // Zero may be a solution
  return ConditionTruthVal();
}

const llvm::APSInt *Z3ConstraintManager::getSymVal(ProgramStateRef State,
                                                   SymbolRef Sym) const {
  BasicValueFactory &BVF = getBasicVals();
  ASTContext &Ctx = BVF.getContext();

  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    QualType Ty = Sym->getType();
    assert(!Ty->isRealFloatingType());
    llvm::APSInt Value(Ctx.getTypeSize(Ty),
                       !Ty->isSignedIntegerOrEnumerationType());

    Z3Expr Exp = getZ3DataExpr(SD->getSymbolID(), Ty);

    Solver.reset();
    Solver.addStateConstraints(State);

    // Constraints are unsatisfiable
    if (Solver.check() != Z3_L_TRUE)
      return nullptr;

    Z3Model Model = Solver.getModel();
    // Model does not assign interpretation
    if (!Model.getInterpretation(Exp, Value))
      return nullptr;

    // A value has been obtained, check if it is the only value
    Z3Expr NotExp = Z3Expr::fromBinOp(
        Exp, BO_NE,
        Ty->isBooleanType() ? Z3Expr::fromBoolean(Value.getBoolValue())
                            : Z3Expr::fromAPSInt(Value),
        false);

    Solver.addConstraint(NotExp);
    if (Solver.check() == Z3_L_TRUE)
      return nullptr;

    // This is the only solution, store it
    return &BVF.getValue(Value);
  } else if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    SymbolRef CastSym = SC->getOperand();
    QualType CastTy = SC->getType();
    // Skip the void type
    if (CastTy->isVoidType())
      return nullptr;

    const llvm::APSInt *Value;
    if (!(Value = getSymVal(State, CastSym)))
      return nullptr;
    return &BVF.Convert(SC->getType(), *Value);
  } else if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    const llvm::APSInt *LHS, *RHS;
    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
      LHS = getSymVal(State, SIE->getLHS());
      RHS = &SIE->getRHS();
    } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
      LHS = &ISE->getLHS();
      RHS = getSymVal(State, ISE->getRHS());
    } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
      // Early termination to avoid expensive call
      LHS = getSymVal(State, SSM->getLHS());
      RHS = LHS ? getSymVal(State, SSM->getRHS()) : nullptr;
    } else {
      llvm_unreachable("Unsupported binary expression to get symbol value!");
    }

    if (!LHS || !RHS)
      return nullptr;

    llvm::APSInt ConvertedLHS, ConvertedRHS;
    QualType LTy, RTy;
    std::tie(ConvertedLHS, LTy) = fixAPSInt(*LHS);
    std::tie(ConvertedRHS, RTy) = fixAPSInt(*RHS);
    doIntTypeConversion<llvm::APSInt, Z3ConstraintManager::castAPSInt>(
        ConvertedLHS, LTy, ConvertedRHS, RTy);
    return BVF.evalAPSInt(BSE->getOpcode(), ConvertedLHS, ConvertedRHS);
  }

  llvm_unreachable("Unsupported expression to get symbol value!");
}

ProgramStateRef
Z3ConstraintManager::removeDeadBindings(ProgramStateRef State,
                                        SymbolReaper &SymReaper) {
  ConstraintZ3Ty CZ = State->get<ConstraintZ3>();
  ConstraintZ3Ty::Factory &CZFactory = State->get_context<ConstraintZ3>();

  for (ConstraintZ3Ty::iterator I = CZ.begin(), E = CZ.end(); I != E; ++I) {
    if (SymReaper.maybeDead(I->first))
      CZ = CZFactory.remove(CZ, *I);
  }

  return State->set<ConstraintZ3>(CZ);
}

void Z3ConstraintManager::addRangeConstraints(ConstraintRangeTy CR) {
  for (const auto &I : CR) {
    SymbolRef Sym = I.first;

    Z3Expr Constraints = Z3Expr::fromBoolean(false);

    for (const auto &Range : I.second) {
      const llvm::APSInt &From = Range.From();
      const llvm::APSInt &To = Range.To();

      QualType FromTy;
      llvm::APSInt NewFromInt;
      std::tie(NewFromInt, FromTy) = fixAPSInt(From);
      Z3Expr FromExp = Z3Expr::fromAPSInt(NewFromInt);
      QualType SymTy;
      Z3Expr Exp = getZ3Expr(Sym, &SymTy);
      bool IsSignedTy = SymTy->isSignedIntegerOrEnumerationType();
      QualType ToTy;
      llvm::APSInt NewToInt;
      std::tie(NewToInt, ToTy) = fixAPSInt(To);
      Z3Expr ToExp = Z3Expr::fromAPSInt(NewToInt);
      assert(FromTy == ToTy && "Range values have different types!");

      Z3Expr LHS =
          getZ3BinExpr(Exp, SymTy, BO_GE, FromExp, FromTy, /*RetTy=*/nullptr);
      Z3Expr RHS =
          getZ3BinExpr(Exp, SymTy, BO_LE, ToExp, FromTy, /*RetTy=*/nullptr);
      Z3Expr SymRange = Z3Expr::fromBinOp(LHS, BO_LAnd, RHS, IsSignedTy);
      Constraints =
          Z3Expr::fromBinOp(Constraints, BO_LOr, SymRange, IsSignedTy);
    }
    Solver.addConstraint(Constraints);
  }
}

clang::ento::ConditionTruthVal Z3ConstraintManager::isModelFeasible() {
  if (Solver.check() == Z3_L_FALSE)
    return false;

  return ConditionTruthVal();
}

//===------------------------------------------------------------------===//
// Internal implementation.
//===------------------------------------------------------------------===//

ProgramStateRef Z3ConstraintManager::assumeZ3Expr(ProgramStateRef State,
                                                  SymbolRef Sym,
                                                  const Z3Expr &Exp) {
  // Check the model, avoid simplifying AST to save time
  if (checkZ3Model(State, Exp) == Z3_L_TRUE)
    return State->add<ConstraintZ3>(std::make_pair(Sym, Exp));

  return nullptr;
}

Z3_lbool Z3ConstraintManager::checkZ3Model(ProgramStateRef State,
                                           const Z3Expr &Exp) const {
  Solver.reset();
  Solver.addConstraint(Exp);
  Solver.addStateConstraints(State);
  return Solver.check();
}

Z3Expr Z3ConstraintManager::getZ3Expr(SymbolRef Sym, QualType *RetTy,
                                      bool *hasComparison) const {
  if (hasComparison) {
    *hasComparison = false;
  }

  return getZ3SymExpr(Sym, RetTy, hasComparison);
}

Z3Expr Z3ConstraintManager::getZ3NotExpr(const Z3Expr &Exp) const {
  return Z3Expr::fromUnOp(UO_LNot, Exp);
}

Z3Expr Z3ConstraintManager::getZ3ZeroExpr(const Z3Expr &Exp, QualType Ty,
                                          bool Assumption) const {
  ASTContext &Ctx = getBasicVals().getContext();
  if (Ty->isRealFloatingType()) {
    llvm::APFloat Zero = llvm::APFloat::getZero(Ctx.getFloatTypeSemantics(Ty));
    return Z3Expr::fromFloatBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                                  Z3Expr::fromAPFloat(Zero));
  } else if (Ty->isIntegralOrEnumerationType() || Ty->isAnyPointerType() ||
             Ty->isBlockPointerType() || Ty->isReferenceType()) {
    bool isSigned = Ty->isSignedIntegerOrEnumerationType();
    // Skip explicit comparison for boolean types
    if (Ty->isBooleanType())
      return Assumption ? getZ3NotExpr(Exp) : Exp;
    return Z3Expr::fromBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                             Z3Expr::fromInt("0", Ctx.getTypeSize(Ty)),
                             isSigned);
  }

  llvm_unreachable("Unsupported type for zero value!");
}

Z3Expr Z3ConstraintManager::getZ3SymExpr(SymbolRef Sym, QualType *RetTy,
                                         bool *hasComparison) const {
  if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
    if (RetTy)
      *RetTy = Sym->getType();

    return getZ3DataExpr(SD->getSymbolID(), Sym->getType());
  } else if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
    if (RetTy)
      *RetTy = Sym->getType();

    QualType FromTy;
    Z3Expr Exp = getZ3SymExpr(SC->getOperand(), &FromTy, hasComparison);
    // Casting an expression with a comparison invalidates it. Note that this
    // must occur after the recursive call above.
    // e.g. (signed char) (x > 0)
    if (hasComparison)
      *hasComparison = false;
    return getZ3CastExpr(Exp, FromTy, Sym->getType());
  } else if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
    Z3Expr Exp = getZ3SymBinExpr(BSE, hasComparison, RetTy);
    // Set the hasComparison parameter, in post-order traversal order.
    if (hasComparison)
      *hasComparison = BinaryOperator::isComparisonOp(BSE->getOpcode());
    return Exp;
  }

  llvm_unreachable("Unsupported SymbolRef type!");
}

Z3Expr Z3ConstraintManager::getZ3DataExpr(const SymbolID ID,
                                          QualType Ty) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Z3Expr::fromData(ID, Ty->isBooleanType(), Ty->isRealFloatingType(),
                          Ctx.getTypeSize(Ty));
}

Z3Expr Z3ConstraintManager::getZ3CastExpr(const Z3Expr &Exp, QualType FromTy,
                                          QualType ToTy) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Z3Expr::fromCast(Exp, ToTy, Ctx.getTypeSize(ToTy), FromTy,
                          Ctx.getTypeSize(FromTy));
}

Z3Expr Z3ConstraintManager::getZ3SymBinExpr(const BinarySymExpr *BSE,
                                            bool *hasComparison,
                                            QualType *RetTy) const {
  QualType LTy, RTy;
  BinaryOperator::Opcode Op = BSE->getOpcode();

  if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
    Z3Expr LHS = getZ3SymExpr(SIE->getLHS(), &LTy, hasComparison);
    llvm::APSInt NewRInt;
    std::tie(NewRInt, RTy) = fixAPSInt(SIE->getRHS());
    Z3Expr RHS = Z3Expr::fromAPSInt(NewRInt);
    return getZ3BinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  } else if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
    llvm::APSInt NewLInt;
    std::tie(NewLInt, LTy) = fixAPSInt(ISE->getLHS());
    Z3Expr LHS = Z3Expr::fromAPSInt(NewLInt);
    Z3Expr RHS = getZ3SymExpr(ISE->getRHS(), &RTy, hasComparison);
    return getZ3BinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  } else if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
    Z3Expr LHS = getZ3SymExpr(SSM->getLHS(), &LTy, hasComparison);
    Z3Expr RHS = getZ3SymExpr(SSM->getRHS(), &RTy, hasComparison);
    return getZ3BinExpr(LHS, LTy, Op, RHS, RTy, RetTy);
  } else {
    llvm_unreachable("Unsupported BinarySymExpr type!");
  }
}

Z3Expr Z3ConstraintManager::getZ3BinExpr(const Z3Expr &LHS, QualType LTy,
                                         BinaryOperator::Opcode Op,
                                         const Z3Expr &RHS, QualType RTy,
                                         QualType *RetTy) const {
  Z3Expr NewLHS = LHS;
  Z3Expr NewRHS = RHS;
  doTypeConversion(NewLHS, NewRHS, LTy, RTy);
  // Update the return type parameter if the output type has changed.
  if (RetTy) {
    // A boolean result can be represented as an integer type in C/C++, but at
    // this point we only care about the Z3 type. Set it as a boolean type to
    // avoid subsequent Z3 errors.
    if (BinaryOperator::isComparisonOp(Op) || BinaryOperator::isLogicalOp(Op)) {
      ASTContext &Ctx = getBasicVals().getContext();
      *RetTy = Ctx.BoolTy;
    } else {
      *RetTy = LTy;
    }

    // If the two operands are pointers and the operation is a subtraction, the
    // result is of type ptrdiff_t, which is signed
    if (LTy->isAnyPointerType() && LTy == RTy && Op == BO_Sub) {
      ASTContext &Ctx = getBasicVals().getContext();
      *RetTy = Ctx.getIntTypeForBitwidth(Ctx.getTypeSize(LTy), true);
    }
  }

  return LTy->isRealFloatingType()
             ? Z3Expr::fromFloatBinOp(NewLHS, Op, NewRHS)
             : Z3Expr::fromBinOp(NewLHS, Op, NewRHS,
                                 LTy->isSignedIntegerOrEnumerationType());
}

//===------------------------------------------------------------------===//
// Helper functions.
//===------------------------------------------------------------------===//

QualType Z3ConstraintManager::getAPSIntType(const llvm::APSInt &Int) const {
  ASTContext &Ctx = getBasicVals().getContext();
  return Ctx.getIntTypeForBitwidth(Int.getBitWidth(), Int.isSigned());
}

std::pair<llvm::APSInt, QualType>
Z3ConstraintManager::fixAPSInt(const llvm::APSInt &Int) const {
  llvm::APSInt NewInt;

  // FIXME: This should be a cast from a 1-bit integer type to a boolean type,
  // but the former is not available in Clang. Instead, extend the APSInt
  // directly.
  if (Int.getBitWidth() == 1 && getAPSIntType(Int).isNull()) {
    ASTContext &Ctx = getBasicVals().getContext();
    NewInt = Int.extend(Ctx.getTypeSize(Ctx.BoolTy));
  } else
    NewInt = Int;

  return std::make_pair(NewInt, getAPSIntType(NewInt));
}

void Z3ConstraintManager::doTypeConversion(Z3Expr &LHS, Z3Expr &RHS,
                                           QualType &LTy, QualType &RTy) const {
  ASTContext &Ctx = getBasicVals().getContext();

  assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");
  // Perform type conversion
  if (LTy->isIntegralOrEnumerationType() &&
      RTy->isIntegralOrEnumerationType()) {
    if (LTy->isArithmeticType() && RTy->isArithmeticType())
      return doIntTypeConversion<Z3Expr, Z3Expr::fromCast>(LHS, LTy, RHS, RTy);
  } else if (LTy->isRealFloatingType() || RTy->isRealFloatingType()) {
    return doFloatTypeConversion<Z3Expr, Z3Expr::fromCast>(LHS, LTy, RHS, RTy);
  } else if ((LTy->isAnyPointerType() || RTy->isAnyPointerType()) ||
             (LTy->isBlockPointerType() || RTy->isBlockPointerType()) ||
             (LTy->isReferenceType() || RTy->isReferenceType())) {
    // TODO: Refactor to Sema::FindCompositePointerType(), and
    // Sema::CheckCompareOperands().

    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    // Cast the non-pointer type to the pointer type.
    // TODO: Be more strict about this.
    if ((LTy->isAnyPointerType() ^ RTy->isAnyPointerType()) ||
        (LTy->isBlockPointerType() ^ RTy->isBlockPointerType()) ||
        (LTy->isReferenceType() ^ RTy->isReferenceType())) {
      if (LTy->isNullPtrType() || LTy->isBlockPointerType() ||
          LTy->isReferenceType()) {
        LHS = Z3Expr::fromCast(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      } else {
        RHS = Z3Expr::fromCast(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      }
    }

    // Cast the void pointer type to the non-void pointer type.
    // For void types, this assumes that the casted value is equal to the value
    // of the original pointer, and does not account for alignment requirements.
    if (LTy->isVoidPointerType() ^ RTy->isVoidPointerType()) {
      assert((Ctx.getTypeSize(LTy) == Ctx.getTypeSize(RTy)) &&
             "Pointer types have different bitwidths!");
      if (RTy->isVoidPointerType())
        RTy = LTy;
      else
        LTy = RTy;
    }

    if (LTy == RTy)
      return;
  }

  // Fallback: for the solver, assume that these types don't really matter
  if ((LTy.getCanonicalType() == RTy.getCanonicalType()) ||
      (LTy->isObjCObjectPointerType() && RTy->isObjCObjectPointerType())) {
    LTy = RTy;
    return;
  }

  // TODO: Refine behavior for invalid type casts
}

template <typename T,
          T(doCast)(const T &, QualType, uint64_t, QualType, uint64_t)>
void Z3ConstraintManager::doIntTypeConversion(T &LHS, QualType &LTy, T &RHS,
                                              QualType &RTy) const {
  ASTContext &Ctx = getBasicVals().getContext();
  uint64_t LBitWidth = Ctx.getTypeSize(LTy);
  uint64_t RBitWidth = Ctx.getTypeSize(RTy);

  assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");
  // Always perform integer promotion before checking type equality.
  // Otherwise, e.g. (bool) a + (bool) b could trigger a backend assertion
  if (LTy->isPromotableIntegerType()) {
    QualType NewTy = Ctx.getPromotedIntegerType(LTy);
    uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
    LHS = (*doCast)(LHS, NewTy, NewBitWidth, LTy, LBitWidth);
    LTy = NewTy;
    LBitWidth = NewBitWidth;
  }
  if (RTy->isPromotableIntegerType()) {
    QualType NewTy = Ctx.getPromotedIntegerType(RTy);
    uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
    RHS = (*doCast)(RHS, NewTy, NewBitWidth, RTy, RBitWidth);
    RTy = NewTy;
    RBitWidth = NewBitWidth;
  }

  if (LTy == RTy)
    return;

  // Perform integer type conversion
  // Note: Safe to skip updating bitwidth because this must terminate
  bool isLSignedTy = LTy->isSignedIntegerOrEnumerationType();
  bool isRSignedTy = RTy->isSignedIntegerOrEnumerationType();

  int order = Ctx.getIntegerTypeOrder(LTy, RTy);
  if (isLSignedTy == isRSignedTy) {
    // Same signedness; use the higher-ranked type
    if (order == 1) {
      RHS = (*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else {
      LHS = (*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    }
  } else if (order != (isLSignedTy ? 1 : -1)) {
    // The unsigned type has greater than or equal rank to the
    // signed type, so use the unsigned type
    if (isRSignedTy) {
      RHS = (*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else {
      LHS = (*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    }
  } else if (LBitWidth != RBitWidth) {
    // The two types are different widths; if we are here, that
    // means the signed type is larger than the unsigned type, so
    // use the signed type.
    if (isLSignedTy) {
      RHS = (*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else {
      LHS = (*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    }
  } else {
    // The signed type is higher-ranked than the unsigned type,
    // but isn't actually any bigger (like unsigned int and long
    // on most 32-bit systems).  Use the unsigned type corresponding
    // to the signed type.
    QualType NewTy = Ctx.getCorrespondingUnsignedType(isLSignedTy ? LTy : RTy);
    RHS = (*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
    RTy = NewTy;
    LHS = (*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
    LTy = NewTy;
  }
}

template <typename T,
          T(doCast)(const T &, QualType, uint64_t, QualType, uint64_t)>
void Z3ConstraintManager::doFloatTypeConversion(T &LHS, QualType &LTy, T &RHS,
                                                QualType &RTy) const {
  ASTContext &Ctx = getBasicVals().getContext();

  uint64_t LBitWidth = Ctx.getTypeSize(LTy);
  uint64_t RBitWidth = Ctx.getTypeSize(RTy);

  // Perform float-point type promotion
  if (!LTy->isRealFloatingType()) {
    LHS = (*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
    LTy = RTy;
    LBitWidth = RBitWidth;
  }
  if (!RTy->isRealFloatingType()) {
    RHS = (*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
    RTy = LTy;
    RBitWidth = LBitWidth;
  }

  if (LTy == RTy)
    return;

  // If we have two real floating types, convert the smaller operand to the
  // bigger result
  // Note: Safe to skip updating bitwidth because this must terminate
  int order = Ctx.getFloatingTypeOrder(LTy, RTy);
  if (order > 0) {
    RHS = Z3Expr::fromCast(RHS, LTy, LBitWidth, RTy, RBitWidth);
    RTy = LTy;
  } else if (order == 0) {
    LHS = Z3Expr::fromCast(LHS, RTy, RBitWidth, LTy, LBitWidth);
    LTy = RTy;
  } else {
    llvm_unreachable("Unsupported floating-point type cast!");
  }
}

llvm::APSInt Z3ConstraintManager::castAPSInt(const llvm::APSInt &V,
                                             QualType ToTy, uint64_t ToWidth,
                                             QualType FromTy,
                                             uint64_t FromWidth) {
  APSIntType TargetType(ToWidth, !ToTy->isSignedIntegerOrEnumerationType());
  return TargetType.convert(V);
}

//==------------------------------------------------------------------------==/
// Pretty-printing.
//==------------------------------------------------------------------------==/

void Z3ConstraintManager::print(ProgramStateRef St, raw_ostream &OS,
                                const char *nl, const char *sep) {

  ConstraintZ3Ty CZ = St->get<ConstraintZ3>();

  OS << nl << sep << "Constraints:";
  for (ConstraintZ3Ty::iterator I = CZ.begin(), E = CZ.end(); I != E; ++I) {
    OS << nl << ' ' << I->first << " : ";
    I->second.print(OS);
  }
  OS << nl;
}

#endif

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
