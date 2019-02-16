//== SMTSolver.h ------------------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic Solver API, which will be the base class
//  for every SMT solver specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTSOLVER_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTSOLVER_H

#include "clang/AST/Expr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTExpr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTSort.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"

namespace clang {
namespace ento {

/// Generic base class for SMT Solvers
///
/// This class is responsible for wrapping all sorts and expression generation,
/// through the mk* methods. It also provides methods to create SMT expressions
/// straight from clang's AST, through the from* methods.
class SMTSolver {
public:
  SMTSolver() = default;
  virtual ~SMTSolver() = default;

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

  // Returns an appropriate floating-point sort for the given bitwidth.
  SMTSortRef getFloatSort(unsigned BitWidth) {
    switch (BitWidth) {
    case 16:
      return getFloat16Sort();
    case 32:
      return getFloat32Sort();
    case 64:
      return getFloat64Sort();
    case 128:
      return getFloat128Sort();
    default:;
    }
    llvm_unreachable("Unsupported floating-point bitwidth!");
  }

  // Returns an appropriate sort, given a QualType and it's bit width.
  SMTSortRef mkSort(const QualType &Ty, unsigned BitWidth) {
    if (Ty->isBooleanType())
      return getBoolSort();

    if (Ty->isRealFloatingType())
      return getFloatSort(BitWidth);

    return getBitvectorSort(BitWidth);
  }

  /// Constructs an SMTExprRef from an unary operator.
  SMTExprRef fromUnOp(const UnaryOperator::Opcode Op, const SMTExprRef &Exp) {
    switch (Op) {
    case UO_Minus:
      return mkBVNeg(Exp);

    case UO_Not:
      return mkBVNot(Exp);

    case UO_LNot:
      return mkNot(Exp);

    default:;
    }
    llvm_unreachable("Unimplemented opcode");
  }

  /// Constructs an SMTExprRef from a floating-point unary operator.
  SMTExprRef fromFloatUnOp(const UnaryOperator::Opcode Op,
                           const SMTExprRef &Exp) {
    switch (Op) {
    case UO_Minus:
      return mkFPNeg(Exp);

    case UO_LNot:
      return fromUnOp(Op, Exp);

    default:;
    }
    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct an SMTExprRef from a n-ary binary operator.
  SMTExprRef fromNBinOp(const BinaryOperator::Opcode Op,
                        const std::vector<SMTExprRef> &ASTs) {
    assert(!ASTs.empty());

    if (Op != BO_LAnd && Op != BO_LOr)
      llvm_unreachable("Unimplemented opcode");

    SMTExprRef res = ASTs.front();
    for (std::size_t i = 1; i < ASTs.size(); ++i)
      res = (Op == BO_LAnd) ? mkAnd(res, ASTs[i]) : mkOr(res, ASTs[i]);
    return res;
  }

  /// Construct an SMTExprRef from a binary operator.
  SMTExprRef fromBinOp(const SMTExprRef &LHS, const BinaryOperator::Opcode Op,
                       const SMTExprRef &RHS, bool isSigned) {
    assert(*getSort(LHS) == *getSort(RHS) && "AST's must have the same sort!");

    switch (Op) {
    // Multiplicative operators
    case BO_Mul:
      return mkBVMul(LHS, RHS);

    case BO_Div:
      return isSigned ? mkBVSDiv(LHS, RHS) : mkBVUDiv(LHS, RHS);

    case BO_Rem:
      return isSigned ? mkBVSRem(LHS, RHS) : mkBVURem(LHS, RHS);

    // Additive operators
    case BO_Add:
      return mkBVAdd(LHS, RHS);

    case BO_Sub:
      return mkBVSub(LHS, RHS);

    // Bitwise shift operators
    case BO_Shl:
      return mkBVShl(LHS, RHS);

    case BO_Shr:
      return isSigned ? mkBVAshr(LHS, RHS) : mkBVLshr(LHS, RHS);

    // Relational operators
    case BO_LT:
      return isSigned ? mkBVSlt(LHS, RHS) : mkBVUlt(LHS, RHS);

    case BO_GT:
      return isSigned ? mkBVSgt(LHS, RHS) : mkBVUgt(LHS, RHS);

    case BO_LE:
      return isSigned ? mkBVSle(LHS, RHS) : mkBVUle(LHS, RHS);

    case BO_GE:
      return isSigned ? mkBVSge(LHS, RHS) : mkBVUge(LHS, RHS);

    // Equality operators
    case BO_EQ:
      return mkEqual(LHS, RHS);

    case BO_NE:
      return fromUnOp(UO_LNot, fromBinOp(LHS, BO_EQ, RHS, isSigned));

    // Bitwise operators
    case BO_And:
      return mkBVAnd(LHS, RHS);

    case BO_Xor:
      return mkBVXor(LHS, RHS);

    case BO_Or:
      return mkBVOr(LHS, RHS);

    // Logical operators
    case BO_LAnd:
      return mkAnd(LHS, RHS);

    case BO_LOr:
      return mkOr(LHS, RHS);

    default:;
    }
    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct an SMTExprRef from a special floating-point binary operator.
  SMTExprRef fromFloatSpecialBinOp(const SMTExprRef &LHS,
                                   const BinaryOperator::Opcode Op,
                                   const llvm::APFloat::fltCategory &RHS) {
    switch (Op) {
    // Equality operators
    case BO_EQ:
      switch (RHS) {
      case llvm::APFloat::fcInfinity:
        return mkFPIsInfinite(LHS);

      case llvm::APFloat::fcNaN:
        return mkFPIsNaN(LHS);

      case llvm::APFloat::fcNormal:
        return mkFPIsNormal(LHS);

      case llvm::APFloat::fcZero:
        return mkFPIsZero(LHS);
      }
      break;

    case BO_NE:
      return fromFloatUnOp(UO_LNot, fromFloatSpecialBinOp(LHS, BO_EQ, RHS));

    default:;
    }

    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct an SMTExprRef from a floating-point binary operator.
  SMTExprRef fromFloatBinOp(const SMTExprRef &LHS,
                            const BinaryOperator::Opcode Op,
                            const SMTExprRef &RHS) {
    assert(*getSort(LHS) == *getSort(RHS) && "AST's must have the same sort!");

    switch (Op) {
    // Multiplicative operators
    case BO_Mul:
      return mkFPMul(LHS, RHS);

    case BO_Div:
      return mkFPDiv(LHS, RHS);

    case BO_Rem:
      return mkFPRem(LHS, RHS);

      // Additive operators
    case BO_Add:
      return mkFPAdd(LHS, RHS);

    case BO_Sub:
      return mkFPSub(LHS, RHS);

      // Relational operators
    case BO_LT:
      return mkFPLt(LHS, RHS);

    case BO_GT:
      return mkFPGt(LHS, RHS);

    case BO_LE:
      return mkFPLe(LHS, RHS);

    case BO_GE:
      return mkFPGe(LHS, RHS);

      // Equality operators
    case BO_EQ:
      return mkFPEqual(LHS, RHS);

    case BO_NE:
      return fromFloatUnOp(UO_LNot, fromFloatBinOp(LHS, BO_EQ, RHS));

      // Logical operators
    case BO_LAnd:
    case BO_LOr:
      return fromBinOp(LHS, Op, RHS, false);

    default:;
    }

    llvm_unreachable("Unimplemented opcode");
  }

  /// Construct an SMTExprRef from a QualType FromTy to a QualType ToTy, and
  /// their bit widths.
  SMTExprRef fromCast(const SMTExprRef &Exp, QualType ToTy, uint64_t ToBitWidth,
                      QualType FromTy, uint64_t FromBitWidth) {
    if ((FromTy->isIntegralOrEnumerationType() &&
         ToTy->isIntegralOrEnumerationType()) ||
        (FromTy->isAnyPointerType() ^ ToTy->isAnyPointerType()) ||
        (FromTy->isBlockPointerType() ^ ToTy->isBlockPointerType()) ||
        (FromTy->isReferenceType() ^ ToTy->isReferenceType())) {

      if (FromTy->isBooleanType()) {
        assert(ToBitWidth > 0 && "BitWidth must be positive!");
        return mkIte(Exp, mkBitvector(llvm::APSInt("1"), ToBitWidth),
                     mkBitvector(llvm::APSInt("0"), ToBitWidth));
      }

      if (ToBitWidth > FromBitWidth)
        return FromTy->isSignedIntegerOrEnumerationType()
                   ? mkBVSignExt(ToBitWidth - FromBitWidth, Exp)
                   : mkBVZeroExt(ToBitWidth - FromBitWidth, Exp);

      if (ToBitWidth < FromBitWidth)
        return mkBVExtract(ToBitWidth - 1, 0, Exp);

      // Both are bitvectors with the same width, ignore the type cast
      return Exp;
    }

    if (FromTy->isRealFloatingType() && ToTy->isRealFloatingType()) {
      if (ToBitWidth != FromBitWidth)
        return mkFPtoFP(Exp, getFloatSort(ToBitWidth));

      return Exp;
    }

    if (FromTy->isIntegralOrEnumerationType() && ToTy->isRealFloatingType()) {
      SMTSortRef Sort = getFloatSort(ToBitWidth);
      return FromTy->isSignedIntegerOrEnumerationType() ? mkFPtoSBV(Exp, Sort)
                                                        : mkFPtoUBV(Exp, Sort);
    }

    if (FromTy->isRealFloatingType() && ToTy->isIntegralOrEnumerationType())
      return ToTy->isSignedIntegerOrEnumerationType()
                 ? mkSBVtoFP(Exp, ToBitWidth)
                 : mkUBVtoFP(Exp, ToBitWidth);

    llvm_unreachable("Unsupported explicit type cast!");
  }

  // Callback function for doCast parameter on APSInt type.
  llvm::APSInt castAPSInt(const llvm::APSInt &V, QualType ToTy,
                          uint64_t ToWidth, QualType FromTy,
                          uint64_t FromWidth) {
    APSIntType TargetType(ToWidth, !ToTy->isSignedIntegerOrEnumerationType());
    return TargetType.convert(V);
  }

  // Generate an SMTExprRef that represents the given symbolic expression.
  // Sets the hasComparison parameter if the expression has a comparison
  // operator.
  // Sets the RetTy parameter to the final return type after promotions and
  // casts.
  SMTExprRef getExpr(ASTContext &Ctx, SymbolRef Sym, QualType *RetTy = nullptr,
                     bool *hasComparison = nullptr) {
    if (hasComparison) {
      *hasComparison = false;
    }

    return getSymExpr(Ctx, Sym, RetTy, hasComparison);
  }

  // Generate an SMTExprRef that compares the expression to zero.
  SMTExprRef getZeroExpr(ASTContext &Ctx, const SMTExprRef &Exp, QualType Ty,
                         bool Assumption) {

    if (Ty->isRealFloatingType()) {
      llvm::APFloat Zero =
          llvm::APFloat::getZero(Ctx.getFloatTypeSemantics(Ty));
      return fromFloatBinOp(Exp, Assumption ? BO_EQ : BO_NE, fromAPFloat(Zero));
    }

    if (Ty->isIntegralOrEnumerationType() || Ty->isAnyPointerType() ||
        Ty->isBlockPointerType() || Ty->isReferenceType()) {

      // Skip explicit comparison for boolean types
      bool isSigned = Ty->isSignedIntegerOrEnumerationType();
      if (Ty->isBooleanType())
        return Assumption ? fromUnOp(UO_LNot, Exp) : Exp;

      return fromBinOp(Exp, Assumption ? BO_EQ : BO_NE,
                       fromInt("0", Ctx.getTypeSize(Ty)), isSigned);
    }

    llvm_unreachable("Unsupported type for zero value!");
  }

  // Recursive implementation to unpack and generate symbolic expression.
  // Sets the hasComparison and RetTy parameters. See getExpr().
  SMTExprRef getSymExpr(ASTContext &Ctx, SymbolRef Sym, QualType *RetTy,
                        bool *hasComparison) {
    if (const SymbolData *SD = dyn_cast<SymbolData>(Sym)) {
      if (RetTy)
        *RetTy = Sym->getType();

      return fromData(SD->getSymbolID(), Sym->getType(),
                      Ctx.getTypeSize(Sym->getType()));
    }

    if (const SymbolCast *SC = dyn_cast<SymbolCast>(Sym)) {
      if (RetTy)
        *RetTy = Sym->getType();

      QualType FromTy;
      SMTExprRef Exp =
          getSymExpr(Ctx, SC->getOperand(), &FromTy, hasComparison);
      // Casting an expression with a comparison invalidates it. Note that this
      // must occur after the recursive call above.
      // e.g. (signed char) (x > 0)
      if (hasComparison)
        *hasComparison = false;
      return getCastExpr(Ctx, Exp, FromTy, Sym->getType());
    }

    if (const BinarySymExpr *BSE = dyn_cast<BinarySymExpr>(Sym)) {
      SMTExprRef Exp = getSymBinExpr(Ctx, BSE, hasComparison, RetTy);
      // Set the hasComparison parameter, in post-order traversal order.
      if (hasComparison)
        *hasComparison = BinaryOperator::isComparisonOp(BSE->getOpcode());
      return Exp;
    }

    llvm_unreachable("Unsupported SymbolRef type!");
  }

  // Wrapper to generate SMTExprRef from SymbolCast data.
  SMTExprRef getCastExpr(ASTContext &Ctx, const SMTExprRef &Exp,
                         QualType FromTy, QualType ToTy) {
    return fromCast(Exp, ToTy, Ctx.getTypeSize(ToTy), FromTy,
                    Ctx.getTypeSize(FromTy));
  }

  // Wrapper to generate SMTExprRef from BinarySymExpr.
  // Sets the hasComparison and RetTy parameters. See getSMTExprRef().
  SMTExprRef getSymBinExpr(ASTContext &Ctx, const BinarySymExpr *BSE,
                           bool *hasComparison, QualType *RetTy) {
    QualType LTy, RTy;
    BinaryOperator::Opcode Op = BSE->getOpcode();

    if (const SymIntExpr *SIE = dyn_cast<SymIntExpr>(BSE)) {
      SMTExprRef LHS = getSymExpr(Ctx, SIE->getLHS(), &LTy, hasComparison);
      llvm::APSInt NewRInt;
      std::tie(NewRInt, RTy) = fixAPSInt(Ctx, SIE->getRHS());
      SMTExprRef RHS = fromAPSInt(NewRInt);
      return getBinExpr(Ctx, LHS, LTy, Op, RHS, RTy, RetTy);
    }

    if (const IntSymExpr *ISE = dyn_cast<IntSymExpr>(BSE)) {
      llvm::APSInt NewLInt;
      std::tie(NewLInt, LTy) = fixAPSInt(Ctx, ISE->getLHS());
      SMTExprRef LHS = fromAPSInt(NewLInt);
      SMTExprRef RHS = getSymExpr(Ctx, ISE->getRHS(), &RTy, hasComparison);
      return getBinExpr(Ctx, LHS, LTy, Op, RHS, RTy, RetTy);
    }

    if (const SymSymExpr *SSM = dyn_cast<SymSymExpr>(BSE)) {
      SMTExprRef LHS = getSymExpr(Ctx, SSM->getLHS(), &LTy, hasComparison);
      SMTExprRef RHS = getSymExpr(Ctx, SSM->getRHS(), &RTy, hasComparison);
      return getBinExpr(Ctx, LHS, LTy, Op, RHS, RTy, RetTy);
    }

    llvm_unreachable("Unsupported BinarySymExpr type!");
  }

  // Wrapper to generate SMTExprRef from unpacked binary symbolic expression.
  // Sets the RetTy parameter. See getSMTExprRef().
  SMTExprRef getBinExpr(ASTContext &Ctx, const SMTExprRef &LHS, QualType LTy,
                        BinaryOperator::Opcode Op, const SMTExprRef &RHS,
                        QualType RTy, QualType *RetTy) {
    SMTExprRef NewLHS = LHS;
    SMTExprRef NewRHS = RHS;
    doTypeConversion(Ctx, NewLHS, NewRHS, LTy, RTy);

    // Update the return type parameter if the output type has changed.
    if (RetTy) {
      // A boolean result can be represented as an integer type in C/C++, but at
      // this point we only care about the SMT sorts. Set it as a boolean type
      // to avoid subsequent SMT errors.
      if (BinaryOperator::isComparisonOp(Op) ||
          BinaryOperator::isLogicalOp(Op)) {
        *RetTy = Ctx.BoolTy;
      } else {
        *RetTy = LTy;
      }

      // If the two operands are pointers and the operation is a subtraction,
      // the result is of type ptrdiff_t, which is signed
      if (LTy->isAnyPointerType() && RTy->isAnyPointerType() && Op == BO_Sub) {
        *RetTy = Ctx.getPointerDiffType();
      }
    }

    return LTy->isRealFloatingType()
               ? fromFloatBinOp(NewLHS, Op, NewRHS)
               : fromBinOp(NewLHS, Op, NewRHS,
                           LTy->isSignedIntegerOrEnumerationType());
  }

  // Wrapper to generate SMTExprRef from a range. If From == To, an equality
  // will be created instead.
  SMTExprRef getRangeExpr(ASTContext &Ctx, SymbolRef Sym,
                          const llvm::APSInt &From, const llvm::APSInt &To,
                          bool InRange) {
    // Convert lower bound
    QualType FromTy;
    llvm::APSInt NewFromInt;
    std::tie(NewFromInt, FromTy) = fixAPSInt(Ctx, From);
    SMTExprRef FromExp = fromAPSInt(NewFromInt);

    // Convert symbol
    QualType SymTy;
    SMTExprRef Exp = getExpr(Ctx, Sym, &SymTy);

    // Construct single (in)equality
    if (From == To)
      return getBinExpr(Ctx, Exp, SymTy, InRange ? BO_EQ : BO_NE, FromExp,
                        FromTy, /*RetTy=*/nullptr);

    QualType ToTy;
    llvm::APSInt NewToInt;
    std::tie(NewToInt, ToTy) = fixAPSInt(Ctx, To);
    SMTExprRef ToExp = fromAPSInt(NewToInt);
    assert(FromTy == ToTy && "Range values have different types!");

    // Construct two (in)equalities, and a logical and/or
    SMTExprRef LHS = getBinExpr(Ctx, Exp, SymTy, InRange ? BO_GE : BO_LT,
                                FromExp, FromTy, /*RetTy=*/nullptr);
    SMTExprRef RHS =
        getBinExpr(Ctx, Exp, SymTy, InRange ? BO_LE : BO_GT, ToExp, ToTy,
                   /*RetTy=*/nullptr);

    return fromBinOp(LHS, InRange ? BO_LAnd : BO_LOr, RHS,
                     SymTy->isSignedIntegerOrEnumerationType());
  }

  // Recover the QualType of an APSInt.
  // TODO: Refactor to put elsewhere
  QualType getAPSIntType(ASTContext &Ctx, const llvm::APSInt &Int) {
    return Ctx.getIntTypeForBitwidth(Int.getBitWidth(), Int.isSigned());
  }

  // Get the QualTy for the input APSInt, and fix it if it has a bitwidth of 1.
  std::pair<llvm::APSInt, QualType> fixAPSInt(ASTContext &Ctx,
                                              const llvm::APSInt &Int) {
    llvm::APSInt NewInt;

    // FIXME: This should be a cast from a 1-bit integer type to a boolean type,
    // but the former is not available in Clang. Instead, extend the APSInt
    // directly.
    if (Int.getBitWidth() == 1 && getAPSIntType(Ctx, Int).isNull()) {
      NewInt = Int.extend(Ctx.getTypeSize(Ctx.BoolTy));
    } else
      NewInt = Int;

    return std::make_pair(NewInt, getAPSIntType(Ctx, NewInt));
  }

  // Perform implicit type conversion on binary symbolic expressions.
  // May modify all input parameters.
  // TODO: Refactor to use built-in conversion functions
  void doTypeConversion(ASTContext &Ctx, SMTExprRef &LHS, SMTExprRef &RHS,
                        QualType &LTy, QualType &RTy) {
    assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");

    // Perform type conversion
    if ((LTy->isIntegralOrEnumerationType() &&
         RTy->isIntegralOrEnumerationType()) &&
        (LTy->isArithmeticType() && RTy->isArithmeticType())) {
      doIntTypeConversion<SMTExprRef, &SMTSolver::fromCast>(Ctx, LHS, LTy, RHS,
                                                            RTy);
      return;
    }

    if (LTy->isRealFloatingType() || RTy->isRealFloatingType()) {
      doFloatTypeConversion<SMTExprRef, &SMTSolver::fromCast>(Ctx, LHS, LTy,
                                                              RHS, RTy);
      return;
    }

    if ((LTy->isAnyPointerType() || RTy->isAnyPointerType()) ||
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
          LHS = fromCast(LHS, RTy, RBitWidth, LTy, LBitWidth);
          LTy = RTy;
        } else {
          RHS = fromCast(RHS, LTy, LBitWidth, RTy, RBitWidth);
          RTy = LTy;
        }
      }

      // Cast the void pointer type to the non-void pointer type.
      // For void types, this assumes that the casted value is equal to the
      // value of the original pointer, and does not account for alignment
      // requirements.
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

  // Perform implicit integer type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleIntegerConversion()
  template <typename T, T (SMTSolver::*doCast)(const T &, QualType, uint64_t,
                                               QualType, uint64_t)>
  void doIntTypeConversion(ASTContext &Ctx, T &LHS, QualType &LTy, T &RHS,
                           QualType &RTy) {

    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    assert(!LTy.isNull() && !RTy.isNull() && "Input type is null!");
    // Always perform integer promotion before checking type equality.
    // Otherwise, e.g. (bool) a + (bool) b could trigger a backend assertion
    if (LTy->isPromotableIntegerType()) {
      QualType NewTy = Ctx.getPromotedIntegerType(LTy);
      uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
      LHS = (this->*doCast)(LHS, NewTy, NewBitWidth, LTy, LBitWidth);
      LTy = NewTy;
      LBitWidth = NewBitWidth;
    }
    if (RTy->isPromotableIntegerType()) {
      QualType NewTy = Ctx.getPromotedIntegerType(RTy);
      uint64_t NewBitWidth = Ctx.getTypeSize(NewTy);
      RHS = (this->*doCast)(RHS, NewTy, NewBitWidth, RTy, RBitWidth);
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
        RHS = (this->*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      } else {
        LHS = (this->*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      }
    } else if (order != (isLSignedTy ? 1 : -1)) {
      // The unsigned type has greater than or equal rank to the
      // signed type, so use the unsigned type
      if (isRSignedTy) {
        RHS = (this->*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      } else {
        LHS = (this->*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      }
    } else if (LBitWidth != RBitWidth) {
      // The two types are different widths; if we are here, that
      // means the signed type is larger than the unsigned type, so
      // use the signed type.
      if (isLSignedTy) {
        RHS = (this->*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
        RTy = LTy;
      } else {
        LHS = (this->*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
        LTy = RTy;
      }
    } else {
      // The signed type is higher-ranked than the unsigned type,
      // but isn't actually any bigger (like unsigned int and long
      // on most 32-bit systems).  Use the unsigned type corresponding
      // to the signed type.
      QualType NewTy =
          Ctx.getCorrespondingUnsignedType(isLSignedTy ? LTy : RTy);
      RHS = (this->*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = NewTy;
      LHS = (this->*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = NewTy;
    }
  }

  // Perform implicit floating-point type conversion.
  // May modify all input parameters.
  // TODO: Refactor to use Sema::handleFloatConversion()
  template <typename T, T (SMTSolver::*doCast)(const T &, QualType, uint64_t,
                                               QualType, uint64_t)>
  void doFloatTypeConversion(ASTContext &Ctx, T &LHS, QualType &LTy, T &RHS,
                             QualType &RTy) {

    uint64_t LBitWidth = Ctx.getTypeSize(LTy);
    uint64_t RBitWidth = Ctx.getTypeSize(RTy);

    // Perform float-point type promotion
    if (!LTy->isRealFloatingType()) {
      LHS = (this->*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
      LBitWidth = RBitWidth;
    }
    if (!RTy->isRealFloatingType()) {
      RHS = (this->*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
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
      RHS = (this->*doCast)(RHS, LTy, LBitWidth, RTy, RBitWidth);
      RTy = LTy;
    } else if (order == 0) {
      LHS = (this->*doCast)(LHS, RTy, RBitWidth, LTy, LBitWidth);
      LTy = RTy;
    } else {
      llvm_unreachable("Unsupported floating-point type cast!");
    }
  }

  // Returns a boolean sort.
  virtual SMTSortRef getBoolSort() = 0;

  // Returns an appropriate bitvector sort for the given bitwidth.
  virtual SMTSortRef getBitvectorSort(const unsigned BitWidth) = 0;

  // Returns a floating-point sort of width 16
  virtual SMTSortRef getFloat16Sort() = 0;

  // Returns a floating-point sort of width 32
  virtual SMTSortRef getFloat32Sort() = 0;

  // Returns a floating-point sort of width 64
  virtual SMTSortRef getFloat64Sort() = 0;

  // Returns a floating-point sort of width 128
  virtual SMTSortRef getFloat128Sort() = 0;

  // Returns an appropriate sort for the given AST.
  virtual SMTSortRef getSort(const SMTExprRef &AST) = 0;

  // Returns a new SMTExprRef from an SMTExpr
  virtual SMTExprRef newExprRef(const SMTExpr &E) const = 0;

  /// Given a constraint, adds it to the solver
  virtual void addConstraint(const SMTExprRef &Exp) const = 0;

  /// Creates a bitvector addition operation
  virtual SMTExprRef mkBVAdd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector subtraction operation
  virtual SMTExprRef mkBVSub(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector multiplication operation
  virtual SMTExprRef mkBVMul(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector signed modulus operation
  virtual SMTExprRef mkBVSRem(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector unsigned modulus operation
  virtual SMTExprRef mkBVURem(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector signed division operation
  virtual SMTExprRef mkBVSDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector unsigned division operation
  virtual SMTExprRef mkBVUDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector logical shift left operation
  virtual SMTExprRef mkBVShl(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector arithmetic shift right operation
  virtual SMTExprRef mkBVAshr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector logical shift right operation
  virtual SMTExprRef mkBVLshr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector negation operation
  virtual SMTExprRef mkBVNeg(const SMTExprRef &Exp) = 0;

  /// Creates a bitvector not operation
  virtual SMTExprRef mkBVNot(const SMTExprRef &Exp) = 0;

  /// Creates a bitvector xor operation
  virtual SMTExprRef mkBVXor(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector or operation
  virtual SMTExprRef mkBVOr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector and operation
  virtual SMTExprRef mkBVAnd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector unsigned less-than operation
  virtual SMTExprRef mkBVUlt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector signed less-than operation
  virtual SMTExprRef mkBVSlt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector unsigned greater-than operation
  virtual SMTExprRef mkBVUgt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector signed greater-than operation
  virtual SMTExprRef mkBVSgt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector unsigned less-equal-than operation
  virtual SMTExprRef mkBVUle(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector signed less-equal-than operation
  virtual SMTExprRef mkBVSle(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector unsigned greater-equal-than operation
  virtual SMTExprRef mkBVUge(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a bitvector signed greater-equal-than operation
  virtual SMTExprRef mkBVSge(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a boolean not operation
  virtual SMTExprRef mkNot(const SMTExprRef &Exp) = 0;

  /// Creates a boolean equality operation
  virtual SMTExprRef mkEqual(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a boolean and operation
  virtual SMTExprRef mkAnd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a boolean or operation
  virtual SMTExprRef mkOr(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a boolean ite operation
  virtual SMTExprRef mkIte(const SMTExprRef &Cond, const SMTExprRef &T,
                           const SMTExprRef &F) = 0;

  /// Creates a bitvector sign extension operation
  virtual SMTExprRef mkBVSignExt(unsigned i, const SMTExprRef &Exp) = 0;

  /// Creates a bitvector zero extension operation
  virtual SMTExprRef mkBVZeroExt(unsigned i, const SMTExprRef &Exp) = 0;

  /// Creates a bitvector extract operation
  virtual SMTExprRef mkBVExtract(unsigned High, unsigned Low,
                                 const SMTExprRef &Exp) = 0;

  /// Creates a bitvector concat operation
  virtual SMTExprRef mkBVConcat(const SMTExprRef &LHS,
                                const SMTExprRef &RHS) = 0;

  /// Creates a floating-point negation operation
  virtual SMTExprRef mkFPNeg(const SMTExprRef &Exp) = 0;

  /// Creates a floating-point isInfinite operation
  virtual SMTExprRef mkFPIsInfinite(const SMTExprRef &Exp) = 0;

  /// Creates a floating-point isNaN operation
  virtual SMTExprRef mkFPIsNaN(const SMTExprRef &Exp) = 0;

  /// Creates a floating-point isNormal operation
  virtual SMTExprRef mkFPIsNormal(const SMTExprRef &Exp) = 0;

  /// Creates a floating-point isZero operation
  virtual SMTExprRef mkFPIsZero(const SMTExprRef &Exp) = 0;

  /// Creates a floating-point multiplication operation
  virtual SMTExprRef mkFPMul(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point division operation
  virtual SMTExprRef mkFPDiv(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point remainder operation
  virtual SMTExprRef mkFPRem(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point addition operation
  virtual SMTExprRef mkFPAdd(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point subtraction operation
  virtual SMTExprRef mkFPSub(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point less-than operation
  virtual SMTExprRef mkFPLt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point greater-than operation
  virtual SMTExprRef mkFPGt(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point less-than-or-equal operation
  virtual SMTExprRef mkFPLe(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point greater-than-or-equal operation
  virtual SMTExprRef mkFPGe(const SMTExprRef &LHS, const SMTExprRef &RHS) = 0;

  /// Creates a floating-point equality operation
  virtual SMTExprRef mkFPEqual(const SMTExprRef &LHS,
                               const SMTExprRef &RHS) = 0;

  /// Creates a floating-point conversion from floatint-point to floating-point
  /// operation
  virtual SMTExprRef mkFPtoFP(const SMTExprRef &From, const SMTSortRef &To) = 0;

  /// Creates a floating-point conversion from floatint-point to signed
  /// bitvector operation
  virtual SMTExprRef mkFPtoSBV(const SMTExprRef &From,
                               const SMTSortRef &To) = 0;

  /// Creates a floating-point conversion from floatint-point to unsigned
  /// bitvector operation
  virtual SMTExprRef mkFPtoUBV(const SMTExprRef &From,
                               const SMTSortRef &To) = 0;

  /// Creates a floating-point conversion from signed bitvector to
  /// floatint-point operation
  virtual SMTExprRef mkSBVtoFP(const SMTExprRef &From, unsigned ToWidth) = 0;

  /// Creates a floating-point conversion from unsigned bitvector to
  /// floatint-point operation
  virtual SMTExprRef mkUBVtoFP(const SMTExprRef &From, unsigned ToWidth) = 0;

  /// Creates a new symbol, given a name and a sort
  virtual SMTExprRef mkSymbol(const char *Name, SMTSortRef Sort) = 0;

  // Returns an appropriate floating-point rounding mode.
  virtual SMTExprRef getFloatRoundingMode() = 0;

  // If the a model is available, returns the value of a given bitvector symbol
  virtual llvm::APSInt getBitvector(const SMTExprRef &Exp, unsigned BitWidth,
                                    bool isUnsigned) = 0;

  // If the a model is available, returns the value of a given boolean symbol
  virtual bool getBoolean(const SMTExprRef &Exp) = 0;

  /// Constructs an SMTExprRef from a boolean.
  virtual SMTExprRef mkBoolean(const bool b) = 0;

  /// Constructs an SMTExprRef from a finite APFloat.
  virtual SMTExprRef mkFloat(const llvm::APFloat Float) = 0;

  /// Constructs an SMTExprRef from an APSInt and its bit width
  virtual SMTExprRef mkBitvector(const llvm::APSInt Int, unsigned BitWidth) = 0;

  /// Given an expression, extract the value of this operand in the model.
  virtual bool getInterpretation(const SMTExprRef &Exp, llvm::APSInt &Int) = 0;

  /// Given an expression extract the value of this operand in the model.
  virtual bool getInterpretation(const SMTExprRef &Exp,
                                 llvm::APFloat &Float) = 0;

  /// Construct an SMTExprRef value from a boolean.
  virtual SMTExprRef fromBoolean(const bool Bool) = 0;

  /// Construct an SMTExprRef value from a finite APFloat.
  virtual SMTExprRef fromAPFloat(const llvm::APFloat &Float) = 0;

  /// Construct an SMTExprRef value from an APSInt.
  virtual SMTExprRef fromAPSInt(const llvm::APSInt &Int) = 0;

  /// Construct an SMTExprRef value from an integer.
  virtual SMTExprRef fromInt(const char *Int, uint64_t BitWidth) = 0;

  /// Construct an SMTExprRef from a SymbolData.
  virtual SMTExprRef fromData(const SymbolID ID, const QualType &Ty,
                              uint64_t BitWidth) = 0;

  /// Check if the constraints are satisfiable
  virtual ConditionTruthVal check() const = 0;

  /// Push the current solver state
  virtual void push() = 0;

  /// Pop the previous solver state
  virtual void pop(unsigned NumStates = 1) = 0;

  /// Reset the solver and remove all constraints.
  virtual void reset() const = 0;

  virtual void print(raw_ostream &OS) const = 0;
};

/// Shared pointer for SMTSolvers.
using SMTSolverRef = std::shared_ptr<SMTSolver>;

/// Convenience method to create and Z3Solver object
std::unique_ptr<SMTSolver> CreateZ3Solver();

} // namespace ento
} // namespace clang

#endif
