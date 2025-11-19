//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as CIR or a function call to be
// later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCall.h"
#include "CIRGenConstantEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/Expr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace llvm;

static RValue emitLibraryCall(CIRGenFunction &cgf, const FunctionDecl *fd,
                              const CallExpr *e, mlir::Operation *calleeValue) {
  CIRGenCallee callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(fd));
  return cgf.emitCall(e->getCallee()->getType(), callee, e, ReturnValueSlot());
}

template <typename Op>
static RValue emitBuiltinBitOp(CIRGenFunction &cgf, const CallExpr *e,
                               bool poisonZero = false) {
  assert(!cir::MissingFeatures::builtinCheckKind());

  mlir::Value arg = cgf.emitScalarExpr(e->getArg(0));
  CIRGenBuilderTy &builder = cgf.getBuilder();

  Op op;
  if constexpr (std::is_same_v<Op, cir::BitClzOp> ||
                std::is_same_v<Op, cir::BitCtzOp>)
    op = Op::create(builder, cgf.getLoc(e->getSourceRange()), arg, poisonZero);
  else
    op = Op::create(builder, cgf.getLoc(e->getSourceRange()), arg);

  mlir::Value result = op.getResult();
  mlir::Type exprTy = cgf.convertType(e->getType());
  if (exprTy != result.getType())
    result = builder.createIntCast(result, exprTy);

  return RValue::get(result);
}

namespace {
struct WidthAndSignedness {
  unsigned width;
  bool isSigned;
};
} // namespace

static WidthAndSignedness
getIntegerWidthAndSignedness(const clang::ASTContext &astContext,
                             const clang::QualType type) {
  assert(type->isIntegerType() && "Given type is not an integer.");
  unsigned width = type->isBooleanType()  ? 1
                   : type->isBitIntType() ? astContext.getIntWidth(type)
                                          : astContext.getTypeInfo(type).Width;
  bool isSigned = type->isSignedIntegerType();
  return {width, isSigned};
}

// Given one or more integer types, this function produces an integer type that
// encompasses them: any value in one of the given types could be expressed in
// the encompassing type.
static struct WidthAndSignedness
EncompassingIntegerType(ArrayRef<struct WidthAndSignedness> types) {
  assert(types.size() > 0 && "Empty list of types.");

  // If any of the given types is signed, we must return a signed type.
  bool isSigned = llvm::any_of(types, [](const auto &t) { return t.isSigned; });

  // The encompassing type must have a width greater than or equal to the width
  // of the specified types.  Additionally, if the encompassing type is signed,
  // its width must be strictly greater than the width of any unsigned types
  // given.
  unsigned width = 0;
  for (const auto &type : types)
    width = std::max(width, type.width + (isSigned && !type.isSigned));

  return {width, isSigned};
}

RValue CIRGenFunction::emitRotate(const CallExpr *e, bool isRotateLeft) {
  mlir::Value input = emitScalarExpr(e->getArg(0));
  mlir::Value amount = emitScalarExpr(e->getArg(1));

  // TODO(cir): MSVC flavor bit rotate builtins use different types for input
  // and amount, but cir.rotate requires them to have the same type. Cast amount
  // to the type of input when necessary.
  assert(!cir::MissingFeatures::msvcBuiltins());

  auto r = cir::RotateOp::create(builder, getLoc(e->getSourceRange()), input,
                                 amount, isRotateLeft);
  return RValue::get(r);
}

template <class Operation>
static RValue emitUnaryMaybeConstrainedFPBuiltin(CIRGenFunction &cgf,
                                                 const CallExpr &e) {
  mlir::Value arg = cgf.emitScalarExpr(e.getArg(0));

  assert(!cir::MissingFeatures::cgFPOptionsRAII());
  assert(!cir::MissingFeatures::fpConstraints());

  auto call =
      Operation::create(cgf.getBuilder(), arg.getLoc(), arg.getType(), arg);
  return RValue::get(call->getResult(0));
}

template <class Operation>
static RValue emitUnaryFPBuiltin(CIRGenFunction &cgf, const CallExpr &e) {
  mlir::Value arg = cgf.emitScalarExpr(e.getArg(0));
  auto call =
      Operation::create(cgf.getBuilder(), arg.getLoc(), arg.getType(), arg);
  return RValue::get(call->getResult(0));
}

RValue CIRGenFunction::emitBuiltinExpr(const GlobalDecl &gd, unsigned builtinID,
                                       const CallExpr *e,
                                       ReturnValueSlot returnValue) {
  mlir::Location loc = getLoc(e->getSourceRange());

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult result;
  if (e->isPRValue() && e->EvaluateAsRValue(result, cgm.getASTContext()) &&
      !result.hasSideEffects()) {
    if (result.Val.isInt())
      return RValue::get(builder.getConstInt(loc, result.Val.getInt()));
    if (result.Val.isFloat()) {
      // Note: we are using result type of CallExpr to determine the type of
      // the constant. Classic codegen uses the result value to determine the
      // type. We feel it should be Ok to use expression type because it is
      // hard to imagine a builtin function evaluates to a value that
      // over/underflows its own defined type.
      mlir::Type type = convertType(e->getType());
      return RValue::get(builder.getConstFP(loc, type, result.Val.getFloat()));
    }
  }

  const FunctionDecl *fd = gd.getDecl()->getAsFunction();

  assert(!cir::MissingFeatures::builtinCallF128());

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  unsigned builtinIDIfNoAsmLabel = fd->hasAttr<AsmLabelAttr>() ? 0 : builtinID;

  assert(!cir::MissingFeatures::builtinCallMathErrno());
  assert(!cir::MissingFeatures::builtinCall());

  switch (builtinIDIfNoAsmLabel) {
  default:
    break;

  // C stdarg builtins.
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__va_start: {
    mlir::Value vaList = builtinID == Builtin::BI__va_start
                             ? emitScalarExpr(e->getArg(0))
                             : emitVAListRef(e->getArg(0)).getPointer();
    mlir::Value count = emitScalarExpr(e->getArg(1));
    emitVAStart(vaList, count);
    return {};
  }

  case Builtin::BI__builtin_va_end:
    emitVAEnd(emitVAListRef(e->getArg(0)).getPointer());
    return {};

  case Builtin::BIalloca:
  case Builtin::BI_alloca:
  case Builtin::BI__builtin_alloca_uninitialized:
  case Builtin::BI__builtin_alloca: {
    // Get alloca size input
    mlir::Value size = emitScalarExpr(e->getArg(0));

    // The alignment of the alloca should correspond to __BIGGEST_ALIGNMENT__.
    const TargetInfo &ti = getContext().getTargetInfo();
    const CharUnits suitableAlignmentInBytes =
        getContext().toCharUnitsFromBits(ti.getSuitableAlign());

    // Emit the alloca op with type `u8 *` to match the semantics of
    // `llvm.alloca`. We later bitcast the type to `void *` to match the
    // semantics of C/C++
    // FIXME(cir): It may make sense to allow AllocaOp of type `u8` to return a
    // pointer of type `void *`. This will require a change to the allocaOp
    // verifier.
    mlir::Value allocaAddr = builder.createAlloca(
        getLoc(e->getSourceRange()), builder.getUInt8PtrTy(),
        builder.getUInt8Ty(), "bi_alloca", suitableAlignmentInBytes, size);

    // Initialize the allocated buffer if required.
    if (builtinID != Builtin::BI__builtin_alloca_uninitialized) {
      // Initialize the alloca with the given size and alignment according to
      // the lang opts. Only the trivial non-initialization is supported for
      // now.

      switch (getLangOpts().getTrivialAutoVarInit()) {
      case LangOptions::TrivialAutoVarInitKind::Uninitialized:
        // Nothing to initialize.
        break;
      case LangOptions::TrivialAutoVarInitKind::Zero:
      case LangOptions::TrivialAutoVarInitKind::Pattern:
        cgm.errorNYI("trivial auto var init");
        break;
      }
    }

    // An alloca will always return a pointer to the alloca (stack) address
    // space. This address space need not be the same as the AST / Language
    // default (e.g. in C / C++ auto vars are in the generic address space). At
    // the AST level this is handled within CreateTempAlloca et al., but for the
    // builtin / dynamic alloca we have to handle it here.
    assert(!cir::MissingFeatures::addressSpace());

    // Bitcast the alloca to the expected type.
    return RValue::get(
        builder.createBitcast(allocaAddr, builder.getVoidPtrTy()));
  }

  case Builtin::BIcos:
  case Builtin::BIcosf:
  case Builtin::BIcosl:
  case Builtin::BI__builtin_cos:
  case Builtin::BI__builtin_cosf:
  case Builtin::BI__builtin_cosf16:
  case Builtin::BI__builtin_cosl:
  case Builtin::BI__builtin_cosf128:
    assert(!cir::MissingFeatures::fastMathFlags());
    return emitUnaryMaybeConstrainedFPBuiltin<cir::CosOp>(*this, *e);

  case Builtin::BIceil:
  case Builtin::BIceilf:
  case Builtin::BIceill:
  case Builtin::BI__builtin_ceil:
  case Builtin::BI__builtin_ceilf:
  case Builtin::BI__builtin_ceilf16:
  case Builtin::BI__builtin_ceill:
  case Builtin::BI__builtin_ceilf128:
    assert(!cir::MissingFeatures::fastMathFlags());
    return emitUnaryMaybeConstrainedFPBuiltin<cir::CeilOp>(*this, *e);

  case Builtin::BIfabs:
  case Builtin::BIfabsf:
  case Builtin::BIfabsl:
  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsf16:
  case Builtin::BI__builtin_fabsl:
  case Builtin::BI__builtin_fabsf128:
    return emitUnaryMaybeConstrainedFPBuiltin<cir::FAbsOp>(*this, *e);

  case Builtin::BI__assume:
  case Builtin::BI__builtin_assume: {
    if (e->getArg(0)->HasSideEffects(getContext()))
      return RValue::get(nullptr);

    mlir::Value argValue = emitCheckedArgForAssume(e->getArg(0));
    cir::AssumeOp::create(builder, loc, argValue);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_assume_separate_storage: {
    mlir::Value value0 = emitScalarExpr(e->getArg(0));
    mlir::Value value1 = emitScalarExpr(e->getArg(1));
    cir::AssumeSepStorageOp::create(builder, loc, value0, value1);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_assume_aligned: {
    const Expr *ptrExpr = e->getArg(0);
    mlir::Value ptrValue = emitScalarExpr(ptrExpr);
    mlir::Value offsetValue =
        (e->getNumArgs() > 2) ? emitScalarExpr(e->getArg(2)) : nullptr;

    std::optional<llvm::APSInt> alignment =
        e->getArg(1)->getIntegerConstantExpr(getContext());
    assert(alignment.has_value() &&
           "the second argument to __builtin_assume_aligned must be an "
           "integral constant expression");

    mlir::Value result =
        emitAlignmentAssumption(ptrValue, ptrExpr, ptrExpr->getExprLoc(),
                                alignment->getSExtValue(), offsetValue);
    return RValue::get(result);
  }

  case Builtin::BI__builtin_complex: {
    mlir::Value real = emitScalarExpr(e->getArg(0));
    mlir::Value imag = emitScalarExpr(e->getArg(1));
    mlir::Value complex = builder.createComplexCreate(loc, real, imag);
    return RValue::getComplex(complex);
  }

  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall:
  case Builtin::BIcreal:
  case Builtin::BIcrealf:
  case Builtin::BIcreall: {
    mlir::Value complex = emitComplexExpr(e->getArg(0));
    mlir::Value real = builder.createComplexReal(loc, complex);
    return RValue::get(real);
  }

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl:
  case Builtin::BIcimag:
  case Builtin::BIcimagf:
  case Builtin::BIcimagl: {
    mlir::Value complex = emitComplexExpr(e->getArg(0));
    mlir::Value imag = builder.createComplexImag(loc, complex);
    return RValue::get(imag);
  }

  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl:
  case Builtin::BIconj:
  case Builtin::BIconjf:
  case Builtin::BIconjl: {
    mlir::Value complex = emitComplexExpr(e->getArg(0));
    mlir::Value conj = builder.createUnaryOp(getLoc(e->getExprLoc()),
                                             cir::UnaryOpKind::Not, complex);
    return RValue::getComplex(conj);
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return emitBuiltinBitOp<cir::BitClrsbOp>(*this, e);

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzg:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitCtzOp>(*this, e, /*poisonZero=*/true);

  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzg:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitClzOp>(*this, e, /*poisonZero=*/true);

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    return emitBuiltinBitOp<cir::BitFfsOp>(*this, e);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return emitBuiltinBitOp<cir::BitParityOp>(*this, e);

  case Builtin::BI__lzcnt16:
  case Builtin::BI__lzcnt:
  case Builtin::BI__lzcnt64:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitClzOp>(*this, e, /*poisonZero=*/false);

  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg:
    return emitBuiltinBitOp<cir::BitPopcountOp>(*this, e);

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability: {
    mlir::Value argValue = emitScalarExpr(e->getArg(0));
    mlir::Value expectedValue = emitScalarExpr(e->getArg(1));

    mlir::FloatAttr probAttr;
    if (builtinIDIfNoAsmLabel == Builtin::BI__builtin_expect_with_probability) {
      llvm::APFloat probability(0.0);
      const Expr *probArg = e->getArg(2);
      [[maybe_unused]] bool evalSucceeded =
          probArg->EvaluateAsFloat(probability, cgm.getASTContext());
      assert(evalSucceeded &&
             "probability should be able to evaluate as float");
      bool loseInfo = false; // ignored
      probability.convert(llvm::APFloat::IEEEdouble(),
                          llvm::RoundingMode::Dynamic, &loseInfo);
      probAttr = mlir::FloatAttr::get(mlir::Float64Type::get(&getMLIRContext()),
                                      probability);
    }

    auto result = cir::ExpectOp::create(builder, loc, argValue.getType(),
                                        argValue, expectedValue, probAttr);
    return RValue::get(result);
  }

  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64:
  case Builtin::BI_byteswap_ushort:
  case Builtin::BI_byteswap_ulong:
  case Builtin::BI_byteswap_uint64: {
    mlir::Value arg = emitScalarExpr(e->getArg(0));
    return RValue::get(cir::ByteSwapOp::create(builder, loc, arg));
  }

  case Builtin::BI__builtin_bitreverse8:
  case Builtin::BI__builtin_bitreverse16:
  case Builtin::BI__builtin_bitreverse32:
  case Builtin::BI__builtin_bitreverse64: {
    mlir::Value arg = emitScalarExpr(e->getArg(0));
    return RValue::get(cir::BitReverseOp::create(builder, loc, arg));
  }

  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
    return emitRotate(e, /*isRotateLeft=*/true);

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
    return emitRotate(e, /*isRotateLeft=*/false);

  case Builtin::BI__builtin_return_address:
  case Builtin::BI__builtin_frame_address: {
    mlir::Location loc = getLoc(e->getExprLoc());
    llvm::APSInt level = e->getArg(0)->EvaluateKnownConstInt(getContext());
    if (builtinID == Builtin::BI__builtin_return_address) {
      return RValue::get(cir::ReturnAddrOp::create(
          builder, loc,
          builder.getConstAPInt(loc, builder.getUInt32Ty(), level)));
    }
    return RValue::get(cir::FrameAddrOp::create(
        builder, loc,
        builder.getConstAPInt(loc, builder.getUInt32Ty(), level)));
  }

  case Builtin::BI__builtin_trap:
    emitTrap(loc, /*createNewBlock=*/true);
    return RValue::get(nullptr);

  case Builtin::BI__builtin_unreachable:
    emitUnreachable(e->getExprLoc(), /*createNewBlock=*/true);
    return RValue::get(nullptr);

  case Builtin::BI__builtin_elementwise_acos:
    return emitUnaryFPBuiltin<cir::ACosOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_asin:
    return emitUnaryFPBuiltin<cir::ASinOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_atan:
    return emitUnaryFPBuiltin<cir::ATanOp>(*this, *e);
  case Builtin::BI__builtin_elementwise_cos:
    return emitUnaryFPBuiltin<cir::CosOp>(*this, *e);
  case Builtin::BI__builtin_coro_id:
  case Builtin::BI__builtin_coro_promise:
  case Builtin::BI__builtin_coro_resume:
  case Builtin::BI__builtin_coro_noop:
  case Builtin::BI__builtin_coro_destroy:
  case Builtin::BI__builtin_coro_done:
  case Builtin::BI__builtin_coro_alloc:
  case Builtin::BI__builtin_coro_begin:
  case Builtin::BI__builtin_coro_end:
  case Builtin::BI__builtin_coro_suspend:
  case Builtin::BI__builtin_coro_align:
    cgm.errorNYI(e->getSourceRange(), "BI__builtin_coro_id like NYI");
    return getUndefRValue(e->getType());

  case Builtin::BI__builtin_coro_frame: {
    cgm.errorNYI(e->getSourceRange(), "BI__builtin_coro_frame NYI");
    assert(!cir::MissingFeatures::coroutineFrame());
    return getUndefRValue(e->getType());
  }
  case Builtin::BI__builtin_coro_free:
  case Builtin::BI__builtin_coro_size: {
    GlobalDecl gd{fd};
    mlir::Type ty = cgm.getTypes().getFunctionType(
        cgm.getTypes().arrangeGlobalDeclaration(gd));
    const auto *nd = cast<NamedDecl>(gd.getDecl());
    cir::FuncOp fnOp =
        cgm.getOrCreateCIRFunction(nd->getName(), ty, gd, /*ForVTable=*/false);
    fnOp.setBuiltin(true);
    return emitCall(e->getCallee()->getType(), CIRGenCallee::forDirect(fnOp), e,
                    returnValue);
  }
  case Builtin::BI__builtin_prefetch: {
    auto evaluateOperandAsInt = [&](const Expr *arg) {
      Expr::EvalResult res;
      [[maybe_unused]] bool evalSucceed =
          arg->EvaluateAsInt(res, cgm.getASTContext());
      assert(evalSucceed && "expression should be able to evaluate as int");
      return res.Val.getInt().getZExtValue();
    };

    bool isWrite = false;
    if (e->getNumArgs() > 1)
      isWrite = evaluateOperandAsInt(e->getArg(1));

    int locality = 3;
    if (e->getNumArgs() > 2)
      locality = evaluateOperandAsInt(e->getArg(2));

    mlir::Value address = emitScalarExpr(e->getArg(0));
    cir::PrefetchOp::create(builder, loc, address, locality, isWrite);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_mul_overflow: {
    const clang::Expr *LeftArg = e->getArg(0);
    const clang::Expr *RightArg = e->getArg(1);
    const clang::Expr *ResultArg = e->getArg(2);

    clang::QualType ResultQTy =
        ResultArg->getType()->castAs<clang::PointerType>()->getPointeeType();

    WidthAndSignedness LeftInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), LeftArg->getType());
    WidthAndSignedness RightInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), RightArg->getType());
    WidthAndSignedness ResultInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), ResultQTy);

    // Note we compute the encompassing type with the consideration to the
    // result type, so later in LLVM lowering we don't get redundant integral
    // extension casts.
    WidthAndSignedness EncompassingInfo =
        EncompassingIntegerType({LeftInfo, RightInfo, ResultInfo});

    auto EncompassingCIRTy = cir::IntType::get(
        &getMLIRContext(), EncompassingInfo.width, EncompassingInfo.isSigned);
    auto ResultCIRTy = mlir::cast<cir::IntType>(cgm.convertType(ResultQTy));

    mlir::Value Left = emitScalarExpr(LeftArg);
    mlir::Value Right = emitScalarExpr(RightArg);
    Address ResultPtr = emitPointerWithAlignment(ResultArg);

    // Extend each operand to the encompassing type, if necessary.
    if (Left.getType() != EncompassingCIRTy)
      Left =
          builder.createCast(cir::CastKind::integral, Left, EncompassingCIRTy);
    if (Right.getType() != EncompassingCIRTy)
      Right =
          builder.createCast(cir::CastKind::integral, Right, EncompassingCIRTy);

    // Perform the operation on the extended values.
    cir::BinOpOverflowKind OpKind;
    switch (builtinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_add_overflow:
      OpKind = cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_sub_overflow:
      OpKind = cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_mul_overflow:
      OpKind = cir::BinOpOverflowKind::Mul;
      break;
    }

    auto Loc = getLoc(e->getSourceRange());
    auto arithOp =
        cir::BinOpOverflowOp::create(builder, Loc, ResultCIRTy, OpKind, Left, Right);

    // Here is a slight difference from the original clang CodeGen:
    //   - In the original clang CodeGen, the checked arithmetic result is
    //     first computed as a value of the encompassing type, and then it is
    //     truncated to the actual result type with a second overflow checking.
    //   - In CIRGen, the checked arithmetic operation directly produce the
    //     checked arithmetic result in its expected type.
    //
    // So we don't need a truncation and a second overflow checking here.

    // Finally, store the result using the pointer.
    bool isVolatile =
        ResultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(Loc, emitToMemory(ArithOp.getResult(), ResultQTy),
                        ResultPtr, isVolatile);

    return RValue::get(ArithOp.getOverflow());
  }

  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow: {
    // Scalarize our inputs.
    mlir::Value X = emitScalarExpr(e->getArg(0));
    mlir::Value Y = emitScalarExpr(e->getArg(1));

    const clang::Expr *ResultArg = e->getArg(2);
    Address ResultPtr = emitPointerWithAlignment(ResultArg);

    // Decide which of the arithmetic operation we are lowering to:
    cir::BinOpOverflowKind ArithKind;
    switch (builtinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_uadd_overflow:
    case Builtin::BI__builtin_uaddl_overflow:
    case Builtin::BI__builtin_uaddll_overflow:
    case Builtin::BI__builtin_sadd_overflow:
    case Builtin::BI__builtin_saddl_overflow:
    case Builtin::BI__builtin_saddll_overflow:
      ArithKind = cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_usub_overflow:
    case Builtin::BI__builtin_usubl_overflow:
    case Builtin::BI__builtin_usubll_overflow:
    case Builtin::BI__builtin_ssub_overflow:
    case Builtin::BI__builtin_ssubl_overflow:
    case Builtin::BI__builtin_ssubll_overflow:
      ArithKind = cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_umul_overflow:
    case Builtin::BI__builtin_umull_overflow:
    case Builtin::BI__builtin_umulll_overflow:
    case Builtin::BI__builtin_smul_overflow:
    case Builtin::BI__builtin_smull_overflow:
    case Builtin::BI__builtin_smulll_overflow:
      ArithKind = cir::BinOpOverflowKind::Mul;
      break;
    }

    clang::QualType ResultQTy =
        ResultArg->getType()->castAs<clang::PointerType>()->getPointeeType();
    auto ResultCIRTy = mlir::cast<cir::IntType>(cgm.convertType(ResultQTy));

    auto Loc = getLoc(e->getSourceRange());
    cir::BinOpOverflowOp ArithOp =
        cir::BinOpOverflowOp::create(builder, Loc, ResultCIRTy, ArithKind, X, Y);

    bool isVolatile =
        ResultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(Loc, emitToMemory(ArithOp.getResult(), ResultQTy),
                        ResultPtr, isVolatile);

    return RValue::get(ArithOp.getOverflow());
  }
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(builtinID))
    return emitLibraryCall(*this, fd, e,
                           cgm.getBuiltinLibFunction(fd, builtinID));

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // returnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  cir::TypeEvaluationKind evalKind = getEvaluationKind(e->getType());
  if (evalKind == cir::TEK_Aggregate && returnValue.isNull()) {
    cgm.errorNYI(e->getSourceRange(), "aggregate return value from builtin");
    return getUndefRValue(e->getType());
  }

  // Now see if we can emit a target-specific builtin.
  if (mlir::Value v = emitTargetBuiltinExpr(builtinID, e, returnValue)) {
    switch (evalKind) {
    case cir::TEK_Scalar:
      if (mlir::isa<cir::VoidType>(v.getType()))
        return RValue::get(nullptr);
      return RValue::get(v);
    case cir::TEK_Aggregate:
      cgm.errorNYI(e->getSourceRange(), "aggregate return value from builtin");
      return getUndefRValue(e->getType());
    case cir::TEK_Complex:
      llvm_unreachable("No current target builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  cgm.errorNYI(e->getSourceRange(),
               std::string("unimplemented builtin call: ") +
                   getContext().BuiltinInfo.getName(builtinID));
  return getUndefRValue(e->getType());
}

static mlir::Value emitTargetArchBuiltinExpr(CIRGenFunction *cgf,
                                             unsigned builtinID,
                                             const CallExpr *e,
                                             ReturnValueSlot &returnValue,
                                             llvm::Triple::ArchType arch) {
  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  // Returning nullptr here leads to the builtin being handled in
  // EmitStdParUnsupportedBuiltin.
  if (cgf->getLangOpts().HIPStdPar && cgf->getLangOpts().CUDAIsDevice &&
      arch != cgf->getTarget().getTriple().getArch())
    return {};

  switch (arch) {
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
    // These are actually NYI, but that will be reported by emitBuiltinExpr.
    // At this point, we don't even know that the builtin is target-specific.
    return nullptr;

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return cgf->emitX86BuiltinExpr(builtinID, e);

  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
  case llvm::Triple::systemz:
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
  case llvm::Triple::hexagon:
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    // These are actually NYI, but that will be reported by emitBuiltinExpr.
    // At this point, we don't even know that the builtin is target-specific.
    return {};
  default:
    return {};
  }
}

mlir::Value
CIRGenFunction::emitTargetBuiltinExpr(unsigned builtinID, const CallExpr *e,
                                      ReturnValueSlot &returnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(builtinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return emitTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(builtinID), e,
        returnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return emitTargetArchBuiltinExpr(this, builtinID, e, returnValue,
                                   getTarget().getTriple().getArch());
}

/// Given a builtin id for a function like "__builtin_fabsf", return a Function*
/// for "fabsf".
cir::FuncOp CIRGenModule::getBuiltinLibFunction(const FunctionDecl *fd,
                                                unsigned builtinID) {
  assert(astContext.BuiltinInfo.isLibFunction(builtinID));

  // Get the name, skip over the __builtin_ prefix (if necessary). We may have
  // to build this up so provide a small stack buffer to handle the vast
  // majority of names.
  llvm::SmallString<64> name;

  assert(!cir::MissingFeatures::asmLabelAttr());
  name = astContext.BuiltinInfo.getName(builtinID).substr(10);

  GlobalDecl d(fd);
  mlir::Type type = convertType(fd->getType());
  return getOrCreateCIRFunction(name, type, d, /*forVTable=*/false);
}

mlir::Value CIRGenFunction::emitCheckedArgForAssume(const Expr *e) {
  mlir::Value argValue = evaluateExprAsBool(e);
  if (!sanOpts.has(SanitizerKind::Builtin))
    return argValue;

  assert(!cir::MissingFeatures::sanitizers());
  cgm.errorNYI(e->getSourceRange(),
               "emitCheckedArgForAssume: sanitizers are NYI");
  return {};
}

void CIRGenFunction::emitVAStart(mlir::Value vaList, mlir::Value count) {
  // LLVM codegen casts to *i8, no real gain on doing this for CIRGen this
  // early, defer to LLVM lowering.
  cir::VAStartOp::create(builder, vaList.getLoc(), vaList, count);
}

void CIRGenFunction::emitVAEnd(mlir::Value vaList) {
  cir::VAEndOp::create(builder, vaList.getLoc(), vaList);
}

// FIXME(cir): This completely abstracts away the ABI with a generic CIR Op. By
// default this lowers to llvm.va_arg which is incomplete and not ABI-compliant
// on most targets so cir.va_arg will need some ABI handling in LoweringPrepare
mlir::Value CIRGenFunction::emitVAArg(VAArgExpr *ve) {
  assert(!cir::MissingFeatures::msabi());
  assert(!cir::MissingFeatures::vlas());
  mlir::Location loc = cgm.getLoc(ve->getExprLoc());
  mlir::Type type = convertType(ve->getType());
  mlir::Value vaList = emitVAListRef(ve->getSubExpr()).getPointer();
  return cir::VAArgOp::create(builder, loc, type, vaList);
}
