//===--- SemaChecking.cpp - Extra Semantic Checking -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements extra semantic analysis beyond what is enforced
//  by the C type system.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Analysis/Analyses/FormatString.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/ConvertUTF.h"
#include <limits>
using namespace clang;
using namespace sema;

SourceLocation Sema::getLocationOfStringLiteralByte(const StringLiteral *SL,
                                                    unsigned ByteNo) const {
  return SL->getLocationOfByte(ByteNo, PP.getSourceManager(),
                               PP.getLangOpts(), PP.getTargetInfo());
}

/// Checks that a call expression's argument count is the desired number.
/// This is useful when doing custom type-checking.  Returns true on error.
static bool checkArgCount(Sema &S, CallExpr *call, unsigned desiredArgCount) {
  unsigned argCount = call->getNumArgs();
  if (argCount == desiredArgCount) return false;

  if (argCount < desiredArgCount)
    return S.Diag(call->getLocEnd(), diag::err_typecheck_call_too_few_args)
        << 0 /*function call*/ << desiredArgCount << argCount
        << call->getSourceRange();

  // Highlight all the excess arguments.
  SourceRange range(call->getArg(desiredArgCount)->getLocStart(),
                    call->getArg(argCount - 1)->getLocEnd());
    
  return S.Diag(range.getBegin(), diag::err_typecheck_call_too_many_args)
    << 0 /*function call*/ << desiredArgCount << argCount
    << call->getArg(1)->getSourceRange();
}

/// Check that the first argument to __builtin_annotation is an integer
/// and the second argument is a non-wide string literal.
static bool SemaBuiltinAnnotation(Sema &S, CallExpr *TheCall) {
  if (checkArgCount(S, TheCall, 2))
    return true;

  // First argument should be an integer.
  Expr *ValArg = TheCall->getArg(0);
  QualType Ty = ValArg->getType();
  if (!Ty->isIntegerType()) {
    S.Diag(ValArg->getLocStart(), diag::err_builtin_annotation_first_arg)
      << ValArg->getSourceRange();
    return true;
  }

  // Second argument should be a constant string.
  Expr *StrArg = TheCall->getArg(1)->IgnoreParenCasts();
  StringLiteral *Literal = dyn_cast<StringLiteral>(StrArg);
  if (!Literal || !Literal->isAscii()) {
    S.Diag(StrArg->getLocStart(), diag::err_builtin_annotation_second_arg)
      << StrArg->getSourceRange();
    return true;
  }

  TheCall->setType(Ty);
  return false;
}

ExprResult
Sema::CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall) {
  ExprResult TheCallResult(Owned(TheCall));

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  Context.GetBuiltinType(BuiltinID, Error, &ICEArguments);
  if (Error != ASTContext::GE_None)
    ICEArguments = 0;  // Don't diagnose previously diagnosed errors.
  
  // If any arguments are required to be ICE's, check and diagnose.
  for (unsigned ArgNo = 0; ICEArguments != 0; ++ArgNo) {
    // Skip arguments not required to be ICE's.
    if ((ICEArguments & (1 << ArgNo)) == 0) continue;
    
    llvm::APSInt Result;
    if (SemaBuiltinConstantArg(TheCall, ArgNo, Result))
      return true;
    ICEArguments &= ~(1 << ArgNo);
  }
  
  switch (BuiltinID) {
  case Builtin::BI__builtin___CFStringMakeConstantString:
    assert(TheCall->getNumArgs() == 1 &&
           "Wrong # arguments to builtin CFStringMakeConstantString");
    if (CheckObjCString(TheCall->getArg(0)))
      return ExprError();
    break;
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
    if (SemaBuiltinVAStart(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered:
    if (SemaBuiltinUnorderedCompare(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_fpclassify:
    if (SemaBuiltinFPClassification(TheCall, 6))
      return ExprError();
    break;
  case Builtin::BI__builtin_isfinite:
  case Builtin::BI__builtin_isinf:
  case Builtin::BI__builtin_isinf_sign:
  case Builtin::BI__builtin_isnan:
  case Builtin::BI__builtin_isnormal:
    if (SemaBuiltinFPClassification(TheCall, 1))
      return ExprError();
    break;
  case Builtin::BI__builtin_shufflevector:
    return SemaBuiltinShuffleVector(TheCall);
    // TheCall will be freed by the smart pointer here, but that's fine, since
    // SemaBuiltinShuffleVector guts it, but then doesn't release it.
  case Builtin::BI__builtin_prefetch:
    if (SemaBuiltinPrefetch(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_object_size:
    if (SemaBuiltinObjectSize(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_longjmp:
    if (SemaBuiltinLongjmp(TheCall))
      return ExprError();
    break;

  case Builtin::BI__builtin_classify_type:
    if (checkArgCount(*this, TheCall, 1)) return true;
    TheCall->setType(Context.IntTy);
    break;
  case Builtin::BI__builtin_constant_p:
    if (checkArgCount(*this, TheCall, 1)) return true;
    TheCall->setType(Context.IntTy);
    break;
  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
  case Builtin::BI__sync_xor_and_fetch:
  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
  case Builtin::BI__sync_lock_test_and_set:
  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
  case Builtin::BI__sync_lock_release:
  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16:
  case Builtin::BI__sync_swap:
  case Builtin::BI__sync_swap_1:
  case Builtin::BI__sync_swap_2:
  case Builtin::BI__sync_swap_4:
  case Builtin::BI__sync_swap_8:
  case Builtin::BI__sync_swap_16:
    return SemaBuiltinAtomicOverloaded(TheCallResult);
#define BUILTIN(ID, TYPE, ATTRS)
#define ATOMIC_BUILTIN(ID, TYPE, ATTRS) \
  case Builtin::BI##ID: \
    return SemaAtomicOpsOverloaded(TheCallResult, AtomicExpr::AO##ID);
#include "clang/Basic/Builtins.def"
  case Builtin::BI__builtin_annotation:
    if (SemaBuiltinAnnotation(*this, TheCall))
      return ExprError();
    break;
  }
  
  // Since the target specific builtins for each arch overlap, only check those
  // of the arch we are compiling for.
  if (BuiltinID >= Builtin::FirstTSBuiltin) {
    switch (Context.getTargetInfo().getTriple().getArch()) {
      case llvm::Triple::arm:
      case llvm::Triple::thumb:
        if (CheckARMBuiltinFunctionCall(BuiltinID, TheCall))
          return ExprError();
        break;
      case llvm::Triple::mips:
      case llvm::Triple::mipsel:
      case llvm::Triple::mips64:
      case llvm::Triple::mips64el:
        if (CheckMipsBuiltinFunctionCall(BuiltinID, TheCall))
          return ExprError();
        break;
      default:
        break;
    }
  }

  return TheCallResult;
}

// Get the valid immediate range for the specified NEON type code.
static unsigned RFT(unsigned t, bool shift = false) {
  NeonTypeFlags Type(t);
  int IsQuad = Type.isQuad();
  switch (Type.getEltType()) {
  case NeonTypeFlags::Int8:
  case NeonTypeFlags::Poly8:
    return shift ? 7 : (8 << IsQuad) - 1;
  case NeonTypeFlags::Int16:
  case NeonTypeFlags::Poly16:
    return shift ? 15 : (4 << IsQuad) - 1;
  case NeonTypeFlags::Int32:
    return shift ? 31 : (2 << IsQuad) - 1;
  case NeonTypeFlags::Int64:
    return shift ? 63 : (1 << IsQuad) - 1;
  case NeonTypeFlags::Float16:
    assert(!shift && "cannot shift float types!");
    return (4 << IsQuad) - 1;
  case NeonTypeFlags::Float32:
    assert(!shift && "cannot shift float types!");
    return (2 << IsQuad) - 1;
  }
  llvm_unreachable("Invalid NeonTypeFlag!");
}

/// getNeonEltType - Return the QualType corresponding to the elements of
/// the vector type specified by the NeonTypeFlags.  This is used to check
/// the pointer arguments for Neon load/store intrinsics.
static QualType getNeonEltType(NeonTypeFlags Flags, ASTContext &Context) {
  switch (Flags.getEltType()) {
  case NeonTypeFlags::Int8:
    return Flags.isUnsigned() ? Context.UnsignedCharTy : Context.SignedCharTy;
  case NeonTypeFlags::Int16:
    return Flags.isUnsigned() ? Context.UnsignedShortTy : Context.ShortTy;
  case NeonTypeFlags::Int32:
    return Flags.isUnsigned() ? Context.UnsignedIntTy : Context.IntTy;
  case NeonTypeFlags::Int64:
    return Flags.isUnsigned() ? Context.UnsignedLongLongTy : Context.LongLongTy;
  case NeonTypeFlags::Poly8:
    return Context.SignedCharTy;
  case NeonTypeFlags::Poly16:
    return Context.ShortTy;
  case NeonTypeFlags::Float16:
    return Context.UnsignedShortTy;
  case NeonTypeFlags::Float32:
    return Context.FloatTy;
  }
  llvm_unreachable("Invalid NeonTypeFlag!");
}

bool Sema::CheckARMBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall) {
  llvm::APSInt Result;

  uint64_t mask = 0;
  unsigned TV = 0;
  int PtrArgNum = -1;
  bool HasConstPtr = false;
  switch (BuiltinID) {
#define GET_NEON_OVERLOAD_CHECK
#include "clang/Basic/arm_neon.inc"
#undef GET_NEON_OVERLOAD_CHECK
  }
  
  // For NEON intrinsics which are overloaded on vector element type, validate
  // the immediate which specifies which variant to emit.
  unsigned ImmArg = TheCall->getNumArgs()-1;
  if (mask) {
    if (SemaBuiltinConstantArg(TheCall, ImmArg, Result))
      return true;
    
    TV = Result.getLimitedValue(64);
    if ((TV > 63) || (mask & (1ULL << TV)) == 0)
      return Diag(TheCall->getLocStart(), diag::err_invalid_neon_type_code)
        << TheCall->getArg(ImmArg)->getSourceRange();
  }

  if (PtrArgNum >= 0) {
    // Check that pointer arguments have the specified type.
    Expr *Arg = TheCall->getArg(PtrArgNum);
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Arg))
      Arg = ICE->getSubExpr();
    ExprResult RHS = DefaultFunctionArrayLvalueConversion(Arg);
    QualType RHSTy = RHS.get()->getType();
    QualType EltTy = getNeonEltType(NeonTypeFlags(TV), Context);
    if (HasConstPtr)
      EltTy = EltTy.withConst();
    QualType LHSTy = Context.getPointerType(EltTy);
    AssignConvertType ConvTy;
    ConvTy = CheckSingleAssignmentConstraints(LHSTy, RHS);
    if (RHS.isInvalid())
      return true;
    if (DiagnoseAssignmentResult(ConvTy, Arg->getLocStart(), LHSTy, RHSTy,
                                 RHS.get(), AA_Assigning))
      return true;
  }
  
  // For NEON intrinsics which take an immediate value as part of the 
  // instruction, range check them here.
  unsigned i = 0, l = 0, u = 0;
  switch (BuiltinID) {
  default: return false;
  case ARM::BI__builtin_arm_ssat: i = 1; l = 1; u = 31; break;
  case ARM::BI__builtin_arm_usat: i = 1; u = 31; break;
  case ARM::BI__builtin_arm_vcvtr_f:
  case ARM::BI__builtin_arm_vcvtr_d: i = 1; u = 1; break;
#define GET_NEON_IMMEDIATE_CHECK
#include "clang/Basic/arm_neon.inc"
#undef GET_NEON_IMMEDIATE_CHECK
  };

  // We can't check the value of a dependent argument.
  if (TheCall->getArg(i)->isTypeDependent() ||
      TheCall->getArg(i)->isValueDependent())
    return false;

  // Check that the immediate argument is actually a constant.
  if (SemaBuiltinConstantArg(TheCall, i, Result))
    return true;

  // Range check against the upper/lower values for this isntruction.
  unsigned Val = Result.getZExtValue();
  if (Val < l || Val > (u + l))
    return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
      << l << u+l << TheCall->getArg(i)->getSourceRange();

  // FIXME: VFP Intrinsics should error if VFP not present.
  return false;
}

bool Sema::CheckMipsBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall) {
  unsigned i = 0, l = 0, u = 0;
  switch (BuiltinID) {
  default: return false;
  case Mips::BI__builtin_mips_wrdsp: i = 1; l = 0; u = 63; break;
  case Mips::BI__builtin_mips_rddsp: i = 0; l = 0; u = 63; break;
  case Mips::BI__builtin_mips_append: i = 2; l = 0; u = 31; break;
  case Mips::BI__builtin_mips_balign: i = 2; l = 0; u = 3; break;
  case Mips::BI__builtin_mips_precr_sra_ph_w: i = 2; l = 0; u = 31; break;
  case Mips::BI__builtin_mips_precr_sra_r_ph_w: i = 2; l = 0; u = 31; break;
  case Mips::BI__builtin_mips_prepend: i = 2; l = 0; u = 31; break;
  };

  // We can't check the value of a dependent argument.
  if (TheCall->getArg(i)->isTypeDependent() ||
      TheCall->getArg(i)->isValueDependent())
    return false;

  // Check that the immediate argument is actually a constant.
  llvm::APSInt Result;
  if (SemaBuiltinConstantArg(TheCall, i, Result))
    return true;

  // Range check against the upper/lower values for this instruction.
  unsigned Val = Result.getZExtValue();
  if (Val < l || Val > u)
    return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
      << l << u << TheCall->getArg(i)->getSourceRange();

  return false;
}

/// Given a FunctionDecl's FormatAttr, attempts to populate the FomatStringInfo
/// parameter with the FormatAttr's correct format_idx and firstDataArg.
/// Returns true when the format fits the function and the FormatStringInfo has
/// been populated.
bool Sema::getFormatStringInfo(const FormatAttr *Format, bool IsCXXMember,
                               FormatStringInfo *FSI) {
  FSI->HasVAListArg = Format->getFirstArg() == 0;
  FSI->FormatIdx = Format->getFormatIdx() - 1;
  FSI->FirstDataArg = FSI->HasVAListArg ? 0 : Format->getFirstArg() - 1;

  // The way the format attribute works in GCC, the implicit this argument
  // of member functions is counted. However, it doesn't appear in our own
  // lists, so decrement format_idx in that case.
  if (IsCXXMember) {
    if(FSI->FormatIdx == 0)
      return false;
    --FSI->FormatIdx;
    if (FSI->FirstDataArg != 0)
      --FSI->FirstDataArg;
  }
  return true;
}

/// Handles the checks for format strings, non-POD arguments to vararg
/// functions, and NULL arguments passed to non-NULL parameters.
void Sema::checkCall(NamedDecl *FDecl, Expr **Args,
                     unsigned NumArgs,
                     unsigned NumProtoArgs,
                     bool IsMemberFunction,
                     SourceLocation Loc,
                     SourceRange Range,
                     VariadicCallType CallType) {
  if (CurContext->isDependentContext())
    return;

  // Printf and scanf checking.
  bool HandledFormatString = false;
  for (specific_attr_iterator<FormatAttr>
         I = FDecl->specific_attr_begin<FormatAttr>(),
         E = FDecl->specific_attr_end<FormatAttr>(); I != E ; ++I)
    if (CheckFormatArguments(*I, Args, NumArgs, IsMemberFunction, CallType,
                             Loc, Range))
        HandledFormatString = true;

  // Refuse POD arguments that weren't caught by the format string
  // checks above.
  if (!HandledFormatString && CallType != VariadicDoesNotApply)
    for (unsigned ArgIdx = NumProtoArgs; ArgIdx < NumArgs; ++ArgIdx) {
      // Args[ArgIdx] can be null in malformed code.
      if (Expr *Arg = Args[ArgIdx])
        variadicArgumentPODCheck(Arg, CallType);
    }

  for (specific_attr_iterator<NonNullAttr>
         I = FDecl->specific_attr_begin<NonNullAttr>(),
         E = FDecl->specific_attr_end<NonNullAttr>(); I != E; ++I)
    CheckNonNullArguments(*I, Args, Loc);

  // Type safety checking.
  for (specific_attr_iterator<ArgumentWithTypeTagAttr>
         i = FDecl->specific_attr_begin<ArgumentWithTypeTagAttr>(),
         e = FDecl->specific_attr_end<ArgumentWithTypeTagAttr>(); i != e; ++i) {
    CheckArgumentWithTypeTag(*i, Args);
  }
}

/// CheckConstructorCall - Check a constructor call for correctness and safety
/// properties not enforced by the C type system.
void Sema::CheckConstructorCall(FunctionDecl *FDecl, Expr **Args,
                                unsigned NumArgs,
                                const FunctionProtoType *Proto,
                                SourceLocation Loc) {
  VariadicCallType CallType =
    Proto->isVariadic() ? VariadicConstructor : VariadicDoesNotApply;
  checkCall(FDecl, Args, NumArgs, Proto->getNumArgs(),
            /*IsMemberFunction=*/true, Loc, SourceRange(), CallType);
}

/// CheckFunctionCall - Check a direct function call for various correctness
/// and safety properties not strictly enforced by the C type system.
bool Sema::CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall,
                             const FunctionProtoType *Proto) {
  bool IsMemberOperatorCall = isa<CXXOperatorCallExpr>(TheCall) &&
                              isa<CXXMethodDecl>(FDecl);
  bool IsMemberFunction = isa<CXXMemberCallExpr>(TheCall) ||
                          IsMemberOperatorCall;
  VariadicCallType CallType = getVariadicCallType(FDecl, Proto,
                                                  TheCall->getCallee());
  unsigned NumProtoArgs = Proto ? Proto->getNumArgs() : 0;
  Expr** Args = TheCall->getArgs();
  unsigned NumArgs = TheCall->getNumArgs();
  if (IsMemberOperatorCall) {
    // If this is a call to a member operator, hide the first argument
    // from checkCall.
    // FIXME: Our choice of AST representation here is less than ideal.
    ++Args;
    --NumArgs;
  }
  checkCall(FDecl, Args, NumArgs, NumProtoArgs,
            IsMemberFunction, TheCall->getRParenLoc(),
            TheCall->getCallee()->getSourceRange(), CallType);

  IdentifierInfo *FnInfo = FDecl->getIdentifier();
  // None of the checks below are needed for functions that don't have
  // simple names (e.g., C++ conversion functions).
  if (!FnInfo)
    return false;

  unsigned CMId = FDecl->getMemoryFunctionKind();
  if (CMId == 0)
    return false;

  // Handle memory setting and copying functions.
  if (CMId == Builtin::BIstrlcpy || CMId == Builtin::BIstrlcat)
    CheckStrlcpycatArguments(TheCall, FnInfo);
  else if (CMId == Builtin::BIstrncat)
    CheckStrncatArguments(TheCall, FnInfo);
  else
    CheckMemaccessArguments(TheCall, CMId, FnInfo);

  return false;
}

bool Sema::CheckObjCMethodCall(ObjCMethodDecl *Method, SourceLocation lbrac, 
                               Expr **Args, unsigned NumArgs) {
  VariadicCallType CallType =
      Method->isVariadic() ? VariadicMethod : VariadicDoesNotApply;

  checkCall(Method, Args, NumArgs, Method->param_size(),
            /*IsMemberFunction=*/false,
            lbrac, Method->getSourceRange(), CallType);

  return false;
}

bool Sema::CheckBlockCall(NamedDecl *NDecl, CallExpr *TheCall,
                          const FunctionProtoType *Proto) {
  const VarDecl *V = dyn_cast<VarDecl>(NDecl);
  if (!V)
    return false;

  QualType Ty = V->getType();
  if (!Ty->isBlockPointerType())
    return false;

  VariadicCallType CallType = 
      Proto && Proto->isVariadic() ? VariadicBlock : VariadicDoesNotApply ;
  unsigned NumProtoArgs = Proto ? Proto->getNumArgs() : 0;

  checkCall(NDecl, TheCall->getArgs(), TheCall->getNumArgs(),
            NumProtoArgs, /*IsMemberFunction=*/false,
            TheCall->getRParenLoc(),
            TheCall->getCallee()->getSourceRange(), CallType);
  
  return false;
}

ExprResult Sema::SemaAtomicOpsOverloaded(ExprResult TheCallResult,
                                         AtomicExpr::AtomicOp Op) {
  CallExpr *TheCall = cast<CallExpr>(TheCallResult.get());
  DeclRefExpr *DRE =cast<DeclRefExpr>(TheCall->getCallee()->IgnoreParenCasts());

  // All these operations take one of the following forms:
  enum {
    // C    __c11_atomic_init(A *, C)
    Init,
    // C    __c11_atomic_load(A *, int)
    Load,
    // void __atomic_load(A *, CP, int)
    Copy,
    // C    __c11_atomic_add(A *, M, int)
    Arithmetic,
    // C    __atomic_exchange_n(A *, CP, int)
    Xchg,
    // void __atomic_exchange(A *, C *, CP, int)
    GNUXchg,
    // bool __c11_atomic_compare_exchange_strong(A *, C *, CP, int, int)
    C11CmpXchg,
    // bool __atomic_compare_exchange(A *, C *, CP, bool, int, int)
    GNUCmpXchg
  } Form = Init;
  const unsigned NumArgs[] = { 2, 2, 3, 3, 3, 4, 5, 6 };
  const unsigned NumVals[] = { 1, 0, 1, 1, 1, 2, 2, 3 };
  // where:
  //   C is an appropriate type,
  //   A is volatile _Atomic(C) for __c11 builtins and is C for GNU builtins,
  //   CP is C for __c11 builtins and GNU _n builtins and is C * otherwise,
  //   M is C if C is an integer, and ptrdiff_t if C is a pointer, and
  //   the int parameters are for orderings.

  assert(AtomicExpr::AO__c11_atomic_init == 0 &&
         AtomicExpr::AO__c11_atomic_fetch_xor + 1 == AtomicExpr::AO__atomic_load
         && "need to update code for modified C11 atomics");
  bool IsC11 = Op >= AtomicExpr::AO__c11_atomic_init &&
               Op <= AtomicExpr::AO__c11_atomic_fetch_xor;
  bool IsN = Op == AtomicExpr::AO__atomic_load_n ||
             Op == AtomicExpr::AO__atomic_store_n ||
             Op == AtomicExpr::AO__atomic_exchange_n ||
             Op == AtomicExpr::AO__atomic_compare_exchange_n;
  bool IsAddSub = false;

  switch (Op) {
  case AtomicExpr::AO__c11_atomic_init:
    Form = Init;
    break;

  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
    Form = Load;
    break;

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__atomic_store_n:
    Form = Copy;
    break;

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__atomic_sub_fetch:
    IsAddSub = true;
    // Fall through.
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__atomic_nand_fetch:
    Form = Arithmetic;
    break;

  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__atomic_exchange_n:
    Form = Xchg;
    break;

  case AtomicExpr::AO__atomic_exchange:
    Form = GNUXchg;
    break;

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    Form = C11CmpXchg;
    break;

  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
    Form = GNUCmpXchg;
    break;
  }

  // Check we have the right number of arguments.
  if (TheCall->getNumArgs() < NumArgs[Form]) {
    Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
      << 0 << NumArgs[Form] << TheCall->getNumArgs()
      << TheCall->getCallee()->getSourceRange();
    return ExprError();
  } else if (TheCall->getNumArgs() > NumArgs[Form]) {
    Diag(TheCall->getArg(NumArgs[Form])->getLocStart(),
         diag::err_typecheck_call_too_many_args)
      << 0 << NumArgs[Form] << TheCall->getNumArgs()
      << TheCall->getCallee()->getSourceRange();
    return ExprError();
  }

  // Inspect the first argument of the atomic operation.
  Expr *Ptr = TheCall->getArg(0);
  Ptr = DefaultFunctionArrayLvalueConversion(Ptr).get();
  const PointerType *pointerType = Ptr->getType()->getAs<PointerType>();
  if (!pointerType) {
    Diag(DRE->getLocStart(), diag::err_atomic_builtin_must_be_pointer)
      << Ptr->getType() << Ptr->getSourceRange();
    return ExprError();
  }

  // For a __c11 builtin, this should be a pointer to an _Atomic type.
  QualType AtomTy = pointerType->getPointeeType(); // 'A'
  QualType ValType = AtomTy; // 'C'
  if (IsC11) {
    if (!AtomTy->isAtomicType()) {
      Diag(DRE->getLocStart(), diag::err_atomic_op_needs_atomic)
        << Ptr->getType() << Ptr->getSourceRange();
      return ExprError();
    }
    if (AtomTy.isConstQualified()) {
      Diag(DRE->getLocStart(), diag::err_atomic_op_needs_non_const_atomic)
        << Ptr->getType() << Ptr->getSourceRange();
      return ExprError();
    }
    ValType = AtomTy->getAs<AtomicType>()->getValueType();
  }

  // For an arithmetic operation, the implied arithmetic must be well-formed.
  if (Form == Arithmetic) {
    // gcc does not enforce these rules for GNU atomics, but we do so for sanity.
    if (IsAddSub && !ValType->isIntegerType() && !ValType->isPointerType()) {
      Diag(DRE->getLocStart(), diag::err_atomic_op_needs_atomic_int_or_ptr)
        << IsC11 << Ptr->getType() << Ptr->getSourceRange();
      return ExprError();
    }
    if (!IsAddSub && !ValType->isIntegerType()) {
      Diag(DRE->getLocStart(), diag::err_atomic_op_bitwise_needs_atomic_int)
        << IsC11 << Ptr->getType() << Ptr->getSourceRange();
      return ExprError();
    }
  } else if (IsN && !ValType->isIntegerType() && !ValType->isPointerType()) {
    // For __atomic_*_n operations, the value type must be a scalar integral or
    // pointer type which is 1, 2, 4, 8 or 16 bytes in length.
    Diag(DRE->getLocStart(), diag::err_atomic_op_needs_atomic_int_or_ptr)
      << IsC11 << Ptr->getType() << Ptr->getSourceRange();
    return ExprError();
  }

  if (!IsC11 && !AtomTy.isTriviallyCopyableType(Context)) {
    // For GNU atomics, require a trivially-copyable type. This is not part of
    // the GNU atomics specification, but we enforce it for sanity.
    Diag(DRE->getLocStart(), diag::err_atomic_op_needs_trivial_copy)
      << Ptr->getType() << Ptr->getSourceRange();
    return ExprError();
  }

  // FIXME: For any builtin other than a load, the ValType must not be
  // const-qualified.

  switch (ValType.getObjCLifetime()) {
  case Qualifiers::OCL_None:
  case Qualifiers::OCL_ExplicitNone:
    // okay
    break;

  case Qualifiers::OCL_Weak:
  case Qualifiers::OCL_Strong:
  case Qualifiers::OCL_Autoreleasing:
    // FIXME: Can this happen? By this point, ValType should be known
    // to be trivially copyable.
    Diag(DRE->getLocStart(), diag::err_arc_atomic_ownership)
      << ValType << Ptr->getSourceRange();
    return ExprError();
  }

  QualType ResultType = ValType;
  if (Form == Copy || Form == GNUXchg || Form == Init)
    ResultType = Context.VoidTy;
  else if (Form == C11CmpXchg || Form == GNUCmpXchg)
    ResultType = Context.BoolTy;

  // The type of a parameter passed 'by value'. In the GNU atomics, such
  // arguments are actually passed as pointers.
  QualType ByValType = ValType; // 'CP'
  if (!IsC11 && !IsN)
    ByValType = Ptr->getType();

  // The first argument --- the pointer --- has a fixed type; we
  // deduce the types of the rest of the arguments accordingly.  Walk
  // the remaining arguments, converting them to the deduced value type.
  for (unsigned i = 1; i != NumArgs[Form]; ++i) {
    QualType Ty;
    if (i < NumVals[Form] + 1) {
      switch (i) {
      case 1:
        // The second argument is the non-atomic operand. For arithmetic, this
        // is always passed by value, and for a compare_exchange it is always
        // passed by address. For the rest, GNU uses by-address and C11 uses
        // by-value.
        assert(Form != Load);
        if (Form == Init || (Form == Arithmetic && ValType->isIntegerType()))
          Ty = ValType;
        else if (Form == Copy || Form == Xchg)
          Ty = ByValType;
        else if (Form == Arithmetic)
          Ty = Context.getPointerDiffType();
        else
          Ty = Context.getPointerType(ValType.getUnqualifiedType());
        break;
      case 2:
        // The third argument to compare_exchange / GNU exchange is a
        // (pointer to a) desired value.
        Ty = ByValType;
        break;
      case 3:
        // The fourth argument to GNU compare_exchange is a 'weak' flag.
        Ty = Context.BoolTy;
        break;
      }
    } else {
      // The order(s) are always converted to int.
      Ty = Context.IntTy;
    }

    InitializedEntity Entity =
        InitializedEntity::InitializeParameter(Context, Ty, false);
    ExprResult Arg = TheCall->getArg(i);
    Arg = PerformCopyInitialization(Entity, SourceLocation(), Arg);
    if (Arg.isInvalid())
      return true;
    TheCall->setArg(i, Arg.get());
  }

  // Permute the arguments into a 'consistent' order.
  SmallVector<Expr*, 5> SubExprs;
  SubExprs.push_back(Ptr);
  switch (Form) {
  case Init:
    // Note, AtomicExpr::getVal1() has a special case for this atomic.
    SubExprs.push_back(TheCall->getArg(1)); // Val1
    break;
  case Load:
    SubExprs.push_back(TheCall->getArg(1)); // Order
    break;
  case Copy:
  case Arithmetic:
  case Xchg:
    SubExprs.push_back(TheCall->getArg(2)); // Order
    SubExprs.push_back(TheCall->getArg(1)); // Val1
    break;
  case GNUXchg:
    // Note, AtomicExpr::getVal2() has a special case for this atomic.
    SubExprs.push_back(TheCall->getArg(3)); // Order
    SubExprs.push_back(TheCall->getArg(1)); // Val1
    SubExprs.push_back(TheCall->getArg(2)); // Val2
    break;
  case C11CmpXchg:
    SubExprs.push_back(TheCall->getArg(3)); // Order
    SubExprs.push_back(TheCall->getArg(1)); // Val1
    SubExprs.push_back(TheCall->getArg(4)); // OrderFail
    SubExprs.push_back(TheCall->getArg(2)); // Val2
    break;
  case GNUCmpXchg:
    SubExprs.push_back(TheCall->getArg(4)); // Order
    SubExprs.push_back(TheCall->getArg(1)); // Val1
    SubExprs.push_back(TheCall->getArg(5)); // OrderFail
    SubExprs.push_back(TheCall->getArg(2)); // Val2
    SubExprs.push_back(TheCall->getArg(3)); // Weak
    break;
  }

  return Owned(new (Context) AtomicExpr(TheCall->getCallee()->getLocStart(),
                                        SubExprs, ResultType, Op,
                                        TheCall->getRParenLoc()));
}


/// checkBuiltinArgument - Given a call to a builtin function, perform
/// normal type-checking on the given argument, updating the call in
/// place.  This is useful when a builtin function requires custom
/// type-checking for some of its arguments but not necessarily all of
/// them.
///
/// Returns true on error.
static bool checkBuiltinArgument(Sema &S, CallExpr *E, unsigned ArgIndex) {
  FunctionDecl *Fn = E->getDirectCallee();
  assert(Fn && "builtin call without direct callee!");

  ParmVarDecl *Param = Fn->getParamDecl(ArgIndex);
  InitializedEntity Entity =
    InitializedEntity::InitializeParameter(S.Context, Param);

  ExprResult Arg = E->getArg(0);
  Arg = S.PerformCopyInitialization(Entity, SourceLocation(), Arg);
  if (Arg.isInvalid())
    return true;

  E->setArg(ArgIndex, Arg.take());
  return false;
}

/// SemaBuiltinAtomicOverloaded - We have a call to a function like
/// __sync_fetch_and_add, which is an overloaded function based on the pointer
/// type of its first argument.  The main ActOnCallExpr routines have already
/// promoted the types of arguments because all of these calls are prototyped as
/// void(...).
///
/// This function goes through and does final semantic checking for these
/// builtins,
ExprResult
Sema::SemaBuiltinAtomicOverloaded(ExprResult TheCallResult) {
  CallExpr *TheCall = (CallExpr *)TheCallResult.get();
  DeclRefExpr *DRE =cast<DeclRefExpr>(TheCall->getCallee()->IgnoreParenCasts());
  FunctionDecl *FDecl = cast<FunctionDecl>(DRE->getDecl());

  // Ensure that we have at least one argument to do type inference from.
  if (TheCall->getNumArgs() < 1) {
    Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args_at_least)
      << 0 << 1 << TheCall->getNumArgs()
      << TheCall->getCallee()->getSourceRange();
    return ExprError();
  }

  // Inspect the first argument of the atomic builtin.  This should always be
  // a pointer type, whose element is an integral scalar or pointer type.
  // Because it is a pointer type, we don't have to worry about any implicit
  // casts here.
  // FIXME: We don't allow floating point scalars as input.
  Expr *FirstArg = TheCall->getArg(0);
  ExprResult FirstArgResult = DefaultFunctionArrayLvalueConversion(FirstArg);
  if (FirstArgResult.isInvalid())
    return ExprError();
  FirstArg = FirstArgResult.take();
  TheCall->setArg(0, FirstArg);

  const PointerType *pointerType = FirstArg->getType()->getAs<PointerType>();
  if (!pointerType) {
    Diag(DRE->getLocStart(), diag::err_atomic_builtin_must_be_pointer)
      << FirstArg->getType() << FirstArg->getSourceRange();
    return ExprError();
  }

  QualType ValType = pointerType->getPointeeType();
  if (!ValType->isIntegerType() && !ValType->isAnyPointerType() &&
      !ValType->isBlockPointerType()) {
    Diag(DRE->getLocStart(), diag::err_atomic_builtin_must_be_pointer_intptr)
      << FirstArg->getType() << FirstArg->getSourceRange();
    return ExprError();
  }

  switch (ValType.getObjCLifetime()) {
  case Qualifiers::OCL_None:
  case Qualifiers::OCL_ExplicitNone:
    // okay
    break;

  case Qualifiers::OCL_Weak:
  case Qualifiers::OCL_Strong:
  case Qualifiers::OCL_Autoreleasing:
    Diag(DRE->getLocStart(), diag::err_arc_atomic_ownership)
      << ValType << FirstArg->getSourceRange();
    return ExprError();
  }

  // Strip any qualifiers off ValType.
  ValType = ValType.getUnqualifiedType();

  // The majority of builtins return a value, but a few have special return
  // types, so allow them to override appropriately below.
  QualType ResultType = ValType;

  // We need to figure out which concrete builtin this maps onto.  For example,
  // __sync_fetch_and_add with a 2 byte object turns into
  // __sync_fetch_and_add_2.
#define BUILTIN_ROW(x) \
  { Builtin::BI##x##_1, Builtin::BI##x##_2, Builtin::BI##x##_4, \
    Builtin::BI##x##_8, Builtin::BI##x##_16 }

  static const unsigned BuiltinIndices[][5] = {
    BUILTIN_ROW(__sync_fetch_and_add),
    BUILTIN_ROW(__sync_fetch_and_sub),
    BUILTIN_ROW(__sync_fetch_and_or),
    BUILTIN_ROW(__sync_fetch_and_and),
    BUILTIN_ROW(__sync_fetch_and_xor),

    BUILTIN_ROW(__sync_add_and_fetch),
    BUILTIN_ROW(__sync_sub_and_fetch),
    BUILTIN_ROW(__sync_and_and_fetch),
    BUILTIN_ROW(__sync_or_and_fetch),
    BUILTIN_ROW(__sync_xor_and_fetch),

    BUILTIN_ROW(__sync_val_compare_and_swap),
    BUILTIN_ROW(__sync_bool_compare_and_swap),
    BUILTIN_ROW(__sync_lock_test_and_set),
    BUILTIN_ROW(__sync_lock_release),
    BUILTIN_ROW(__sync_swap)
  };
#undef BUILTIN_ROW

  // Determine the index of the size.
  unsigned SizeIndex;
  switch (Context.getTypeSizeInChars(ValType).getQuantity()) {
  case 1: SizeIndex = 0; break;
  case 2: SizeIndex = 1; break;
  case 4: SizeIndex = 2; break;
  case 8: SizeIndex = 3; break;
  case 16: SizeIndex = 4; break;
  default:
    Diag(DRE->getLocStart(), diag::err_atomic_builtin_pointer_size)
      << FirstArg->getType() << FirstArg->getSourceRange();
    return ExprError();
  }

  // Each of these builtins has one pointer argument, followed by some number of
  // values (0, 1 or 2) followed by a potentially empty varags list of stuff
  // that we ignore.  Find out which row of BuiltinIndices to read from as well
  // as the number of fixed args.
  unsigned BuiltinID = FDecl->getBuiltinID();
  unsigned BuiltinIndex, NumFixed = 1;
  switch (BuiltinID) {
  default: llvm_unreachable("Unknown overloaded atomic builtin!");
  case Builtin::BI__sync_fetch_and_add: 
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16:
    BuiltinIndex = 0; 
    break;
      
  case Builtin::BI__sync_fetch_and_sub: 
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16:
    BuiltinIndex = 1; 
    break;
      
  case Builtin::BI__sync_fetch_and_or:  
  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
    BuiltinIndex = 2; 
    break;
      
  case Builtin::BI__sync_fetch_and_and: 
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
    BuiltinIndex = 3; 
    break;

  case Builtin::BI__sync_fetch_and_xor: 
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
    BuiltinIndex = 4; 
    break;

  case Builtin::BI__sync_add_and_fetch: 
  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
    BuiltinIndex = 5; 
    break;
      
  case Builtin::BI__sync_sub_and_fetch: 
  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
    BuiltinIndex = 6; 
    break;
      
  case Builtin::BI__sync_and_and_fetch: 
  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
    BuiltinIndex = 7; 
    break;
      
  case Builtin::BI__sync_or_and_fetch:  
  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
    BuiltinIndex = 8; 
    break;
      
  case Builtin::BI__sync_xor_and_fetch: 
  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
    BuiltinIndex = 9; 
    break;

  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
    BuiltinIndex = 10;
    NumFixed = 2;
    break;
      
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
    BuiltinIndex = 11;
    NumFixed = 2;
    ResultType = Context.BoolTy;
    break;
      
  case Builtin::BI__sync_lock_test_and_set: 
  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
    BuiltinIndex = 12; 
    break;
      
  case Builtin::BI__sync_lock_release:
  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16:
    BuiltinIndex = 13;
    NumFixed = 0;
    ResultType = Context.VoidTy;
    break;
      
  case Builtin::BI__sync_swap: 
  case Builtin::BI__sync_swap_1:
  case Builtin::BI__sync_swap_2:
  case Builtin::BI__sync_swap_4:
  case Builtin::BI__sync_swap_8:
  case Builtin::BI__sync_swap_16:
    BuiltinIndex = 14; 
    break;
  }

  // Now that we know how many fixed arguments we expect, first check that we
  // have at least that many.
  if (TheCall->getNumArgs() < 1+NumFixed) {
    Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args_at_least)
      << 0 << 1+NumFixed << TheCall->getNumArgs()
      << TheCall->getCallee()->getSourceRange();
    return ExprError();
  }

  // Get the decl for the concrete builtin from this, we can tell what the
  // concrete integer type we should convert to is.
  unsigned NewBuiltinID = BuiltinIndices[BuiltinIndex][SizeIndex];
  const char *NewBuiltinName = Context.BuiltinInfo.GetName(NewBuiltinID);
  FunctionDecl *NewBuiltinDecl;
  if (NewBuiltinID == BuiltinID)
    NewBuiltinDecl = FDecl;
  else {
    // Perform builtin lookup to avoid redeclaring it.
    DeclarationName DN(&Context.Idents.get(NewBuiltinName));
    LookupResult Res(*this, DN, DRE->getLocStart(), LookupOrdinaryName);
    LookupName(Res, TUScope, /*AllowBuiltinCreation=*/true);
    assert(Res.getFoundDecl());
    NewBuiltinDecl = dyn_cast<FunctionDecl>(Res.getFoundDecl());
    if (NewBuiltinDecl == 0)
      return ExprError();
  }

  // The first argument --- the pointer --- has a fixed type; we
  // deduce the types of the rest of the arguments accordingly.  Walk
  // the remaining arguments, converting them to the deduced value type.
  for (unsigned i = 0; i != NumFixed; ++i) {
    ExprResult Arg = TheCall->getArg(i+1);

    // GCC does an implicit conversion to the pointer or integer ValType.  This
    // can fail in some cases (1i -> int**), check for this error case now.
    // Initialize the argument.
    InitializedEntity Entity = InitializedEntity::InitializeParameter(Context,
                                                   ValType, /*consume*/ false);
    Arg = PerformCopyInitialization(Entity, SourceLocation(), Arg);
    if (Arg.isInvalid())
      return ExprError();

    // Okay, we have something that *can* be converted to the right type.  Check
    // to see if there is a potentially weird extension going on here.  This can
    // happen when you do an atomic operation on something like an char* and
    // pass in 42.  The 42 gets converted to char.  This is even more strange
    // for things like 45.123 -> char, etc.
    // FIXME: Do this check.
    TheCall->setArg(i+1, Arg.take());
  }

  ASTContext& Context = this->getASTContext();

  // Create a new DeclRefExpr to refer to the new decl.
  DeclRefExpr* NewDRE = DeclRefExpr::Create(
      Context,
      DRE->getQualifierLoc(),
      SourceLocation(),
      NewBuiltinDecl,
      /*enclosing*/ false,
      DRE->getLocation(),
      Context.BuiltinFnTy,
      DRE->getValueKind());

  // Set the callee in the CallExpr.
  // FIXME: This loses syntactic information.
  QualType CalleePtrTy = Context.getPointerType(NewBuiltinDecl->getType());
  ExprResult PromotedCall = ImpCastExprToType(NewDRE, CalleePtrTy,
                                              CK_BuiltinFnToFnPtr);
  TheCall->setCallee(PromotedCall.take());

  // Change the result type of the call to match the original value type. This
  // is arbitrary, but the codegen for these builtins ins design to handle it
  // gracefully.
  TheCall->setType(ResultType);

  return TheCallResult;
}

/// CheckObjCString - Checks that the argument to the builtin
/// CFString constructor is correct
/// Note: It might also make sense to do the UTF-16 conversion here (would
/// simplify the backend).
bool Sema::CheckObjCString(Expr *Arg) {
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Literal = dyn_cast<StringLiteral>(Arg);

  if (!Literal || !Literal->isAscii()) {
    Diag(Arg->getLocStart(), diag::err_cfstring_literal_not_string_constant)
      << Arg->getSourceRange();
    return true;
  }

  if (Literal->containsNonAsciiOrNull()) {
    StringRef String = Literal->getString();
    unsigned NumBytes = String.size();
    SmallVector<UTF16, 128> ToBuf(NumBytes);
    const UTF8 *FromPtr = (const UTF8 *)String.data();
    UTF16 *ToPtr = &ToBuf[0];
    
    ConversionResult Result = ConvertUTF8toUTF16(&FromPtr, FromPtr + NumBytes,
                                                 &ToPtr, ToPtr + NumBytes,
                                                 strictConversion);
    // Check for conversion failure.
    if (Result != conversionOK)
      Diag(Arg->getLocStart(),
           diag::warn_cfstring_truncated) << Arg->getSourceRange();
  }
  return false;
}

/// SemaBuiltinVAStart - Check the arguments to __builtin_va_start for validity.
/// Emit an error and return true on failure, return false on success.
bool Sema::SemaBuiltinVAStart(CallExpr *TheCall) {
  Expr *Fn = TheCall->getCallee();
  if (TheCall->getNumArgs() > 2) {
    Diag(TheCall->getArg(2)->getLocStart(),
         diag::err_typecheck_call_too_many_args)
      << 0 /*function call*/ << 2 << TheCall->getNumArgs()
      << Fn->getSourceRange()
      << SourceRange(TheCall->getArg(2)->getLocStart(),
                     (*(TheCall->arg_end()-1))->getLocEnd());
    return true;
  }

  if (TheCall->getNumArgs() < 2) {
    return Diag(TheCall->getLocEnd(),
      diag::err_typecheck_call_too_few_args_at_least)
      << 0 /*function call*/ << 2 << TheCall->getNumArgs();
  }

  // Type-check the first argument normally.
  if (checkBuiltinArgument(*this, TheCall, 0))
    return true;

  // Determine whether the current function is variadic or not.
  BlockScopeInfo *CurBlock = getCurBlock();
  bool isVariadic;
  if (CurBlock)
    isVariadic = CurBlock->TheDecl->isVariadic();
  else if (FunctionDecl *FD = getCurFunctionDecl())
    isVariadic = FD->isVariadic();
  else
    isVariadic = getCurMethodDecl()->isVariadic();

  if (!isVariadic) {
    Diag(Fn->getLocStart(), diag::err_va_start_used_in_non_variadic_function);
    return true;
  }

  // Verify that the second argument to the builtin is the last argument of the
  // current function or method.
  bool SecondArgIsLastNamedArgument = false;
  const Expr *Arg = TheCall->getArg(1)->IgnoreParenCasts();

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Arg)) {
    if (const ParmVarDecl *PV = dyn_cast<ParmVarDecl>(DR->getDecl())) {
      // FIXME: This isn't correct for methods (results in bogus warning).
      // Get the last formal in the current function.
      const ParmVarDecl *LastArg;
      if (CurBlock)
        LastArg = *(CurBlock->TheDecl->param_end()-1);
      else if (FunctionDecl *FD = getCurFunctionDecl())
        LastArg = *(FD->param_end()-1);
      else
        LastArg = *(getCurMethodDecl()->param_end()-1);
      SecondArgIsLastNamedArgument = PV == LastArg;
    }
  }

  if (!SecondArgIsLastNamedArgument)
    Diag(TheCall->getArg(1)->getLocStart(),
         diag::warn_second_parameter_of_va_start_not_last_named_argument);
  return false;
}

/// SemaBuiltinUnorderedCompare - Handle functions like __builtin_isgreater and
/// friends.  This is declared to take (...), so we have to check everything.
bool Sema::SemaBuiltinUnorderedCompare(CallExpr *TheCall) {
  if (TheCall->getNumArgs() < 2)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
      << 0 << 2 << TheCall->getNumArgs()/*function call*/;
  if (TheCall->getNumArgs() > 2)
    return Diag(TheCall->getArg(2)->getLocStart(),
                diag::err_typecheck_call_too_many_args)
      << 0 /*function call*/ << 2 << TheCall->getNumArgs()
      << SourceRange(TheCall->getArg(2)->getLocStart(),
                     (*(TheCall->arg_end()-1))->getLocEnd());

  ExprResult OrigArg0 = TheCall->getArg(0);
  ExprResult OrigArg1 = TheCall->getArg(1);

  // Do standard promotions between the two arguments, returning their common
  // type.
  QualType Res = UsualArithmeticConversions(OrigArg0, OrigArg1, false);
  if (OrigArg0.isInvalid() || OrigArg1.isInvalid())
    return true;

  // Make sure any conversions are pushed back into the call; this is
  // type safe since unordered compare builtins are declared as "_Bool
  // foo(...)".
  TheCall->setArg(0, OrigArg0.get());
  TheCall->setArg(1, OrigArg1.get());

  if (OrigArg0.get()->isTypeDependent() || OrigArg1.get()->isTypeDependent())
    return false;

  // If the common type isn't a real floating type, then the arguments were
  // invalid for this operation.
  if (Res.isNull() || !Res->isRealFloatingType())
    return Diag(OrigArg0.get()->getLocStart(),
                diag::err_typecheck_call_invalid_ordered_compare)
      << OrigArg0.get()->getType() << OrigArg1.get()->getType()
      << SourceRange(OrigArg0.get()->getLocStart(), OrigArg1.get()->getLocEnd());

  return false;
}

/// SemaBuiltinSemaBuiltinFPClassification - Handle functions like
/// __builtin_isnan and friends.  This is declared to take (...), so we have
/// to check everything. We expect the last argument to be a floating point
/// value.
bool Sema::SemaBuiltinFPClassification(CallExpr *TheCall, unsigned NumArgs) {
  if (TheCall->getNumArgs() < NumArgs)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
      << 0 << NumArgs << TheCall->getNumArgs()/*function call*/;
  if (TheCall->getNumArgs() > NumArgs)
    return Diag(TheCall->getArg(NumArgs)->getLocStart(),
                diag::err_typecheck_call_too_many_args)
      << 0 /*function call*/ << NumArgs << TheCall->getNumArgs()
      << SourceRange(TheCall->getArg(NumArgs)->getLocStart(),
                     (*(TheCall->arg_end()-1))->getLocEnd());

  Expr *OrigArg = TheCall->getArg(NumArgs-1);

  if (OrigArg->isTypeDependent())
    return false;

  // This operation requires a non-_Complex floating-point number.
  if (!OrigArg->getType()->isRealFloatingType())
    return Diag(OrigArg->getLocStart(),
                diag::err_typecheck_call_invalid_unary_fp)
      << OrigArg->getType() << OrigArg->getSourceRange();

  // If this is an implicit conversion from float -> double, remove it.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(OrigArg)) {
    Expr *CastArg = Cast->getSubExpr();
    if (CastArg->getType()->isSpecificBuiltinType(BuiltinType::Float)) {
      assert(Cast->getType()->isSpecificBuiltinType(BuiltinType::Double) &&
             "promotion from float to double is the only expected cast here");
      Cast->setSubExpr(0);
      TheCall->setArg(NumArgs-1, CastArg);
    }
  }
  
  return false;
}

/// SemaBuiltinShuffleVector - Handle __builtin_shufflevector.
// This is declared to take (...), so we have to check everything.
ExprResult Sema::SemaBuiltinShuffleVector(CallExpr *TheCall) {
  if (TheCall->getNumArgs() < 2)
    return ExprError(Diag(TheCall->getLocEnd(),
                          diag::err_typecheck_call_too_few_args_at_least)
      << 0 /*function call*/ << 2 << TheCall->getNumArgs()
      << TheCall->getSourceRange());

  // Determine which of the following types of shufflevector we're checking:
  // 1) unary, vector mask: (lhs, mask)
  // 2) binary, vector mask: (lhs, rhs, mask)
  // 3) binary, scalar mask: (lhs, rhs, index, ..., index)
  QualType resType = TheCall->getArg(0)->getType();
  unsigned numElements = 0;
  
  if (!TheCall->getArg(0)->isTypeDependent() &&
      !TheCall->getArg(1)->isTypeDependent()) {
    QualType LHSType = TheCall->getArg(0)->getType();
    QualType RHSType = TheCall->getArg(1)->getType();
    
    if (!LHSType->isVectorType() || !RHSType->isVectorType()) {
      Diag(TheCall->getLocStart(), diag::err_shufflevector_non_vector)
        << SourceRange(TheCall->getArg(0)->getLocStart(),
                       TheCall->getArg(1)->getLocEnd());
      return ExprError();
    }
    
    numElements = LHSType->getAs<VectorType>()->getNumElements();
    unsigned numResElements = TheCall->getNumArgs() - 2;

    // Check to see if we have a call with 2 vector arguments, the unary shuffle
    // with mask.  If so, verify that RHS is an integer vector type with the
    // same number of elts as lhs.
    if (TheCall->getNumArgs() == 2) {
      if (!RHSType->hasIntegerRepresentation() || 
          RHSType->getAs<VectorType>()->getNumElements() != numElements)
        Diag(TheCall->getLocStart(), diag::err_shufflevector_incompatible_vector)
          << SourceRange(TheCall->getArg(1)->getLocStart(),
                         TheCall->getArg(1)->getLocEnd());
      numResElements = numElements;
    }
    else if (!Context.hasSameUnqualifiedType(LHSType, RHSType)) {
      Diag(TheCall->getLocStart(), diag::err_shufflevector_incompatible_vector)
        << SourceRange(TheCall->getArg(0)->getLocStart(),
                       TheCall->getArg(1)->getLocEnd());
      return ExprError();
    } else if (numElements != numResElements) {
      QualType eltType = LHSType->getAs<VectorType>()->getElementType();
      resType = Context.getVectorType(eltType, numResElements,
                                      VectorType::GenericVector);
    }
  }

  for (unsigned i = 2; i < TheCall->getNumArgs(); i++) {
    if (TheCall->getArg(i)->isTypeDependent() ||
        TheCall->getArg(i)->isValueDependent())
      continue;

    llvm::APSInt Result(32);
    if (!TheCall->getArg(i)->isIntegerConstantExpr(Result, Context))
      return ExprError(Diag(TheCall->getLocStart(),
                  diag::err_shufflevector_nonconstant_argument)
                << TheCall->getArg(i)->getSourceRange());

    if (Result.getActiveBits() > 64 || Result.getZExtValue() >= numElements*2)
      return ExprError(Diag(TheCall->getLocStart(),
                  diag::err_shufflevector_argument_too_large)
               << TheCall->getArg(i)->getSourceRange());
  }

  SmallVector<Expr*, 32> exprs;

  for (unsigned i = 0, e = TheCall->getNumArgs(); i != e; i++) {
    exprs.push_back(TheCall->getArg(i));
    TheCall->setArg(i, 0);
  }

  return Owned(new (Context) ShuffleVectorExpr(Context, exprs, resType,
                                            TheCall->getCallee()->getLocStart(),
                                            TheCall->getRParenLoc()));
}

/// SemaBuiltinPrefetch - Handle __builtin_prefetch.
// This is declared to take (const void*, ...) and can take two
// optional constant int args.
bool Sema::SemaBuiltinPrefetch(CallExpr *TheCall) {
  unsigned NumArgs = TheCall->getNumArgs();

  if (NumArgs > 3)
    return Diag(TheCall->getLocEnd(),
             diag::err_typecheck_call_too_many_args_at_most)
             << 0 /*function call*/ << 3 << NumArgs
             << TheCall->getSourceRange();

  // Argument 0 is checked for us and the remaining arguments must be
  // constant integers.
  for (unsigned i = 1; i != NumArgs; ++i) {
    Expr *Arg = TheCall->getArg(i);

    // We can't check the value of a dependent argument.
    if (Arg->isTypeDependent() || Arg->isValueDependent())
      continue;

    llvm::APSInt Result;
    if (SemaBuiltinConstantArg(TheCall, i, Result))
      return true;

    // FIXME: gcc issues a warning and rewrites these to 0. These
    // seems especially odd for the third argument since the default
    // is 3.
    if (i == 1) {
      if (Result.getLimitedValue() > 1)
        return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
             << "0" << "1" << Arg->getSourceRange();
    } else {
      if (Result.getLimitedValue() > 3)
        return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
            << "0" << "3" << Arg->getSourceRange();
    }
  }

  return false;
}

/// SemaBuiltinConstantArg - Handle a check if argument ArgNum of CallExpr
/// TheCall is a constant expression.
bool Sema::SemaBuiltinConstantArg(CallExpr *TheCall, int ArgNum,
                                  llvm::APSInt &Result) {
  Expr *Arg = TheCall->getArg(ArgNum);
  DeclRefExpr *DRE =cast<DeclRefExpr>(TheCall->getCallee()->IgnoreParenCasts());
  FunctionDecl *FDecl = cast<FunctionDecl>(DRE->getDecl());
  
  if (Arg->isTypeDependent() || Arg->isValueDependent()) return false;
  
  if (!Arg->isIntegerConstantExpr(Result, Context))
    return Diag(TheCall->getLocStart(), diag::err_constant_integer_arg_type)
                << FDecl->getDeclName() <<  Arg->getSourceRange();
  
  return false;
}

/// SemaBuiltinObjectSize - Handle __builtin_object_size(void *ptr,
/// int type). This simply type checks that type is one of the defined
/// constants (0-3).
// For compatibility check 0-3, llvm only handles 0 and 2.
bool Sema::SemaBuiltinObjectSize(CallExpr *TheCall) {
  llvm::APSInt Result;

  // We can't check the value of a dependent argument.
  if (TheCall->getArg(1)->isTypeDependent() ||
      TheCall->getArg(1)->isValueDependent())
    return false;

  // Check constant-ness first.
  if (SemaBuiltinConstantArg(TheCall, 1, Result))
    return true;

  Expr *Arg = TheCall->getArg(1);
  if (Result.getSExtValue() < 0 || Result.getSExtValue() > 3) {
    return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
             << "0" << "3" << SourceRange(Arg->getLocStart(), Arg->getLocEnd());
  }

  return false;
}

/// SemaBuiltinLongjmp - Handle __builtin_longjmp(void *env[5], int val).
/// This checks that val is a constant 1.
bool Sema::SemaBuiltinLongjmp(CallExpr *TheCall) {
  Expr *Arg = TheCall->getArg(1);
  llvm::APSInt Result;

  // TODO: This is less than ideal. Overload this to take a value.
  if (SemaBuiltinConstantArg(TheCall, 1, Result))
    return true;
  
  if (Result != 1)
    return Diag(TheCall->getLocStart(), diag::err_builtin_longjmp_invalid_val)
             << SourceRange(Arg->getLocStart(), Arg->getLocEnd());

  return false;
}

// Determine if an expression is a string literal or constant string.
// If this function returns false on the arguments to a function expecting a
// format string, we will usually need to emit a warning.
// True string literals are then checked by CheckFormatString.
Sema::StringLiteralCheckType
Sema::checkFormatStringExpr(const Expr *E, Expr **Args,
                            unsigned NumArgs, bool HasVAListArg,
                            unsigned format_idx, unsigned firstDataArg,
                            FormatStringType Type, VariadicCallType CallType,
                            bool inFunctionCall) {
 tryAgain:
  if (E->isTypeDependent() || E->isValueDependent())
    return SLCT_NotALiteral;

  E = E->IgnoreParenCasts();

  if (E->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNotNull))
    // Technically -Wformat-nonliteral does not warn about this case.
    // The behavior of printf and friends in this case is implementation
    // dependent.  Ideally if the format string cannot be null then
    // it should have a 'nonnull' attribute in the function prototype.
    return SLCT_CheckedLiteral;

  switch (E->getStmtClass()) {
  case Stmt::BinaryConditionalOperatorClass:
  case Stmt::ConditionalOperatorClass: {
    // The expression is a literal if both sub-expressions were, and it was
    // completely checked only if both sub-expressions were checked.
    const AbstractConditionalOperator *C =
        cast<AbstractConditionalOperator>(E);
    StringLiteralCheckType Left =
        checkFormatStringExpr(C->getTrueExpr(), Args, NumArgs,
                              HasVAListArg, format_idx, firstDataArg,
                              Type, CallType, inFunctionCall);
    if (Left == SLCT_NotALiteral)
      return SLCT_NotALiteral;
    StringLiteralCheckType Right =
        checkFormatStringExpr(C->getFalseExpr(), Args, NumArgs,
                              HasVAListArg, format_idx, firstDataArg,
                              Type, CallType, inFunctionCall);
    return Left < Right ? Left : Right;
  }

  case Stmt::ImplicitCastExprClass: {
    E = cast<ImplicitCastExpr>(E)->getSubExpr();
    goto tryAgain;
  }

  case Stmt::OpaqueValueExprClass:
    if (const Expr *src = cast<OpaqueValueExpr>(E)->getSourceExpr()) {
      E = src;
      goto tryAgain;
    }
    return SLCT_NotALiteral;

  case Stmt::PredefinedExprClass:
    // While __func__, etc., are technically not string literals, they
    // cannot contain format specifiers and thus are not a security
    // liability.
    return SLCT_UncheckedLiteral;
      
  case Stmt::DeclRefExprClass: {
    const DeclRefExpr *DR = cast<DeclRefExpr>(E);

    // As an exception, do not flag errors for variables binding to
    // const string literals.
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      bool isConstant = false;
      QualType T = DR->getType();

      if (const ArrayType *AT = Context.getAsArrayType(T)) {
        isConstant = AT->getElementType().isConstant(Context);
      } else if (const PointerType *PT = T->getAs<PointerType>()) {
        isConstant = T.isConstant(Context) &&
                     PT->getPointeeType().isConstant(Context);
      } else if (T->isObjCObjectPointerType()) {
        // In ObjC, there is usually no "const ObjectPointer" type,
        // so don't check if the pointee type is constant.
        isConstant = T.isConstant(Context);
      }

      if (isConstant) {
        if (const Expr *Init = VD->getAnyInitializer()) {
          // Look through initializers like const char c[] = { "foo" }
          if (const InitListExpr *InitList = dyn_cast<InitListExpr>(Init)) {
            if (InitList->isStringLiteralInit())
              Init = InitList->getInit(0)->IgnoreParenImpCasts();
          }
          return checkFormatStringExpr(Init, Args, NumArgs,
                                       HasVAListArg, format_idx,
                                       firstDataArg, Type, CallType,
                                       /*inFunctionCall*/false);
        }
      }

      // For vprintf* functions (i.e., HasVAListArg==true), we add a
      // special check to see if the format string is a function parameter
      // of the function calling the printf function.  If the function
      // has an attribute indicating it is a printf-like function, then we
      // should suppress warnings concerning non-literals being used in a call
      // to a vprintf function.  For example:
      //
      // void
      // logmessage(char const *fmt __attribute__ (format (printf, 1, 2)), ...){
      //      va_list ap;
      //      va_start(ap, fmt);
      //      vprintf(fmt, ap);  // Do NOT emit a warning about "fmt".
      //      ...
      //
      if (HasVAListArg) {
        if (const ParmVarDecl *PV = dyn_cast<ParmVarDecl>(VD)) {
          if (const NamedDecl *ND = dyn_cast<NamedDecl>(PV->getDeclContext())) {
            int PVIndex = PV->getFunctionScopeIndex() + 1;
            for (specific_attr_iterator<FormatAttr>
                 i = ND->specific_attr_begin<FormatAttr>(),
                 e = ND->specific_attr_end<FormatAttr>(); i != e ; ++i) {
              FormatAttr *PVFormat = *i;
              // adjust for implicit parameter
              if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(ND))
                if (MD->isInstance())
                  ++PVIndex;
              // We also check if the formats are compatible.
              // We can't pass a 'scanf' string to a 'printf' function.
              if (PVIndex == PVFormat->getFormatIdx() &&
                  Type == GetFormatStringType(PVFormat))
                return SLCT_UncheckedLiteral;
            }
          }
        }
      }
    }

    return SLCT_NotALiteral;
  }

  case Stmt::CallExprClass:
  case Stmt::CXXMemberCallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl())) {
      if (const FormatArgAttr *FA = ND->getAttr<FormatArgAttr>()) {
        unsigned ArgIndex = FA->getFormatIdx();
        if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(ND))
          if (MD->isInstance())
            --ArgIndex;
        const Expr *Arg = CE->getArg(ArgIndex - 1);

        return checkFormatStringExpr(Arg, Args, NumArgs,
                                     HasVAListArg, format_idx, firstDataArg,
                                     Type, CallType, inFunctionCall);
      } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
        unsigned BuiltinID = FD->getBuiltinID();
        if (BuiltinID == Builtin::BI__builtin___CFStringMakeConstantString ||
            BuiltinID == Builtin::BI__builtin___NSStringMakeConstantString) {
          const Expr *Arg = CE->getArg(0);
          return checkFormatStringExpr(Arg, Args, NumArgs,
                                       HasVAListArg, format_idx,
                                       firstDataArg, Type, CallType,
                                       inFunctionCall);
        }
      }
    }

    return SLCT_NotALiteral;
  }
  case Stmt::ObjCStringLiteralClass:
  case Stmt::StringLiteralClass: {
    const StringLiteral *StrE = NULL;

    if (const ObjCStringLiteral *ObjCFExpr = dyn_cast<ObjCStringLiteral>(E))
      StrE = ObjCFExpr->getString();
    else
      StrE = cast<StringLiteral>(E);

    if (StrE) {
      CheckFormatString(StrE, E, Args, NumArgs, HasVAListArg, format_idx,
                        firstDataArg, Type, inFunctionCall, CallType);
      return SLCT_CheckedLiteral;
    }

    return SLCT_NotALiteral;
  }

  default:
    return SLCT_NotALiteral;
  }
}

void
Sema::CheckNonNullArguments(const NonNullAttr *NonNull,
                            const Expr * const *ExprArgs,
                            SourceLocation CallSiteLoc) {
  for (NonNullAttr::args_iterator i = NonNull->args_begin(),
                                  e = NonNull->args_end();
       i != e; ++i) {
    const Expr *ArgExpr = ExprArgs[*i];
    if (ArgExpr->isNullPointerConstant(Context,
                                       Expr::NPC_ValueDependentIsNotNull))
      Diag(CallSiteLoc, diag::warn_null_arg) << ArgExpr->getSourceRange();
  }
}

Sema::FormatStringType Sema::GetFormatStringType(const FormatAttr *Format) {
  return llvm::StringSwitch<FormatStringType>(Format->getType())
  .Case("scanf", FST_Scanf)
  .Cases("printf", "printf0", FST_Printf)
  .Cases("NSString", "CFString", FST_NSString)
  .Case("strftime", FST_Strftime)
  .Case("strfmon", FST_Strfmon)
  .Cases("kprintf", "cmn_err", "vcmn_err", "zcmn_err", FST_Kprintf)
  .Default(FST_Unknown);
}

/// CheckFormatArguments - Check calls to printf and scanf (and similar
/// functions) for correct use of format strings.
/// Returns true if a format string has been fully checked.
bool Sema::CheckFormatArguments(const FormatAttr *Format, Expr **Args,
                                unsigned NumArgs, bool IsCXXMember,
                                VariadicCallType CallType,
                                SourceLocation Loc, SourceRange Range) {
  FormatStringInfo FSI;
  if (getFormatStringInfo(Format, IsCXXMember, &FSI))
    return CheckFormatArguments(Args, NumArgs, FSI.HasVAListArg, FSI.FormatIdx,
                                FSI.FirstDataArg, GetFormatStringType(Format),
                                CallType, Loc, Range);
  return false;
}

bool Sema::CheckFormatArguments(Expr **Args, unsigned NumArgs,
                                bool HasVAListArg, unsigned format_idx,
                                unsigned firstDataArg, FormatStringType Type,
                                VariadicCallType CallType,
                                SourceLocation Loc, SourceRange Range) {
  // CHECK: printf/scanf-like function is called with no format string.
  if (format_idx >= NumArgs) {
    Diag(Loc, diag::warn_missing_format_string) << Range;
    return false;
  }

  const Expr *OrigFormatExpr = Args[format_idx]->IgnoreParenCasts();

  // CHECK: format string is not a string literal.
  //
  // Dynamically generated format strings are difficult to
  // automatically vet at compile time.  Requiring that format strings
  // are string literals: (1) permits the checking of format strings by
  // the compiler and thereby (2) can practically remove the source of
  // many format string exploits.

  // Format string can be either ObjC string (e.g. @"%d") or
  // C string (e.g. "%d")
  // ObjC string uses the same format specifiers as C string, so we can use
  // the same format string checking logic for both ObjC and C strings.
  StringLiteralCheckType CT =
      checkFormatStringExpr(OrigFormatExpr, Args, NumArgs, HasVAListArg,
                            format_idx, firstDataArg, Type, CallType);
  if (CT != SLCT_NotALiteral)
    // Literal format string found, check done!
    return CT == SLCT_CheckedLiteral;

  // Strftime is particular as it always uses a single 'time' argument,
  // so it is safe to pass a non-literal string.
  if (Type == FST_Strftime)
    return false;

  // Do not emit diag when the string param is a macro expansion and the
  // format is either NSString or CFString. This is a hack to prevent
  // diag when using the NSLocalizedString and CFCopyLocalizedString macros
  // which are usually used in place of NS and CF string literals.
  if (Type == FST_NSString &&
      SourceMgr.isInSystemMacro(Args[format_idx]->getLocStart()))
    return false;

  // If there are no arguments specified, warn with -Wformat-security, otherwise
  // warn only with -Wformat-nonliteral.
  if (NumArgs == format_idx+1)
    Diag(Args[format_idx]->getLocStart(),
         diag::warn_format_nonliteral_noargs)
      << OrigFormatExpr->getSourceRange();
  else
    Diag(Args[format_idx]->getLocStart(),
         diag::warn_format_nonliteral)
           << OrigFormatExpr->getSourceRange();
  return false;
}

namespace {
class CheckFormatHandler : public analyze_format_string::FormatStringHandler {
protected:
  Sema &S;
  const StringLiteral *FExpr;
  const Expr *OrigFormatExpr;
  const unsigned FirstDataArg;
  const unsigned NumDataArgs;
  const char *Beg; // Start of format string.
  const bool HasVAListArg;
  const Expr * const *Args;
  const unsigned NumArgs;
  unsigned FormatIdx;
  llvm::BitVector CoveredArgs;
  bool usesPositionalArgs;
  bool atFirstArg;
  bool inFunctionCall;
  Sema::VariadicCallType CallType;
public:
  CheckFormatHandler(Sema &s, const StringLiteral *fexpr,
                     const Expr *origFormatExpr, unsigned firstDataArg,
                     unsigned numDataArgs, const char *beg, bool hasVAListArg,
                     Expr **args, unsigned numArgs,
                     unsigned formatIdx, bool inFunctionCall,
                     Sema::VariadicCallType callType)
    : S(s), FExpr(fexpr), OrigFormatExpr(origFormatExpr),
      FirstDataArg(firstDataArg), NumDataArgs(numDataArgs),
      Beg(beg), HasVAListArg(hasVAListArg),
      Args(args), NumArgs(numArgs), FormatIdx(formatIdx),
      usesPositionalArgs(false), atFirstArg(true),
      inFunctionCall(inFunctionCall), CallType(callType) {
        CoveredArgs.resize(numDataArgs);
        CoveredArgs.reset();
      }

  void DoneProcessing();

  void HandleIncompleteSpecifier(const char *startSpecifier,
                                 unsigned specifierLen);

  void HandleInvalidLengthModifier(
      const analyze_format_string::FormatSpecifier &FS,
      const analyze_format_string::ConversionSpecifier &CS,
      const char *startSpecifier, unsigned specifierLen, unsigned DiagID);

  void HandleNonStandardLengthModifier(
      const analyze_format_string::FormatSpecifier &FS,
      const char *startSpecifier, unsigned specifierLen);

  void HandleNonStandardConversionSpecifier(
      const analyze_format_string::ConversionSpecifier &CS,
      const char *startSpecifier, unsigned specifierLen);

  virtual void HandlePosition(const char *startPos, unsigned posLen);

  virtual void HandleInvalidPosition(const char *startSpecifier,
                                     unsigned specifierLen,
                                     analyze_format_string::PositionContext p);

  virtual void HandleZeroPosition(const char *startPos, unsigned posLen);

  void HandleNullChar(const char *nullCharacter);

  template <typename Range>
  static void EmitFormatDiagnostic(Sema &S, bool inFunctionCall,
                                   const Expr *ArgumentExpr,
                                   PartialDiagnostic PDiag,
                                   SourceLocation StringLoc,
                                   bool IsStringLocation, Range StringRange,
                            ArrayRef<FixItHint> Fixit = ArrayRef<FixItHint>());

protected:
  bool HandleInvalidConversionSpecifier(unsigned argIndex, SourceLocation Loc,
                                        const char *startSpec,
                                        unsigned specifierLen,
                                        const char *csStart, unsigned csLen);

  void HandlePositionalNonpositionalArgs(SourceLocation Loc,
                                         const char *startSpec,
                                         unsigned specifierLen);
  
  SourceRange getFormatStringRange();
  CharSourceRange getSpecifierRange(const char *startSpecifier,
                                    unsigned specifierLen);
  SourceLocation getLocationOfByte(const char *x);

  const Expr *getDataArg(unsigned i) const;
  
  bool CheckNumArgs(const analyze_format_string::FormatSpecifier &FS,
                    const analyze_format_string::ConversionSpecifier &CS,
                    const char *startSpecifier, unsigned specifierLen,
                    unsigned argIndex);

  template <typename Range>
  void EmitFormatDiagnostic(PartialDiagnostic PDiag, SourceLocation StringLoc,
                            bool IsStringLocation, Range StringRange,
                            ArrayRef<FixItHint> Fixit = ArrayRef<FixItHint>());

  void CheckPositionalAndNonpositionalArgs(
      const analyze_format_string::FormatSpecifier *FS);
};
}

SourceRange CheckFormatHandler::getFormatStringRange() {
  return OrigFormatExpr->getSourceRange();
}

CharSourceRange CheckFormatHandler::
getSpecifierRange(const char *startSpecifier, unsigned specifierLen) {
  SourceLocation Start = getLocationOfByte(startSpecifier);
  SourceLocation End   = getLocationOfByte(startSpecifier + specifierLen - 1);

  // Advance the end SourceLocation by one due to half-open ranges.
  End = End.getLocWithOffset(1);

  return CharSourceRange::getCharRange(Start, End);
}

SourceLocation CheckFormatHandler::getLocationOfByte(const char *x) {
  return S.getLocationOfStringLiteralByte(FExpr, x - Beg);
}

void CheckFormatHandler::HandleIncompleteSpecifier(const char *startSpecifier,
                                                   unsigned specifierLen){
  EmitFormatDiagnostic(S.PDiag(diag::warn_printf_incomplete_specifier),
                       getLocationOfByte(startSpecifier),
                       /*IsStringLocation*/true,
                       getSpecifierRange(startSpecifier, specifierLen));
}

void CheckFormatHandler::HandleInvalidLengthModifier(
    const analyze_format_string::FormatSpecifier &FS,
    const analyze_format_string::ConversionSpecifier &CS,
    const char *startSpecifier, unsigned specifierLen, unsigned DiagID) {
  using namespace analyze_format_string;

  const LengthModifier &LM = FS.getLengthModifier();
  CharSourceRange LMRange = getSpecifierRange(LM.getStart(), LM.getLength());

  // See if we know how to fix this length modifier.
  llvm::Optional<LengthModifier> FixedLM = FS.getCorrectedLengthModifier();
  if (FixedLM) {
    EmitFormatDiagnostic(S.PDiag(DiagID) << LM.toString() << CS.toString(),
                         getLocationOfByte(LM.getStart()),
                         /*IsStringLocation*/true,
                         getSpecifierRange(startSpecifier, specifierLen));

    S.Diag(getLocationOfByte(LM.getStart()), diag::note_format_fix_specifier)
      << FixedLM->toString()
      << FixItHint::CreateReplacement(LMRange, FixedLM->toString());

  } else {
    FixItHint Hint;
    if (DiagID == diag::warn_format_nonsensical_length)
      Hint = FixItHint::CreateRemoval(LMRange);

    EmitFormatDiagnostic(S.PDiag(DiagID) << LM.toString() << CS.toString(),
                         getLocationOfByte(LM.getStart()),
                         /*IsStringLocation*/true,
                         getSpecifierRange(startSpecifier, specifierLen),
                         Hint);
  }
}

void CheckFormatHandler::HandleNonStandardLengthModifier(
    const analyze_format_string::FormatSpecifier &FS,
    const char *startSpecifier, unsigned specifierLen) {
  using namespace analyze_format_string;

  const LengthModifier &LM = FS.getLengthModifier();
  CharSourceRange LMRange = getSpecifierRange(LM.getStart(), LM.getLength());

  // See if we know how to fix this length modifier.
  llvm::Optional<LengthModifier> FixedLM = FS.getCorrectedLengthModifier();
  if (FixedLM) {
    EmitFormatDiagnostic(S.PDiag(diag::warn_format_non_standard)
                           << LM.toString() << 0,
                         getLocationOfByte(LM.getStart()),
                         /*IsStringLocation*/true,
                         getSpecifierRange(startSpecifier, specifierLen));

    S.Diag(getLocationOfByte(LM.getStart()), diag::note_format_fix_specifier)
      << FixedLM->toString()
      << FixItHint::CreateReplacement(LMRange, FixedLM->toString());

  } else {
    EmitFormatDiagnostic(S.PDiag(diag::warn_format_non_standard)
                           << LM.toString() << 0,
                         getLocationOfByte(LM.getStart()),
                         /*IsStringLocation*/true,
                         getSpecifierRange(startSpecifier, specifierLen));
  }
}

void CheckFormatHandler::HandleNonStandardConversionSpecifier(
    const analyze_format_string::ConversionSpecifier &CS,
    const char *startSpecifier, unsigned specifierLen) {
  using namespace analyze_format_string;

  // See if we know how to fix this conversion specifier.
  llvm::Optional<ConversionSpecifier> FixedCS = CS.getStandardSpecifier();
  if (FixedCS) {
    EmitFormatDiagnostic(S.PDiag(diag::warn_format_non_standard)
                          << CS.toString() << /*conversion specifier*/1,
                         getLocationOfByte(CS.getStart()),
                         /*IsStringLocation*/true,
                         getSpecifierRange(startSpecifier, specifierLen));

    CharSourceRange CSRange = getSpecifierRange(CS.getStart(), CS.getLength());
    S.Diag(getLocationOfByte(CS.getStart()), diag::note_format_fix_specifier)
      << FixedCS->toString()
      << FixItHint::CreateReplacement(CSRange, FixedCS->toString());
  } else {
    EmitFormatDiagnostic(S.PDiag(diag::warn_format_non_standard)
                          << CS.toString() << /*conversion specifier*/1,
                         getLocationOfByte(CS.getStart()),
                         /*IsStringLocation*/true,
                         getSpecifierRange(startSpecifier, specifierLen));
  }
}

void CheckFormatHandler::HandlePosition(const char *startPos,
                                        unsigned posLen) {
  EmitFormatDiagnostic(S.PDiag(diag::warn_format_non_standard_positional_arg),
                               getLocationOfByte(startPos),
                               /*IsStringLocation*/true,
                               getSpecifierRange(startPos, posLen));
}

void
CheckFormatHandler::HandleInvalidPosition(const char *startPos, unsigned posLen,
                                     analyze_format_string::PositionContext p) {
  EmitFormatDiagnostic(S.PDiag(diag::warn_format_invalid_positional_specifier)
                         << (unsigned) p,
                       getLocationOfByte(startPos), /*IsStringLocation*/true,
                       getSpecifierRange(startPos, posLen));
}

void CheckFormatHandler::HandleZeroPosition(const char *startPos,
                                            unsigned posLen) {
  EmitFormatDiagnostic(S.PDiag(diag::warn_format_zero_positional_specifier),
                               getLocationOfByte(startPos),
                               /*IsStringLocation*/true,
                               getSpecifierRange(startPos, posLen));
}

void CheckFormatHandler::HandleNullChar(const char *nullCharacter) {
  if (!isa<ObjCStringLiteral>(OrigFormatExpr)) {
    // The presence of a null character is likely an error.
    EmitFormatDiagnostic(
      S.PDiag(diag::warn_printf_format_string_contains_null_char),
      getLocationOfByte(nullCharacter), /*IsStringLocation*/true,
      getFormatStringRange());
  }
}

// Note that this may return NULL if there was an error parsing or building
// one of the argument expressions.
const Expr *CheckFormatHandler::getDataArg(unsigned i) const {
  return Args[FirstDataArg + i];
}

void CheckFormatHandler::DoneProcessing() {
    // Does the number of data arguments exceed the number of
    // format conversions in the format string?
  if (!HasVAListArg) {
      // Find any arguments that weren't covered.
    CoveredArgs.flip();
    signed notCoveredArg = CoveredArgs.find_first();
    if (notCoveredArg >= 0) {
      assert((unsigned)notCoveredArg < NumDataArgs);
      if (const Expr *E = getDataArg((unsigned) notCoveredArg)) {
        SourceLocation Loc = E->getLocStart();
        if (!S.getSourceManager().isInSystemMacro(Loc)) {
          EmitFormatDiagnostic(S.PDiag(diag::warn_printf_data_arg_not_used),
                               Loc, /*IsStringLocation*/false,
                               getFormatStringRange());
        }
      }
    }
  }
}

bool
CheckFormatHandler::HandleInvalidConversionSpecifier(unsigned argIndex,
                                                     SourceLocation Loc,
                                                     const char *startSpec,
                                                     unsigned specifierLen,
                                                     const char *csStart,
                                                     unsigned csLen) {
  
  bool keepGoing = true;
  if (argIndex < NumDataArgs) {
    // Consider the argument coverered, even though the specifier doesn't
    // make sense.
    CoveredArgs.set(argIndex);
  }
  else {
    // If argIndex exceeds the number of data arguments we
    // don't issue a warning because that is just a cascade of warnings (and
    // they may have intended '%%' anyway). We don't want to continue processing
    // the format string after this point, however, as we will like just get
    // gibberish when trying to match arguments.
    keepGoing = false;
  }
  
  EmitFormatDiagnostic(S.PDiag(diag::warn_format_invalid_conversion)
                         << StringRef(csStart, csLen),
                       Loc, /*IsStringLocation*/true,
                       getSpecifierRange(startSpec, specifierLen));
  
  return keepGoing;
}

void
CheckFormatHandler::HandlePositionalNonpositionalArgs(SourceLocation Loc,
                                                      const char *startSpec,
                                                      unsigned specifierLen) {
  EmitFormatDiagnostic(
    S.PDiag(diag::warn_format_mix_positional_nonpositional_args),
    Loc, /*isStringLoc*/true, getSpecifierRange(startSpec, specifierLen));
}

bool
CheckFormatHandler::CheckNumArgs(
  const analyze_format_string::FormatSpecifier &FS,
  const analyze_format_string::ConversionSpecifier &CS,
  const char *startSpecifier, unsigned specifierLen, unsigned argIndex) {

  if (argIndex >= NumDataArgs) {
    PartialDiagnostic PDiag = FS.usesPositionalArg()
      ? (S.PDiag(diag::warn_printf_positional_arg_exceeds_data_args)
           << (argIndex+1) << NumDataArgs)
      : S.PDiag(diag::warn_printf_insufficient_data_args);
    EmitFormatDiagnostic(
      PDiag, getLocationOfByte(CS.getStart()), /*IsStringLocation*/true,
      getSpecifierRange(startSpecifier, specifierLen));
    return false;
  }
  return true;
}

template<typename Range>
void CheckFormatHandler::EmitFormatDiagnostic(PartialDiagnostic PDiag,
                                              SourceLocation Loc,
                                              bool IsStringLocation,
                                              Range StringRange,
                                              ArrayRef<FixItHint> FixIt) {
  EmitFormatDiagnostic(S, inFunctionCall, Args[FormatIdx], PDiag,
                       Loc, IsStringLocation, StringRange, FixIt);
}

/// \brief If the format string is not within the funcion call, emit a note
/// so that the function call and string are in diagnostic messages.
///
/// \param InFunctionCall if true, the format string is within the function
/// call and only one diagnostic message will be produced.  Otherwise, an
/// extra note will be emitted pointing to location of the format string.
///
/// \param ArgumentExpr the expression that is passed as the format string
/// argument in the function call.  Used for getting locations when two
/// diagnostics are emitted.
///
/// \param PDiag the callee should already have provided any strings for the
/// diagnostic message.  This function only adds locations and fixits
/// to diagnostics.
///
/// \param Loc primary location for diagnostic.  If two diagnostics are
/// required, one will be at Loc and a new SourceLocation will be created for
/// the other one.
///
/// \param IsStringLocation if true, Loc points to the format string should be
/// used for the note.  Otherwise, Loc points to the argument list and will
/// be used with PDiag.
///
/// \param StringRange some or all of the string to highlight.  This is
/// templated so it can accept either a CharSourceRange or a SourceRange.
///
/// \param FixIt optional fix it hint for the format string.
template<typename Range>
void CheckFormatHandler::EmitFormatDiagnostic(Sema &S, bool InFunctionCall,
                                              const Expr *ArgumentExpr,
                                              PartialDiagnostic PDiag,
                                              SourceLocation Loc,
                                              bool IsStringLocation,
                                              Range StringRange,
                                              ArrayRef<FixItHint> FixIt) {
  if (InFunctionCall) {
    const Sema::SemaDiagnosticBuilder &D = S.Diag(Loc, PDiag);
    D << StringRange;
    for (ArrayRef<FixItHint>::iterator I = FixIt.begin(), E = FixIt.end();
         I != E; ++I) {
      D << *I;
    }
  } else {
    S.Diag(IsStringLocation ? ArgumentExpr->getExprLoc() : Loc, PDiag)
      << ArgumentExpr->getSourceRange();

    const Sema::SemaDiagnosticBuilder &Note =
      S.Diag(IsStringLocation ? Loc : StringRange.getBegin(),
             diag::note_format_string_defined);

    Note << StringRange;
    for (ArrayRef<FixItHint>::iterator I = FixIt.begin(), E = FixIt.end();
         I != E; ++I) {
      Note << *I;
    }
  }
}

//===--- CHECK: Printf format string checking ------------------------------===//

namespace {
class CheckPrintfHandler : public CheckFormatHandler {
  bool ObjCContext;
public:
  CheckPrintfHandler(Sema &s, const StringLiteral *fexpr,
                     const Expr *origFormatExpr, unsigned firstDataArg,
                     unsigned numDataArgs, bool isObjC,
                     const char *beg, bool hasVAListArg,
                     Expr **Args, unsigned NumArgs,
                     unsigned formatIdx, bool inFunctionCall,
                     Sema::VariadicCallType CallType)
  : CheckFormatHandler(s, fexpr, origFormatExpr, firstDataArg,
                       numDataArgs, beg, hasVAListArg, Args, NumArgs,
                       formatIdx, inFunctionCall, CallType), ObjCContext(isObjC)
  {}

  
  bool HandleInvalidPrintfConversionSpecifier(
                                      const analyze_printf::PrintfSpecifier &FS,
                                      const char *startSpecifier,
                                      unsigned specifierLen);
  
  bool HandlePrintfSpecifier(const analyze_printf::PrintfSpecifier &FS,
                             const char *startSpecifier,
                             unsigned specifierLen);
  bool checkFormatExpr(const analyze_printf::PrintfSpecifier &FS,
                       const char *StartSpecifier,
                       unsigned SpecifierLen,
                       const Expr *E);

  bool HandleAmount(const analyze_format_string::OptionalAmount &Amt, unsigned k,
                    const char *startSpecifier, unsigned specifierLen);
  void HandleInvalidAmount(const analyze_printf::PrintfSpecifier &FS,
                           const analyze_printf::OptionalAmount &Amt,
                           unsigned type,
                           const char *startSpecifier, unsigned specifierLen);
  void HandleFlag(const analyze_printf::PrintfSpecifier &FS,
                  const analyze_printf::OptionalFlag &flag,
                  const char *startSpecifier, unsigned specifierLen);
  void HandleIgnoredFlag(const analyze_printf::PrintfSpecifier &FS,
                         const analyze_printf::OptionalFlag &ignoredFlag,
                         const analyze_printf::OptionalFlag &flag,
                         const char *startSpecifier, unsigned specifierLen);
  bool checkForCStrMembers(const analyze_printf::ArgType &AT,
                           const Expr *E, const CharSourceRange &CSR);

};  
}

bool CheckPrintfHandler::HandleInvalidPrintfConversionSpecifier(
                                      const analyze_printf::PrintfSpecifier &FS,
                                      const char *startSpecifier,
                                      unsigned specifierLen) {
  const analyze_printf::PrintfConversionSpecifier &CS =
    FS.getConversionSpecifier();
  
  return HandleInvalidConversionSpecifier(FS.getArgIndex(),
                                          getLocationOfByte(CS.getStart()),
                                          startSpecifier, specifierLen,
                                          CS.getStart(), CS.getLength());
}

bool CheckPrintfHandler::HandleAmount(
                               const analyze_format_string::OptionalAmount &Amt,
                               unsigned k, const char *startSpecifier,
                               unsigned specifierLen) {

  if (Amt.hasDataArgument()) {
    if (!HasVAListArg) {
      unsigned argIndex = Amt.getArgIndex();
      if (argIndex >= NumDataArgs) {
        EmitFormatDiagnostic(S.PDiag(diag::warn_printf_asterisk_missing_arg)
                               << k,
                             getLocationOfByte(Amt.getStart()),
                             /*IsStringLocation*/true,
                             getSpecifierRange(startSpecifier, specifierLen));
        // Don't do any more checking.  We will just emit
        // spurious errors.
        return false;
      }

      // Type check the data argument.  It should be an 'int'.
      // Although not in conformance with C99, we also allow the argument to be
      // an 'unsigned int' as that is a reasonably safe case.  GCC also
      // doesn't emit a warning for that case.
      CoveredArgs.set(argIndex);
      const Expr *Arg = getDataArg(argIndex);
      if (!Arg)
        return false;

      QualType T = Arg->getType();

      const analyze_printf::ArgType &AT = Amt.getArgType(S.Context);
      assert(AT.isValid());

      if (!AT.matchesType(S.Context, T)) {
        EmitFormatDiagnostic(S.PDiag(diag::warn_printf_asterisk_wrong_type)
                               << k << AT.getRepresentativeTypeName(S.Context)
                               << T << Arg->getSourceRange(),
                             getLocationOfByte(Amt.getStart()),
                             /*IsStringLocation*/true,
                             getSpecifierRange(startSpecifier, specifierLen));
        // Don't do any more checking.  We will just emit
        // spurious errors.
        return false;
      }
    }
  }
  return true;
}

void CheckPrintfHandler::HandleInvalidAmount(
                                      const analyze_printf::PrintfSpecifier &FS,
                                      const analyze_printf::OptionalAmount &Amt,
                                      unsigned type,
                                      const char *startSpecifier,
                                      unsigned specifierLen) {
  const analyze_printf::PrintfConversionSpecifier &CS =
    FS.getConversionSpecifier();

  FixItHint fixit =
    Amt.getHowSpecified() == analyze_printf::OptionalAmount::Constant
      ? FixItHint::CreateRemoval(getSpecifierRange(Amt.getStart(),
                                 Amt.getConstantLength()))
      : FixItHint();

  EmitFormatDiagnostic(S.PDiag(diag::warn_printf_nonsensical_optional_amount)
                         << type << CS.toString(),
                       getLocationOfByte(Amt.getStart()),
                       /*IsStringLocation*/true,
                       getSpecifierRange(startSpecifier, specifierLen),
                       fixit);
}

void CheckPrintfHandler::HandleFlag(const analyze_printf::PrintfSpecifier &FS,
                                    const analyze_printf::OptionalFlag &flag,
                                    const char *startSpecifier,
                                    unsigned specifierLen) {
  // Warn about pointless flag with a fixit removal.
  const analyze_printf::PrintfConversionSpecifier &CS =
    FS.getConversionSpecifier();
  EmitFormatDiagnostic(S.PDiag(diag::warn_printf_nonsensical_flag)
                         << flag.toString() << CS.toString(),
                       getLocationOfByte(flag.getPosition()),
                       /*IsStringLocation*/true,
                       getSpecifierRange(startSpecifier, specifierLen),
                       FixItHint::CreateRemoval(
                         getSpecifierRange(flag.getPosition(), 1)));
}

void CheckPrintfHandler::HandleIgnoredFlag(
                                const analyze_printf::PrintfSpecifier &FS,
                                const analyze_printf::OptionalFlag &ignoredFlag,
                                const analyze_printf::OptionalFlag &flag,
                                const char *startSpecifier,
                                unsigned specifierLen) {
  // Warn about ignored flag with a fixit removal.
  EmitFormatDiagnostic(S.PDiag(diag::warn_printf_ignored_flag)
                         << ignoredFlag.toString() << flag.toString(),
                       getLocationOfByte(ignoredFlag.getPosition()),
                       /*IsStringLocation*/true,
                       getSpecifierRange(startSpecifier, specifierLen),
                       FixItHint::CreateRemoval(
                         getSpecifierRange(ignoredFlag.getPosition(), 1)));
}

// Determines if the specified is a C++ class or struct containing
// a member with the specified name and kind (e.g. a CXXMethodDecl named
// "c_str()").
template<typename MemberKind>
static llvm::SmallPtrSet<MemberKind*, 1>
CXXRecordMembersNamed(StringRef Name, Sema &S, QualType Ty) {
  const RecordType *RT = Ty->getAs<RecordType>();
  llvm::SmallPtrSet<MemberKind*, 1> Results;

  if (!RT)
    return Results;
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return Results;

  LookupResult R(S, &S.PP.getIdentifierTable().get(Name), SourceLocation(),
                 Sema::LookupMemberName);

  // We just need to include all members of the right kind turned up by the
  // filter, at this point.
  if (S.LookupQualifiedName(R, RT->getDecl()))
    for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I) {
      NamedDecl *decl = (*I)->getUnderlyingDecl();
      if (MemberKind *FK = dyn_cast<MemberKind>(decl))
        Results.insert(FK);
    }
  return Results;
}

// Check if a (w)string was passed when a (w)char* was needed, and offer a
// better diagnostic if so. AT is assumed to be valid.
// Returns true when a c_str() conversion method is found.
bool CheckPrintfHandler::checkForCStrMembers(
    const analyze_printf::ArgType &AT, const Expr *E,
    const CharSourceRange &CSR) {
  typedef llvm::SmallPtrSet<CXXMethodDecl*, 1> MethodSet;

  MethodSet Results =
      CXXRecordMembersNamed<CXXMethodDecl>("c_str", S, E->getType());

  for (MethodSet::iterator MI = Results.begin(), ME = Results.end();
       MI != ME; ++MI) {
    const CXXMethodDecl *Method = *MI;
    if (Method->getNumParams() == 0 &&
          AT.matchesType(S.Context, Method->getResultType())) {
      // FIXME: Suggest parens if the expression needs them.
      SourceLocation EndLoc =
          S.getPreprocessor().getLocForEndOfToken(E->getLocEnd());
      S.Diag(E->getLocStart(), diag::note_printf_c_str)
          << "c_str()"
          << FixItHint::CreateInsertion(EndLoc, ".c_str()");
      return true;
    }
  }

  return false;
}

bool
CheckPrintfHandler::HandlePrintfSpecifier(const analyze_printf::PrintfSpecifier
                                            &FS,
                                          const char *startSpecifier,
                                          unsigned specifierLen) {

  using namespace analyze_format_string;
  using namespace analyze_printf;  
  const PrintfConversionSpecifier &CS = FS.getConversionSpecifier();

  if (FS.consumesDataArgument()) {
    if (atFirstArg) {
        atFirstArg = false;
        usesPositionalArgs = FS.usesPositionalArg();
    }
    else if (usesPositionalArgs != FS.usesPositionalArg()) {
      HandlePositionalNonpositionalArgs(getLocationOfByte(CS.getStart()),
                                        startSpecifier, specifierLen);
      return false;
    }
  }

  // First check if the field width, precision, and conversion specifier
  // have matching data arguments.
  if (!HandleAmount(FS.getFieldWidth(), /* field width */ 0,
                    startSpecifier, specifierLen)) {
    return false;
  }

  if (!HandleAmount(FS.getPrecision(), /* precision */ 1,
                    startSpecifier, specifierLen)) {
    return false;
  }

  if (!CS.consumesDataArgument()) {
    // FIXME: Technically specifying a precision or field width here
    // makes no sense.  Worth issuing a warning at some point.
    return true;
  }

  // Consume the argument.
  unsigned argIndex = FS.getArgIndex();
  if (argIndex < NumDataArgs) {
    // The check to see if the argIndex is valid will come later.
    // We set the bit here because we may exit early from this
    // function if we encounter some other error.
    CoveredArgs.set(argIndex);
  }

  // Check for using an Objective-C specific conversion specifier
  // in a non-ObjC literal.
  if (!ObjCContext && CS.isObjCArg()) {
    return HandleInvalidPrintfConversionSpecifier(FS, startSpecifier,
                                                  specifierLen);
  }

  // Check for invalid use of field width
  if (!FS.hasValidFieldWidth()) {
    HandleInvalidAmount(FS, FS.getFieldWidth(), /* field width */ 0,
        startSpecifier, specifierLen);
  }

  // Check for invalid use of precision
  if (!FS.hasValidPrecision()) {
    HandleInvalidAmount(FS, FS.getPrecision(), /* precision */ 1,
        startSpecifier, specifierLen);
  }

  // Check each flag does not conflict with any other component.
  if (!FS.hasValidThousandsGroupingPrefix())
    HandleFlag(FS, FS.hasThousandsGrouping(), startSpecifier, specifierLen);
  if (!FS.hasValidLeadingZeros())
    HandleFlag(FS, FS.hasLeadingZeros(), startSpecifier, specifierLen);
  if (!FS.hasValidPlusPrefix())
    HandleFlag(FS, FS.hasPlusPrefix(), startSpecifier, specifierLen);
  if (!FS.hasValidSpacePrefix())
    HandleFlag(FS, FS.hasSpacePrefix(), startSpecifier, specifierLen);
  if (!FS.hasValidAlternativeForm())
    HandleFlag(FS, FS.hasAlternativeForm(), startSpecifier, specifierLen);
  if (!FS.hasValidLeftJustified())
    HandleFlag(FS, FS.isLeftJustified(), startSpecifier, specifierLen);

  // Check that flags are not ignored by another flag
  if (FS.hasSpacePrefix() && FS.hasPlusPrefix()) // ' ' ignored by '+'
    HandleIgnoredFlag(FS, FS.hasSpacePrefix(), FS.hasPlusPrefix(),
        startSpecifier, specifierLen);
  if (FS.hasLeadingZeros() && FS.isLeftJustified()) // '0' ignored by '-'
    HandleIgnoredFlag(FS, FS.hasLeadingZeros(), FS.isLeftJustified(),
            startSpecifier, specifierLen);

  // Check the length modifier is valid with the given conversion specifier.
  if (!FS.hasValidLengthModifier(S.getASTContext().getTargetInfo()))
    HandleInvalidLengthModifier(FS, CS, startSpecifier, specifierLen,
                                diag::warn_format_nonsensical_length);
  else if (!FS.hasStandardLengthModifier())
    HandleNonStandardLengthModifier(FS, startSpecifier, specifierLen);
  else if (!FS.hasStandardLengthConversionCombination())
    HandleInvalidLengthModifier(FS, CS, startSpecifier, specifierLen,
                                diag::warn_format_non_standard_conversion_spec);

  if (!FS.hasStandardConversionSpecifier(S.getLangOpts()))
    HandleNonStandardConversionSpecifier(CS, startSpecifier, specifierLen);

  // The remaining checks depend on the data arguments.
  if (HasVAListArg)
    return true;

  if (!CheckNumArgs(FS, CS, startSpecifier, specifierLen, argIndex))
    return false;

  const Expr *Arg = getDataArg(argIndex);
  if (!Arg)
    return true;

  return checkFormatExpr(FS, startSpecifier, specifierLen, Arg);
}

static bool requiresParensToAddCast(const Expr *E) {
  // FIXME: We should have a general way to reason about operator
  // precedence and whether parens are actually needed here.
  // Take care of a few common cases where they aren't.
  const Expr *Inside = E->IgnoreImpCasts();
  if (const PseudoObjectExpr *POE = dyn_cast<PseudoObjectExpr>(Inside))
    Inside = POE->getSyntacticForm()->IgnoreImpCasts();

  switch (Inside->getStmtClass()) {
  case Stmt::ArraySubscriptExprClass:
  case Stmt::CallExprClass:
  case Stmt::DeclRefExprClass:
  case Stmt::MemberExprClass:
  case Stmt::ObjCIvarRefExprClass:
  case Stmt::ObjCMessageExprClass:
  case Stmt::ObjCPropertyRefExprClass:
  case Stmt::ParenExprClass:
  case Stmt::UnaryOperatorClass:
    return false;
  default:
    return true;
  }
}

bool
CheckPrintfHandler::checkFormatExpr(const analyze_printf::PrintfSpecifier &FS,
                                    const char *StartSpecifier,
                                    unsigned SpecifierLen,
                                    const Expr *E) {
  using namespace analyze_format_string;
  using namespace analyze_printf;
  // Now type check the data expression that matches the
  // format specifier.
  const analyze_printf::ArgType &AT = FS.getArgType(S.Context,
                                                    ObjCContext);
  if (!AT.isValid())
    return true;

  QualType IntendedTy = E->getType();
  if (AT.matchesType(S.Context, IntendedTy))
    return true;

  // Look through argument promotions for our error message's reported type.
  // This includes the integral and floating promotions, but excludes array
  // and function pointer decay; seeing that an argument intended to be a
  // string has type 'char [6]' is probably more confusing than 'char *'.
  if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_IntegralCast ||
        ICE->getCastKind() == CK_FloatingCast) {
      E = ICE->getSubExpr();
      IntendedTy = E->getType();

      // Check if we didn't match because of an implicit cast from a 'char'
      // or 'short' to an 'int'.  This is done because printf is a varargs
      // function.
      if (ICE->getType() == S.Context.IntTy ||
          ICE->getType() == S.Context.UnsignedIntTy) {
        // All further checking is done on the subexpression.
        if (AT.matchesType(S.Context, IntendedTy))
          return true;
      }
    }
  }

  if (S.Context.getTargetInfo().getTriple().isOSDarwin()) {
    // Special-case some of Darwin's platform-independence types.
    if (const TypedefType *UserTy = IntendedTy->getAs<TypedefType>()) {
      StringRef Name = UserTy->getDecl()->getName();
      IntendedTy = llvm::StringSwitch<QualType>(Name)
        .Case("NSInteger", S.Context.LongTy)
        .Case("NSUInteger", S.Context.UnsignedLongTy)
        .Case("SInt32", S.Context.IntTy)
        .Case("UInt32", S.Context.UnsignedIntTy)
        .Default(IntendedTy);
    }
  }

  // We may be able to offer a FixItHint if it is a supported type.
  PrintfSpecifier fixedFS = FS;
  bool success = fixedFS.fixType(IntendedTy, S.getLangOpts(),
                                 S.Context, ObjCContext);

  if (success) {
    // Get the fix string from the fixed format specifier
    SmallString<16> buf;
    llvm::raw_svector_ostream os(buf);
    fixedFS.toString(os);

    CharSourceRange SpecRange = getSpecifierRange(StartSpecifier, SpecifierLen);

    if (IntendedTy != E->getType()) {
      // The canonical type for formatting this value is different from the
      // actual type of the expression. (This occurs, for example, with Darwin's
      // NSInteger on 32-bit platforms, where it is typedef'd as 'int', but
      // should be printed as 'long' for 64-bit compatibility.)
      // Rather than emitting a normal format/argument mismatch, we want to
      // add a cast to the recommended type (and correct the format string
      // if necessary).
      SmallString<16> CastBuf;
      llvm::raw_svector_ostream CastFix(CastBuf);
      CastFix << "(";
      IntendedTy.print(CastFix, S.Context.getPrintingPolicy());
      CastFix << ")";

      SmallVector<FixItHint,4> Hints;
      if (!AT.matchesType(S.Context, IntendedTy))
        Hints.push_back(FixItHint::CreateReplacement(SpecRange, os.str()));

      if (const CStyleCastExpr *CCast = dyn_cast<CStyleCastExpr>(E)) {
        // If there's already a cast present, just replace it.
        SourceRange CastRange(CCast->getLParenLoc(), CCast->getRParenLoc());
        Hints.push_back(FixItHint::CreateReplacement(CastRange, CastFix.str()));

      } else if (!requiresParensToAddCast(E)) {
        // If the expression has high enough precedence,
        // just write the C-style cast.
        Hints.push_back(FixItHint::CreateInsertion(E->getLocStart(),
                                                   CastFix.str()));
      } else {
        // Otherwise, add parens around the expression as well as the cast.
        CastFix << "(";
        Hints.push_back(FixItHint::CreateInsertion(E->getLocStart(),
                                                   CastFix.str()));

        SourceLocation After = S.PP.getLocForEndOfToken(E->getLocEnd());
        Hints.push_back(FixItHint::CreateInsertion(After, ")"));
      }

      // We extract the name from the typedef because we don't want to show
      // the underlying type in the diagnostic.
      const TypedefType *UserTy = cast<TypedefType>(E->getType());
      StringRef Name = UserTy->getDecl()->getName();

      // Finally, emit the diagnostic.
      EmitFormatDiagnostic(S.PDiag(diag::warn_format_argument_needs_cast)
                             << Name << IntendedTy
                             << E->getSourceRange(),
                           E->getLocStart(), /*IsStringLocation=*/false,
                           SpecRange, Hints);
    } else {
      EmitFormatDiagnostic(
        S.PDiag(diag::warn_printf_conversion_argument_type_mismatch)
          << AT.getRepresentativeTypeName(S.Context) << IntendedTy
          << E->getSourceRange(),
        E->getLocStart(),
        /*IsStringLocation*/false,
        SpecRange,
        FixItHint::CreateReplacement(SpecRange, os.str()));
    }
  } else {
    const CharSourceRange &CSR = getSpecifierRange(StartSpecifier,
                                                   SpecifierLen);
    // Since the warning for passing non-POD types to variadic functions
    // was deferred until now, we emit a warning for non-POD
    // arguments here.
    if (S.isValidVarArgType(E->getType()) == Sema::VAK_Invalid) {
      unsigned DiagKind;
      if (E->getType()->isObjCObjectType())
        DiagKind = diag::err_cannot_pass_objc_interface_to_vararg_format;
      else
        DiagKind = diag::warn_non_pod_vararg_with_format_string;

      EmitFormatDiagnostic(
        S.PDiag(DiagKind)
          << S.getLangOpts().CPlusPlus0x
          << E->getType()
          << CallType
          << AT.getRepresentativeTypeName(S.Context)
          << CSR
          << E->getSourceRange(),
        E->getLocStart(), /*IsStringLocation*/false, CSR);

      checkForCStrMembers(AT, E, CSR);
    } else
      EmitFormatDiagnostic(
        S.PDiag(diag::warn_printf_conversion_argument_type_mismatch)
          << AT.getRepresentativeTypeName(S.Context) << E->getType()
          << CSR
          << E->getSourceRange(),
        E->getLocStart(), /*IsStringLocation*/false, CSR);
  }

  return true;
}

//===--- CHECK: Scanf format string checking ------------------------------===//

namespace {  
class CheckScanfHandler : public CheckFormatHandler {
public:
  CheckScanfHandler(Sema &s, const StringLiteral *fexpr,
                    const Expr *origFormatExpr, unsigned firstDataArg,
                    unsigned numDataArgs, const char *beg, bool hasVAListArg,
                    Expr **Args, unsigned NumArgs,
                    unsigned formatIdx, bool inFunctionCall,
                    Sema::VariadicCallType CallType)
  : CheckFormatHandler(s, fexpr, origFormatExpr, firstDataArg,
                       numDataArgs, beg, hasVAListArg,
                       Args, NumArgs, formatIdx, inFunctionCall, CallType)
  {}
  
  bool HandleScanfSpecifier(const analyze_scanf::ScanfSpecifier &FS,
                            const char *startSpecifier,
                            unsigned specifierLen);
  
  bool HandleInvalidScanfConversionSpecifier(
          const analyze_scanf::ScanfSpecifier &FS,
          const char *startSpecifier,
          unsigned specifierLen);

  void HandleIncompleteScanList(const char *start, const char *end);
};
}

void CheckScanfHandler::HandleIncompleteScanList(const char *start,
                                                 const char *end) {
  EmitFormatDiagnostic(S.PDiag(diag::warn_scanf_scanlist_incomplete),
                       getLocationOfByte(end), /*IsStringLocation*/true,
                       getSpecifierRange(start, end - start));
}

bool CheckScanfHandler::HandleInvalidScanfConversionSpecifier(
                                        const analyze_scanf::ScanfSpecifier &FS,
                                        const char *startSpecifier,
                                        unsigned specifierLen) {

  const analyze_scanf::ScanfConversionSpecifier &CS =
    FS.getConversionSpecifier();

  return HandleInvalidConversionSpecifier(FS.getArgIndex(),
                                          getLocationOfByte(CS.getStart()),
                                          startSpecifier, specifierLen,
                                          CS.getStart(), CS.getLength());
}

bool CheckScanfHandler::HandleScanfSpecifier(
                                       const analyze_scanf::ScanfSpecifier &FS,
                                       const char *startSpecifier,
                                       unsigned specifierLen) {
  
  using namespace analyze_scanf;
  using namespace analyze_format_string;  

  const ScanfConversionSpecifier &CS = FS.getConversionSpecifier();

  // Handle case where '%' and '*' don't consume an argument.  These shouldn't
  // be used to decide if we are using positional arguments consistently.
  if (FS.consumesDataArgument()) {
    if (atFirstArg) {
      atFirstArg = false;
      usesPositionalArgs = FS.usesPositionalArg();
    }
    else if (usesPositionalArgs != FS.usesPositionalArg()) {
      HandlePositionalNonpositionalArgs(getLocationOfByte(CS.getStart()),
                                        startSpecifier, specifierLen);
      return false;
    }
  }
  
  // Check if the field with is non-zero.
  const OptionalAmount &Amt = FS.getFieldWidth();
  if (Amt.getHowSpecified() == OptionalAmount::Constant) {
    if (Amt.getConstantAmount() == 0) {
      const CharSourceRange &R = getSpecifierRange(Amt.getStart(),
                                                   Amt.getConstantLength());
      EmitFormatDiagnostic(S.PDiag(diag::warn_scanf_nonzero_width),
                           getLocationOfByte(Amt.getStart()),
                           /*IsStringLocation*/true, R,
                           FixItHint::CreateRemoval(R));
    }
  }
  
  if (!FS.consumesDataArgument()) {
    // FIXME: Technically specifying a precision or field width here
    // makes no sense.  Worth issuing a warning at some point.
    return true;
  }
  
  // Consume the argument.
  unsigned argIndex = FS.getArgIndex();
  if (argIndex < NumDataArgs) {
      // The check to see if the argIndex is valid will come later.
      // We set the bit here because we may exit early from this
      // function if we encounter some other error.
    CoveredArgs.set(argIndex);
  }
  
  // Check the length modifier is valid with the given conversion specifier.
  if (!FS.hasValidLengthModifier(S.getASTContext().getTargetInfo()))
    HandleInvalidLengthModifier(FS, CS, startSpecifier, specifierLen,
                                diag::warn_format_nonsensical_length);
  else if (!FS.hasStandardLengthModifier())
    HandleNonStandardLengthModifier(FS, startSpecifier, specifierLen);
  else if (!FS.hasStandardLengthConversionCombination())
    HandleInvalidLengthModifier(FS, CS, startSpecifier, specifierLen,
                                diag::warn_format_non_standard_conversion_spec);

  if (!FS.hasStandardConversionSpecifier(S.getLangOpts()))
    HandleNonStandardConversionSpecifier(CS, startSpecifier, specifierLen);

  // The remaining checks depend on the data arguments.
  if (HasVAListArg)
    return true;
  
  if (!CheckNumArgs(FS, CS, startSpecifier, specifierLen, argIndex))
    return false;
  
  // Check that the argument type matches the format specifier.
  const Expr *Ex = getDataArg(argIndex);
  if (!Ex)
    return true;

  const analyze_format_string::ArgType &AT = FS.getArgType(S.Context);
  if (AT.isValid() && !AT.matchesType(S.Context, Ex->getType())) {
    ScanfSpecifier fixedFS = FS;
    bool success = fixedFS.fixType(Ex->getType(), S.getLangOpts(),
                                   S.Context);

    if (success) {
      // Get the fix string from the fixed format specifier.
      SmallString<128> buf;
      llvm::raw_svector_ostream os(buf);
      fixedFS.toString(os);

      EmitFormatDiagnostic(
        S.PDiag(diag::warn_printf_conversion_argument_type_mismatch)
          << AT.getRepresentativeTypeName(S.Context) << Ex->getType()
          << Ex->getSourceRange(),
        Ex->getLocStart(),
        /*IsStringLocation*/false,
        getSpecifierRange(startSpecifier, specifierLen),
        FixItHint::CreateReplacement(
          getSpecifierRange(startSpecifier, specifierLen),
          os.str()));
    } else {
      EmitFormatDiagnostic(
        S.PDiag(diag::warn_printf_conversion_argument_type_mismatch)
          << AT.getRepresentativeTypeName(S.Context) << Ex->getType()
          << Ex->getSourceRange(),
        Ex->getLocStart(),
        /*IsStringLocation*/false,
        getSpecifierRange(startSpecifier, specifierLen));
    }
  }

  return true;
}

void Sema::CheckFormatString(const StringLiteral *FExpr,
                             const Expr *OrigFormatExpr,
                             Expr **Args, unsigned NumArgs,
                             bool HasVAListArg, unsigned format_idx,
                             unsigned firstDataArg, FormatStringType Type,
                             bool inFunctionCall, VariadicCallType CallType) {
  
  // CHECK: is the format string a wide literal?
  if (!FExpr->isAscii() && !FExpr->isUTF8()) {
    CheckFormatHandler::EmitFormatDiagnostic(
      *this, inFunctionCall, Args[format_idx],
      PDiag(diag::warn_format_string_is_wide_literal), FExpr->getLocStart(),
      /*IsStringLocation*/true, OrigFormatExpr->getSourceRange());
    return;
  }
  
  // Str - The format string.  NOTE: this is NOT null-terminated!
  StringRef StrRef = FExpr->getString();
  const char *Str = StrRef.data();
  unsigned StrLen = StrRef.size();
  const unsigned numDataArgs = NumArgs - firstDataArg;
  
  // CHECK: empty format string?
  if (StrLen == 0 && numDataArgs > 0) {
    CheckFormatHandler::EmitFormatDiagnostic(
      *this, inFunctionCall, Args[format_idx],
      PDiag(diag::warn_empty_format_string), FExpr->getLocStart(),
      /*IsStringLocation*/true, OrigFormatExpr->getSourceRange());
    return;
  }
  
  if (Type == FST_Printf || Type == FST_NSString) {
    CheckPrintfHandler H(*this, FExpr, OrigFormatExpr, firstDataArg,
                         numDataArgs, (Type == FST_NSString),
                         Str, HasVAListArg, Args, NumArgs, format_idx,
                         inFunctionCall, CallType);
  
    if (!analyze_format_string::ParsePrintfString(H, Str, Str + StrLen,
                                                  getLangOpts(),
                                                  Context.getTargetInfo()))
      H.DoneProcessing();
  } else if (Type == FST_Scanf) {
    CheckScanfHandler H(*this, FExpr, OrigFormatExpr, firstDataArg, numDataArgs,
                        Str, HasVAListArg, Args, NumArgs, format_idx,
                        inFunctionCall, CallType);
    
    if (!analyze_format_string::ParseScanfString(H, Str, Str + StrLen,
                                                 getLangOpts(),
                                                 Context.getTargetInfo()))
      H.DoneProcessing();
  } // TODO: handle other formats
}

//===--- CHECK: Standard memory functions ---------------------------------===//

/// \brief Determine whether the given type is a dynamic class type (e.g.,
/// whether it has a vtable).
static bool isDynamicClassType(QualType T) {
  if (CXXRecordDecl *Record = T->getAsCXXRecordDecl())
    if (CXXRecordDecl *Definition = Record->getDefinition())
      if (Definition->isDynamicClass())
        return true;
  
  return false;
}

/// \brief If E is a sizeof expression, returns its argument expression,
/// otherwise returns NULL.
static const Expr *getSizeOfExprArg(const Expr* E) {
  if (const UnaryExprOrTypeTraitExpr *SizeOf =
      dyn_cast<UnaryExprOrTypeTraitExpr>(E))
    if (SizeOf->getKind() == clang::UETT_SizeOf && !SizeOf->isArgumentType())
      return SizeOf->getArgumentExpr()->IgnoreParenImpCasts();

  return 0;
}

/// \brief If E is a sizeof expression, returns its argument type.
static QualType getSizeOfArgType(const Expr* E) {
  if (const UnaryExprOrTypeTraitExpr *SizeOf =
      dyn_cast<UnaryExprOrTypeTraitExpr>(E))
    if (SizeOf->getKind() == clang::UETT_SizeOf)
      return SizeOf->getTypeOfArgument();

  return QualType();
}

/// \brief Check for dangerous or invalid arguments to memset().
///
/// This issues warnings on known problematic, dangerous or unspecified
/// arguments to the standard 'memset', 'memcpy', 'memmove', and 'memcmp'
/// function calls.
///
/// \param Call The call expression to diagnose.
void Sema::CheckMemaccessArguments(const CallExpr *Call,
                                   unsigned BId,
                                   IdentifierInfo *FnName) {
  assert(BId != 0);

  // It is possible to have a non-standard definition of memset.  Validate
  // we have enough arguments, and if not, abort further checking.
  unsigned ExpectedNumArgs = (BId == Builtin::BIstrndup ? 2 : 3);
  if (Call->getNumArgs() < ExpectedNumArgs)
    return;

  unsigned LastArg = (BId == Builtin::BImemset ||
                      BId == Builtin::BIstrndup ? 1 : 2);
  unsigned LenArg = (BId == Builtin::BIstrndup ? 1 : 2);
  const Expr *LenExpr = Call->getArg(LenArg)->IgnoreParenImpCasts();

  // We have special checking when the length is a sizeof expression.
  QualType SizeOfArgTy = getSizeOfArgType(LenExpr);
  const Expr *SizeOfArg = getSizeOfExprArg(LenExpr);
  llvm::FoldingSetNodeID SizeOfArgID;

  for (unsigned ArgIdx = 0; ArgIdx != LastArg; ++ArgIdx) {
    const Expr *Dest = Call->getArg(ArgIdx)->IgnoreParenImpCasts();
    SourceRange ArgRange = Call->getArg(ArgIdx)->getSourceRange();

    QualType DestTy = Dest->getType();
    if (const PointerType *DestPtrTy = DestTy->getAs<PointerType>()) {
      QualType PointeeTy = DestPtrTy->getPointeeType();

      // Never warn about void type pointers. This can be used to suppress
      // false positives.
      if (PointeeTy->isVoidType())
        continue;

      // Catch "memset(p, 0, sizeof(p))" -- needs to be sizeof(*p). Do this by
      // actually comparing the expressions for equality. Because computing the
      // expression IDs can be expensive, we only do this if the diagnostic is
      // enabled.
      if (SizeOfArg &&
          Diags.getDiagnosticLevel(diag::warn_sizeof_pointer_expr_memaccess,
                                   SizeOfArg->getExprLoc())) {
        // We only compute IDs for expressions if the warning is enabled, and
        // cache the sizeof arg's ID.
        if (SizeOfArgID == llvm::FoldingSetNodeID())
          SizeOfArg->Profile(SizeOfArgID, Context, true);
        llvm::FoldingSetNodeID DestID;
        Dest->Profile(DestID, Context, true);
        if (DestID == SizeOfArgID) {
          // TODO: For strncpy() and friends, this could suggest sizeof(dst)
          //       over sizeof(src) as well.
          unsigned ActionIdx = 0; // Default is to suggest dereferencing.
          StringRef ReadableName = FnName->getName();

          if (const UnaryOperator *UnaryOp = dyn_cast<UnaryOperator>(Dest))
            if (UnaryOp->getOpcode() == UO_AddrOf)
              ActionIdx = 1; // If its an address-of operator, just remove it.
          if (Context.getTypeSize(PointeeTy) == Context.getCharWidth())
            ActionIdx = 2; // If the pointee's size is sizeof(char),
                           // suggest an explicit length.

          // If the function is defined as a builtin macro, do not show macro
          // expansion.
          SourceLocation SL = SizeOfArg->getExprLoc();
          SourceRange DSR = Dest->getSourceRange();
          SourceRange SSR = SizeOfArg->getSourceRange();
          SourceManager &SM  = PP.getSourceManager();

          if (SM.isMacroArgExpansion(SL)) {
            ReadableName = Lexer::getImmediateMacroName(SL, SM, LangOpts);
            SL = SM.getSpellingLoc(SL);
            DSR = SourceRange(SM.getSpellingLoc(DSR.getBegin()),
                             SM.getSpellingLoc(DSR.getEnd()));
            SSR = SourceRange(SM.getSpellingLoc(SSR.getBegin()),
                             SM.getSpellingLoc(SSR.getEnd()));
          }

          DiagRuntimeBehavior(SL, SizeOfArg,
                              PDiag(diag::warn_sizeof_pointer_expr_memaccess)
                                << ReadableName
                                << PointeeTy
                                << DestTy
                                << DSR
                                << SSR);
          DiagRuntimeBehavior(SL, SizeOfArg,
                         PDiag(diag::warn_sizeof_pointer_expr_memaccess_note)
                                << ActionIdx
                                << SSR);

          break;
        }
      }

      // Also check for cases where the sizeof argument is the exact same
      // type as the memory argument, and where it points to a user-defined
      // record type.
      if (SizeOfArgTy != QualType()) {
        if (PointeeTy->isRecordType() &&
            Context.typesAreCompatible(SizeOfArgTy, DestTy)) {
          DiagRuntimeBehavior(LenExpr->getExprLoc(), Dest,
                              PDiag(diag::warn_sizeof_pointer_type_memaccess)
                                << FnName << SizeOfArgTy << ArgIdx
                                << PointeeTy << Dest->getSourceRange()
                                << LenExpr->getSourceRange());
          break;
        }
      }

      // Always complain about dynamic classes.
      if (isDynamicClassType(PointeeTy)) {

        unsigned OperationType = 0;
        // "overwritten" if we're warning about the destination for any call
        // but memcmp; otherwise a verb appropriate to the call.
        if (ArgIdx != 0 || BId == Builtin::BImemcmp) {
          if (BId == Builtin::BImemcpy)
            OperationType = 1;
          else if(BId == Builtin::BImemmove)
            OperationType = 2;
          else if (BId == Builtin::BImemcmp)
            OperationType = 3;
        }
          
        DiagRuntimeBehavior(
          Dest->getExprLoc(), Dest,
          PDiag(diag::warn_dyn_class_memaccess)
            << (BId == Builtin::BImemcmp ? ArgIdx + 2 : ArgIdx)
            << FnName << PointeeTy
            << OperationType
            << Call->getCallee()->getSourceRange());
      } else if (PointeeTy.hasNonTrivialObjCLifetime() &&
               BId != Builtin::BImemset)
        DiagRuntimeBehavior(
          Dest->getExprLoc(), Dest,
          PDiag(diag::warn_arc_object_memaccess)
            << ArgIdx << FnName << PointeeTy
            << Call->getCallee()->getSourceRange());
      else
        continue;

      DiagRuntimeBehavior(
        Dest->getExprLoc(), Dest,
        PDiag(diag::note_bad_memaccess_silence)
          << FixItHint::CreateInsertion(ArgRange.getBegin(), "(void*)"));
      break;
    }
  }
}

// A little helper routine: ignore addition and subtraction of integer literals.
// This intentionally does not ignore all integer constant expressions because
// we don't want to remove sizeof().
static const Expr *ignoreLiteralAdditions(const Expr *Ex, ASTContext &Ctx) {
  Ex = Ex->IgnoreParenCasts();

  for (;;) {
    const BinaryOperator * BO = dyn_cast<BinaryOperator>(Ex);
    if (!BO || !BO->isAdditiveOp())
      break;

    const Expr *RHS = BO->getRHS()->IgnoreParenCasts();
    const Expr *LHS = BO->getLHS()->IgnoreParenCasts();
    
    if (isa<IntegerLiteral>(RHS))
      Ex = LHS;
    else if (isa<IntegerLiteral>(LHS))
      Ex = RHS;
    else
      break;
  }

  return Ex;
}

static bool isConstantSizeArrayWithMoreThanOneElement(QualType Ty,
                                                      ASTContext &Context) {
  // Only handle constant-sized or VLAs, but not flexible members.
  if (const ConstantArrayType *CAT = Context.getAsConstantArrayType(Ty)) {
    // Only issue the FIXIT for arrays of size > 1.
    if (CAT->getSize().getSExtValue() <= 1)
      return false;
  } else if (!Ty->isVariableArrayType()) {
    return false;
  }
  return true;
}

// Warn if the user has made the 'size' argument to strlcpy or strlcat
// be the size of the source, instead of the destination.
void Sema::CheckStrlcpycatArguments(const CallExpr *Call,
                                    IdentifierInfo *FnName) {

  // Don't crash if the user has the wrong number of arguments
  if (Call->getNumArgs() != 3)
    return;

  const Expr *SrcArg = ignoreLiteralAdditions(Call->getArg(1), Context);
  const Expr *SizeArg = ignoreLiteralAdditions(Call->getArg(2), Context);
  const Expr *CompareWithSrc = NULL;
  
  // Look for 'strlcpy(dst, x, sizeof(x))'
  if (const Expr *Ex = getSizeOfExprArg(SizeArg))
    CompareWithSrc = Ex;
  else {
    // Look for 'strlcpy(dst, x, strlen(x))'
    if (const CallExpr *SizeCall = dyn_cast<CallExpr>(SizeArg)) {
      if (SizeCall->isBuiltinCall() == Builtin::BIstrlen
          && SizeCall->getNumArgs() == 1)
        CompareWithSrc = ignoreLiteralAdditions(SizeCall->getArg(0), Context);
    }
  }

  if (!CompareWithSrc)
    return;

  // Determine if the argument to sizeof/strlen is equal to the source
  // argument.  In principle there's all kinds of things you could do
  // here, for instance creating an == expression and evaluating it with
  // EvaluateAsBooleanCondition, but this uses a more direct technique:
  const DeclRefExpr *SrcArgDRE = dyn_cast<DeclRefExpr>(SrcArg);
  if (!SrcArgDRE)
    return;
  
  const DeclRefExpr *CompareWithSrcDRE = dyn_cast<DeclRefExpr>(CompareWithSrc);
  if (!CompareWithSrcDRE || 
      SrcArgDRE->getDecl() != CompareWithSrcDRE->getDecl())
    return;
  
  const Expr *OriginalSizeArg = Call->getArg(2);
  Diag(CompareWithSrcDRE->getLocStart(), diag::warn_strlcpycat_wrong_size)
    << OriginalSizeArg->getSourceRange() << FnName;
  
  // Output a FIXIT hint if the destination is an array (rather than a
  // pointer to an array).  This could be enhanced to handle some
  // pointers if we know the actual size, like if DstArg is 'array+2'
  // we could say 'sizeof(array)-2'.
  const Expr *DstArg = Call->getArg(0)->IgnoreParenImpCasts();
  if (!isConstantSizeArrayWithMoreThanOneElement(DstArg->getType(), Context))
    return;

  SmallString<128> sizeString;
  llvm::raw_svector_ostream OS(sizeString);
  OS << "sizeof(";
  DstArg->printPretty(OS, 0, getPrintingPolicy());
  OS << ")";
  
  Diag(OriginalSizeArg->getLocStart(), diag::note_strlcpycat_wrong_size)
    << FixItHint::CreateReplacement(OriginalSizeArg->getSourceRange(),
                                    OS.str());
}

/// Check if two expressions refer to the same declaration.
static bool referToTheSameDecl(const Expr *E1, const Expr *E2) {
  if (const DeclRefExpr *D1 = dyn_cast_or_null<DeclRefExpr>(E1))
    if (const DeclRefExpr *D2 = dyn_cast_or_null<DeclRefExpr>(E2))
      return D1->getDecl() == D2->getDecl();
  return false;
}

static const Expr *getStrlenExprArg(const Expr *E) {
  if (const CallExpr *CE = dyn_cast<CallExpr>(E)) {
    const FunctionDecl *FD = CE->getDirectCallee();
    if (!FD || FD->getMemoryFunctionKind() != Builtin::BIstrlen)
      return 0;
    return CE->getArg(0)->IgnoreParenCasts();
  }
  return 0;
}

// Warn on anti-patterns as the 'size' argument to strncat.
// The correct size argument should look like following:
//   strncat(dst, src, sizeof(dst) - strlen(dest) - 1);
void Sema::CheckStrncatArguments(const CallExpr *CE,
                                 IdentifierInfo *FnName) {
  // Don't crash if the user has the wrong number of arguments.
  if (CE->getNumArgs() < 3)
    return;
  const Expr *DstArg = CE->getArg(0)->IgnoreParenCasts();
  const Expr *SrcArg = CE->getArg(1)->IgnoreParenCasts();
  const Expr *LenArg = CE->getArg(2)->IgnoreParenCasts();

  // Identify common expressions, which are wrongly used as the size argument
  // to strncat and may lead to buffer overflows.
  unsigned PatternType = 0;
  if (const Expr *SizeOfArg = getSizeOfExprArg(LenArg)) {
    // - sizeof(dst)
    if (referToTheSameDecl(SizeOfArg, DstArg))
      PatternType = 1;
    // - sizeof(src)
    else if (referToTheSameDecl(SizeOfArg, SrcArg))
      PatternType = 2;
  } else if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(LenArg)) {
    if (BE->getOpcode() == BO_Sub) {
      const Expr *L = BE->getLHS()->IgnoreParenCasts();
      const Expr *R = BE->getRHS()->IgnoreParenCasts();
      // - sizeof(dst) - strlen(dst)
      if (referToTheSameDecl(DstArg, getSizeOfExprArg(L)) &&
          referToTheSameDecl(DstArg, getStrlenExprArg(R)))
        PatternType = 1;
      // - sizeof(src) - (anything)
      else if (referToTheSameDecl(SrcArg, getSizeOfExprArg(L)))
        PatternType = 2;
    }
  }

  if (PatternType == 0)
    return;

  // Generate the diagnostic.
  SourceLocation SL = LenArg->getLocStart();
  SourceRange SR = LenArg->getSourceRange();
  SourceManager &SM  = PP.getSourceManager();

  // If the function is defined as a builtin macro, do not show macro expansion.
  if (SM.isMacroArgExpansion(SL)) {
    SL = SM.getSpellingLoc(SL);
    SR = SourceRange(SM.getSpellingLoc(SR.getBegin()),
                     SM.getSpellingLoc(SR.getEnd()));
  }

  // Check if the destination is an array (rather than a pointer to an array).
  QualType DstTy = DstArg->getType();
  bool isKnownSizeArray = isConstantSizeArrayWithMoreThanOneElement(DstTy,
                                                                    Context);
  if (!isKnownSizeArray) {
    if (PatternType == 1)
      Diag(SL, diag::warn_strncat_wrong_size) << SR;
    else
      Diag(SL, diag::warn_strncat_src_size) << SR;
    return;
  }

  if (PatternType == 1)
    Diag(SL, diag::warn_strncat_large_size) << SR;
  else
    Diag(SL, diag::warn_strncat_src_size) << SR;

  SmallString<128> sizeString;
  llvm::raw_svector_ostream OS(sizeString);
  OS << "sizeof(";
  DstArg->printPretty(OS, 0, getPrintingPolicy());
  OS << ") - ";
  OS << "strlen(";
  DstArg->printPretty(OS, 0, getPrintingPolicy());
  OS << ") - 1";

  Diag(SL, diag::note_strncat_wrong_size)
    << FixItHint::CreateReplacement(SR, OS.str());
}

//===--- CHECK: Return Address of Stack Variable --------------------------===//

static Expr *EvalVal(Expr *E, SmallVectorImpl<DeclRefExpr *> &refVars,
                     Decl *ParentDecl);
static Expr *EvalAddr(Expr* E, SmallVectorImpl<DeclRefExpr *> &refVars,
                      Decl *ParentDecl);

/// CheckReturnStackAddr - Check if a return statement returns the address
///   of a stack variable.
void
Sema::CheckReturnStackAddr(Expr *RetValExp, QualType lhsType,
                           SourceLocation ReturnLoc) {

  Expr *stackE = 0;
  SmallVector<DeclRefExpr *, 8> refVars;

  // Perform checking for returned stack addresses, local blocks,
  // label addresses or references to temporaries.
  if (lhsType->isPointerType() ||
      (!getLangOpts().ObjCAutoRefCount && lhsType->isBlockPointerType())) {
    stackE = EvalAddr(RetValExp, refVars, /*ParentDecl=*/0);
  } else if (lhsType->isReferenceType()) {
    stackE = EvalVal(RetValExp, refVars, /*ParentDecl=*/0);
  }

  if (stackE == 0)
    return; // Nothing suspicious was found.

  SourceLocation diagLoc;
  SourceRange diagRange;
  if (refVars.empty()) {
    diagLoc = stackE->getLocStart();
    diagRange = stackE->getSourceRange();
  } else {
    // We followed through a reference variable. 'stackE' contains the
    // problematic expression but we will warn at the return statement pointing
    // at the reference variable. We will later display the "trail" of
    // reference variables using notes.
    diagLoc = refVars[0]->getLocStart();
    diagRange = refVars[0]->getSourceRange();
  }

  if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(stackE)) { //address of local var.
    Diag(diagLoc, lhsType->isReferenceType() ? diag::warn_ret_stack_ref
                                             : diag::warn_ret_stack_addr)
     << DR->getDecl()->getDeclName() << diagRange;
  } else if (isa<BlockExpr>(stackE)) { // local block.
    Diag(diagLoc, diag::err_ret_local_block) << diagRange;
  } else if (isa<AddrLabelExpr>(stackE)) { // address of label.
    Diag(diagLoc, diag::warn_ret_addr_label) << diagRange;
  } else { // local temporary.
    Diag(diagLoc, lhsType->isReferenceType() ? diag::warn_ret_local_temp_ref
                                             : diag::warn_ret_local_temp_addr)
     << diagRange;
  }

  // Display the "trail" of reference variables that we followed until we
  // found the problematic expression using notes.
  for (unsigned i = 0, e = refVars.size(); i != e; ++i) {
    VarDecl *VD = cast<VarDecl>(refVars[i]->getDecl());
    // If this var binds to another reference var, show the range of the next
    // var, otherwise the var binds to the problematic expression, in which case
    // show the range of the expression.
    SourceRange range = (i < e-1) ? refVars[i+1]->getSourceRange()
                                  : stackE->getSourceRange();
    Diag(VD->getLocation(), diag::note_ref_var_local_bind)
      << VD->getDeclName() << range;
  }
}

/// EvalAddr - EvalAddr and EvalVal are mutually recursive functions that
///  check if the expression in a return statement evaluates to an address
///  to a location on the stack, a local block, an address of a label, or a
///  reference to local temporary. The recursion is used to traverse the
///  AST of the return expression, with recursion backtracking when we
///  encounter a subexpression that (1) clearly does not lead to one of the
///  above problematic expressions (2) is something we cannot determine leads to
///  a problematic expression based on such local checking.
///
///  Both EvalAddr and EvalVal follow through reference variables to evaluate
///  the expression that they point to. Such variables are added to the
///  'refVars' vector so that we know what the reference variable "trail" was.
///
///  EvalAddr processes expressions that are pointers that are used as
///  references (and not L-values).  EvalVal handles all other values.
///  At the base case of the recursion is a check for the above problematic
///  expressions.
///
///  This implementation handles:
///
///   * pointer-to-pointer casts
///   * implicit conversions from array references to pointers
///   * taking the address of fields
///   * arbitrary interplay between "&" and "*" operators
///   * pointer arithmetic from an address of a stack variable
///   * taking the address of an array element where the array is on the stack
static Expr *EvalAddr(Expr *E, SmallVectorImpl<DeclRefExpr *> &refVars,
                      Decl *ParentDecl) {
  if (E->isTypeDependent())
      return NULL;

  // We should only be called for evaluating pointer expressions.
  assert((E->getType()->isAnyPointerType() ||
          E->getType()->isBlockPointerType() ||
          E->getType()->isObjCQualifiedIdType()) &&
         "EvalAddr only works on pointers");

  E = E->IgnoreParens();

  // Our "symbolic interpreter" is just a dispatch off the currently
  // viewed AST node.  We then recursively traverse the AST by calling
  // EvalAddr and EvalVal appropriately.
  switch (E->getStmtClass()) {
  case Stmt::DeclRefExprClass: {
    DeclRefExpr *DR = cast<DeclRefExpr>(E);

    if (VarDecl *V = dyn_cast<VarDecl>(DR->getDecl()))
      // If this is a reference variable, follow through to the expression that
      // it points to.
      if (V->hasLocalStorage() &&
          V->getType()->isReferenceType() && V->hasInit()) {
        // Add the reference variable to the "trail".
        refVars.push_back(DR);
        return EvalAddr(V->getInit(), refVars, ParentDecl);
      }

    return NULL;
  }

  case Stmt::UnaryOperatorClass: {
    // The only unary operator that make sense to handle here
    // is AddrOf.  All others don't make sense as pointers.
    UnaryOperator *U = cast<UnaryOperator>(E);

    if (U->getOpcode() == UO_AddrOf)
      return EvalVal(U->getSubExpr(), refVars, ParentDecl);
    else
      return NULL;
  }

  case Stmt::BinaryOperatorClass: {
    // Handle pointer arithmetic.  All other binary operators are not valid
    // in this context.
    BinaryOperator *B = cast<BinaryOperator>(E);
    BinaryOperatorKind op = B->getOpcode();

    if (op != BO_Add && op != BO_Sub)
      return NULL;

    Expr *Base = B->getLHS();

    // Determine which argument is the real pointer base.  It could be
    // the RHS argument instead of the LHS.
    if (!Base->getType()->isPointerType()) Base = B->getRHS();

    assert (Base->getType()->isPointerType());
    return EvalAddr(Base, refVars, ParentDecl);
  }

  // For conditional operators we need to see if either the LHS or RHS are
  // valid DeclRefExpr*s.  If one of them is valid, we return it.
  case Stmt::ConditionalOperatorClass: {
    ConditionalOperator *C = cast<ConditionalOperator>(E);

    // Handle the GNU extension for missing LHS.
    if (Expr *lhsExpr = C->getLHS()) {
    // In C++, we can have a throw-expression, which has 'void' type.
      if (!lhsExpr->getType()->isVoidType())
        if (Expr* LHS = EvalAddr(lhsExpr, refVars, ParentDecl))
          return LHS;
    }

    // In C++, we can have a throw-expression, which has 'void' type.
    if (C->getRHS()->getType()->isVoidType())
      return NULL;

    return EvalAddr(C->getRHS(), refVars, ParentDecl);
  }
  
  case Stmt::BlockExprClass:
    if (cast<BlockExpr>(E)->getBlockDecl()->hasCaptures())
      return E; // local block.
    return NULL;

  case Stmt::AddrLabelExprClass:
    return E; // address of label.

  case Stmt::ExprWithCleanupsClass:
    return EvalAddr(cast<ExprWithCleanups>(E)->getSubExpr(), refVars,
                    ParentDecl);

  // For casts, we need to handle conversions from arrays to
  // pointer values, and pointer-to-pointer conversions.
  case Stmt::ImplicitCastExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::CXXFunctionalCastExprClass:
  case Stmt::ObjCBridgedCastExprClass:
  case Stmt::CXXStaticCastExprClass:
  case Stmt::CXXDynamicCastExprClass:
  case Stmt::CXXConstCastExprClass:
  case Stmt::CXXReinterpretCastExprClass: {
    Expr* SubExpr = cast<CastExpr>(E)->getSubExpr();
    switch (cast<CastExpr>(E)->getCastKind()) {
    case CK_BitCast:
    case CK_LValueToRValue:
    case CK_NoOp:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_Dynamic:
    case CK_CPointerToObjCPointerCast:
    case CK_BlockPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
      return EvalAddr(SubExpr, refVars, ParentDecl);

    case CK_ArrayToPointerDecay:
      return EvalVal(SubExpr, refVars, ParentDecl);

    default:
      return 0;
    }
  }

  case Stmt::MaterializeTemporaryExprClass:
    if (Expr *Result = EvalAddr(
                         cast<MaterializeTemporaryExpr>(E)->GetTemporaryExpr(),
                                refVars, ParentDecl))
      return Result;
      
    return E;
      
  // Everything else: we simply don't reason about them.
  default:
    return NULL;
  }
}


///  EvalVal - This function is complements EvalAddr in the mutual recursion.
///   See the comments for EvalAddr for more details.
static Expr *EvalVal(Expr *E, SmallVectorImpl<DeclRefExpr *> &refVars,
                     Decl *ParentDecl) {
do {
  // We should only be called for evaluating non-pointer expressions, or
  // expressions with a pointer type that are not used as references but instead
  // are l-values (e.g., DeclRefExpr with a pointer type).

  // Our "symbolic interpreter" is just a dispatch off the currently
  // viewed AST node.  We then recursively traverse the AST by calling
  // EvalAddr and EvalVal appropriately.

  E = E->IgnoreParens();
  switch (E->getStmtClass()) {
  case Stmt::ImplicitCastExprClass: {
    ImplicitCastExpr *IE = cast<ImplicitCastExpr>(E);
    if (IE->getValueKind() == VK_LValue) {
      E = IE->getSubExpr();
      continue;
    }
    return NULL;
  }

  case Stmt::ExprWithCleanupsClass:
    return EvalVal(cast<ExprWithCleanups>(E)->getSubExpr(), refVars,ParentDecl);

  case Stmt::DeclRefExprClass: {
    // When we hit a DeclRefExpr we are looking at code that refers to a
    // variable's name. If it's not a reference variable we check if it has
    // local storage within the function, and if so, return the expression.
    DeclRefExpr *DR = cast<DeclRefExpr>(E);

    if (VarDecl *V = dyn_cast<VarDecl>(DR->getDecl())) {
      // Check if it refers to itself, e.g. "int& i = i;".
      if (V == ParentDecl)
        return DR;

      if (V->hasLocalStorage()) {
        if (!V->getType()->isReferenceType())
          return DR;

        // Reference variable, follow through to the expression that
        // it points to.
        if (V->hasInit()) {
          // Add the reference variable to the "trail".
          refVars.push_back(DR);
          return EvalVal(V->getInit(), refVars, V);
        }
      }
    }

    return NULL;
  }

  case Stmt::UnaryOperatorClass: {
    // The only unary operator that make sense to handle here
    // is Deref.  All others don't resolve to a "name."  This includes
    // handling all sorts of rvalues passed to a unary operator.
    UnaryOperator *U = cast<UnaryOperator>(E);

    if (U->getOpcode() == UO_Deref)
      return EvalAddr(U->getSubExpr(), refVars, ParentDecl);

    return NULL;
  }

  case Stmt::ArraySubscriptExprClass: {
    // Array subscripts are potential references to data on the stack.  We
    // retrieve the DeclRefExpr* for the array variable if it indeed
    // has local storage.
    return EvalAddr(cast<ArraySubscriptExpr>(E)->getBase(), refVars,ParentDecl);
  }

  case Stmt::ConditionalOperatorClass: {
    // For conditional operators we need to see if either the LHS or RHS are
    // non-NULL Expr's.  If one is non-NULL, we return it.
    ConditionalOperator *C = cast<ConditionalOperator>(E);

    // Handle the GNU extension for missing LHS.
    if (Expr *lhsExpr = C->getLHS())
      if (Expr *LHS = EvalVal(lhsExpr, refVars, ParentDecl))
        return LHS;

    return EvalVal(C->getRHS(), refVars, ParentDecl);
  }

  // Accesses to members are potential references to data on the stack.
  case Stmt::MemberExprClass: {
    MemberExpr *M = cast<MemberExpr>(E);

    // Check for indirect access.  We only want direct field accesses.
    if (M->isArrow())
      return NULL;

    // Check whether the member type is itself a reference, in which case
    // we're not going to refer to the member, but to what the member refers to.
    if (M->getMemberDecl()->getType()->isReferenceType())
      return NULL;

    return EvalVal(M->getBase(), refVars, ParentDecl);
  }

  case Stmt::MaterializeTemporaryExprClass:
    if (Expr *Result = EvalVal(
                          cast<MaterializeTemporaryExpr>(E)->GetTemporaryExpr(),
                               refVars, ParentDecl))
      return Result;
      
    return E;

  default:
    // Check that we don't return or take the address of a reference to a
    // temporary. This is only useful in C++.
    if (!E->isTypeDependent() && E->isRValue())
      return E;

    // Everything else: we simply don't reason about them.
    return NULL;
  }
} while (true);
}

//===--- CHECK: Floating-Point comparisons (-Wfloat-equal) ---------------===//

/// Check for comparisons of floating point operands using != and ==.
/// Issue a warning if these are no self-comparisons, as they are not likely
/// to do what the programmer intended.
void Sema::CheckFloatComparison(SourceLocation Loc, Expr* LHS, Expr *RHS) {
  Expr* LeftExprSansParen = LHS->IgnoreParenImpCasts();
  Expr* RightExprSansParen = RHS->IgnoreParenImpCasts();

  // Special case: check for x == x (which is OK).
  // Do not emit warnings for such cases.
  if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(LeftExprSansParen))
    if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(RightExprSansParen))
      if (DRL->getDecl() == DRR->getDecl())
        return;


  // Special case: check for comparisons against literals that can be exactly
  //  represented by APFloat.  In such cases, do not emit a warning.  This
  //  is a heuristic: often comparison against such literals are used to
  //  detect if a value in a variable has not changed.  This clearly can
  //  lead to false negatives.
  if (FloatingLiteral* FLL = dyn_cast<FloatingLiteral>(LeftExprSansParen)) {
    if (FLL->isExact())
      return;
  } else
    if (FloatingLiteral* FLR = dyn_cast<FloatingLiteral>(RightExprSansParen))
      if (FLR->isExact())
        return;

  // Check for comparisons with builtin types.
  if (CallExpr* CL = dyn_cast<CallExpr>(LeftExprSansParen))
    if (CL->isBuiltinCall())
      return;

  if (CallExpr* CR = dyn_cast<CallExpr>(RightExprSansParen))
    if (CR->isBuiltinCall())
      return;

  // Emit the diagnostic.
  Diag(Loc, diag::warn_floatingpoint_eq)
    << LHS->getSourceRange() << RHS->getSourceRange();
}

//===--- CHECK: Integer mixed-sign comparisons (-Wsign-compare) --------===//
//===--- CHECK: Lossy implicit conversions (-Wconversion) --------------===//

namespace {

/// Structure recording the 'active' range of an integer-valued
/// expression.
struct IntRange {
  /// The number of bits active in the int.
  unsigned Width;

  /// True if the int is known not to have negative values.
  bool NonNegative;

  IntRange(unsigned Width, bool NonNegative)
    : Width(Width), NonNegative(NonNegative)
  {}

  /// Returns the range of the bool type.
  static IntRange forBoolType() {
    return IntRange(1, true);
  }

  /// Returns the range of an opaque value of the given integral type.
  static IntRange forValueOfType(ASTContext &C, QualType T) {
    return forValueOfCanonicalType(C,
                          T->getCanonicalTypeInternal().getTypePtr());
  }

  /// Returns the range of an opaque value of a canonical integral type.
  static IntRange forValueOfCanonicalType(ASTContext &C, const Type *T) {
    assert(T->isCanonicalUnqualified());

    if (const VectorType *VT = dyn_cast<VectorType>(T))
      T = VT->getElementType().getTypePtr();
    if (const ComplexType *CT = dyn_cast<ComplexType>(T))
      T = CT->getElementType().getTypePtr();

    // For enum types, use the known bit width of the enumerators.
    if (const EnumType *ET = dyn_cast<EnumType>(T)) {
      EnumDecl *Enum = ET->getDecl();
      if (!Enum->isCompleteDefinition())
        return IntRange(C.getIntWidth(QualType(T, 0)), false);

      unsigned NumPositive = Enum->getNumPositiveBits();
      unsigned NumNegative = Enum->getNumNegativeBits();

      return IntRange(std::max(NumPositive, NumNegative), NumNegative == 0);
    }

    const BuiltinType *BT = cast<BuiltinType>(T);
    assert(BT->isInteger());

    return IntRange(C.getIntWidth(QualType(T, 0)), BT->isUnsignedInteger());
  }

  /// Returns the "target" range of a canonical integral type, i.e.
  /// the range of values expressible in the type.
  ///
  /// This matches forValueOfCanonicalType except that enums have the
  /// full range of their type, not the range of their enumerators.
  static IntRange forTargetOfCanonicalType(ASTContext &C, const Type *T) {
    assert(T->isCanonicalUnqualified());

    if (const VectorType *VT = dyn_cast<VectorType>(T))
      T = VT->getElementType().getTypePtr();
    if (const ComplexType *CT = dyn_cast<ComplexType>(T))
      T = CT->getElementType().getTypePtr();
    if (const EnumType *ET = dyn_cast<EnumType>(T))
      T = C.getCanonicalType(ET->getDecl()->getIntegerType()).getTypePtr();

    const BuiltinType *BT = cast<BuiltinType>(T);
    assert(BT->isInteger());

    return IntRange(C.getIntWidth(QualType(T, 0)), BT->isUnsignedInteger());
  }

  /// Returns the supremum of two ranges: i.e. their conservative merge.
  static IntRange join(IntRange L, IntRange R) {
    return IntRange(std::max(L.Width, R.Width),
                    L.NonNegative && R.NonNegative);
  }

  /// Returns the infinum of two ranges: i.e. their aggressive merge.
  static IntRange meet(IntRange L, IntRange R) {
    return IntRange(std::min(L.Width, R.Width),
                    L.NonNegative || R.NonNegative);
  }
};

static IntRange GetValueRange(ASTContext &C, llvm::APSInt &value,
                              unsigned MaxWidth) {
  if (value.isSigned() && value.isNegative())
    return IntRange(value.getMinSignedBits(), false);

  if (value.getBitWidth() > MaxWidth)
    value = value.trunc(MaxWidth);

  // isNonNegative() just checks the sign bit without considering
  // signedness.
  return IntRange(value.getActiveBits(), true);
}

static IntRange GetValueRange(ASTContext &C, APValue &result, QualType Ty,
                              unsigned MaxWidth) {
  if (result.isInt())
    return GetValueRange(C, result.getInt(), MaxWidth);

  if (result.isVector()) {
    IntRange R = GetValueRange(C, result.getVectorElt(0), Ty, MaxWidth);
    for (unsigned i = 1, e = result.getVectorLength(); i != e; ++i) {
      IntRange El = GetValueRange(C, result.getVectorElt(i), Ty, MaxWidth);
      R = IntRange::join(R, El);
    }
    return R;
  }

  if (result.isComplexInt()) {
    IntRange R = GetValueRange(C, result.getComplexIntReal(), MaxWidth);
    IntRange I = GetValueRange(C, result.getComplexIntImag(), MaxWidth);
    return IntRange::join(R, I);
  }

  // This can happen with lossless casts to intptr_t of "based" lvalues.
  // Assume it might use arbitrary bits.
  // FIXME: The only reason we need to pass the type in here is to get
  // the sign right on this one case.  It would be nice if APValue
  // preserved this.
  assert(result.isLValue() || result.isAddrLabelDiff());
  return IntRange(MaxWidth, Ty->isUnsignedIntegerOrEnumerationType());
}

/// Pseudo-evaluate the given integer expression, estimating the
/// range of values it might take.
///
/// \param MaxWidth - the width to which the value will be truncated
static IntRange GetExprRange(ASTContext &C, Expr *E, unsigned MaxWidth) {
  E = E->IgnoreParens();

  // Try a full evaluation first.
  Expr::EvalResult result;
  if (E->EvaluateAsRValue(result, C))
    return GetValueRange(C, result.Val, E->getType(), MaxWidth);

  // I think we only want to look through implicit casts here; if the
  // user has an explicit widening cast, we should treat the value as
  // being of the new, wider type.
  if (ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E)) {
    if (CE->getCastKind() == CK_NoOp || CE->getCastKind() == CK_LValueToRValue)
      return GetExprRange(C, CE->getSubExpr(), MaxWidth);

    IntRange OutputTypeRange = IntRange::forValueOfType(C, CE->getType());

    bool isIntegerCast = (CE->getCastKind() == CK_IntegralCast);

    // Assume that non-integer casts can span the full range of the type.
    if (!isIntegerCast)
      return OutputTypeRange;

    IntRange SubRange
      = GetExprRange(C, CE->getSubExpr(),
                     std::min(MaxWidth, OutputTypeRange.Width));

    // Bail out if the subexpr's range is as wide as the cast type.
    if (SubRange.Width >= OutputTypeRange.Width)
      return OutputTypeRange;

    // Otherwise, we take the smaller width, and we're non-negative if
    // either the output type or the subexpr is.
    return IntRange(SubRange.Width,
                    SubRange.NonNegative || OutputTypeRange.NonNegative);
  }

  if (ConditionalOperator *CO = dyn_cast<ConditionalOperator>(E)) {
    // If we can fold the condition, just take that operand.
    bool CondResult;
    if (CO->getCond()->EvaluateAsBooleanCondition(CondResult, C))
      return GetExprRange(C, CondResult ? CO->getTrueExpr()
                                        : CO->getFalseExpr(),
                          MaxWidth);

    // Otherwise, conservatively merge.
    IntRange L = GetExprRange(C, CO->getTrueExpr(), MaxWidth);
    IntRange R = GetExprRange(C, CO->getFalseExpr(), MaxWidth);
    return IntRange::join(L, R);
  }

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {

    // Boolean-valued operations are single-bit and positive.
    case BO_LAnd:
    case BO_LOr:
    case BO_LT:
    case BO_GT:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
      return IntRange::forBoolType();

    // The type of the assignments is the type of the LHS, so the RHS
    // is not necessarily the same type.
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      // TODO: bitfields?
      return IntRange::forValueOfType(C, E->getType());

    // Simple assignments just pass through the RHS, which will have
    // been coerced to the LHS type.
    case BO_Assign:
      // TODO: bitfields?
      return GetExprRange(C, BO->getRHS(), MaxWidth);

    // Operations with opaque sources are black-listed.
    case BO_PtrMemD:
    case BO_PtrMemI:
      return IntRange::forValueOfType(C, E->getType());

    // Bitwise-and uses the *infinum* of the two source ranges.
    case BO_And:
    case BO_AndAssign:
      return IntRange::meet(GetExprRange(C, BO->getLHS(), MaxWidth),
                            GetExprRange(C, BO->getRHS(), MaxWidth));

    // Left shift gets black-listed based on a judgement call.
    case BO_Shl:
      // ...except that we want to treat '1 << (blah)' as logically
      // positive.  It's an important idiom.
      if (IntegerLiteral *I
            = dyn_cast<IntegerLiteral>(BO->getLHS()->IgnoreParenCasts())) {
        if (I->getValue() == 1) {
          IntRange R = IntRange::forValueOfType(C, E->getType());
          return IntRange(R.Width, /*NonNegative*/ true);
        }
      }
      // fallthrough

    case BO_ShlAssign:
      return IntRange::forValueOfType(C, E->getType());

    // Right shift by a constant can narrow its left argument.
    case BO_Shr:
    case BO_ShrAssign: {
      IntRange L = GetExprRange(C, BO->getLHS(), MaxWidth);

      // If the shift amount is a positive constant, drop the width by
      // that much.
      llvm::APSInt shift;
      if (BO->getRHS()->isIntegerConstantExpr(shift, C) &&
          shift.isNonNegative()) {
        unsigned zext = shift.getZExtValue();
        if (zext >= L.Width)
          L.Width = (L.NonNegative ? 0 : 1);
        else
          L.Width -= zext;
      }

      return L;
    }

    // Comma acts as its right operand.
    case BO_Comma:
      return GetExprRange(C, BO->getRHS(), MaxWidth);

    // Black-list pointer subtractions.
    case BO_Sub:
      if (BO->getLHS()->getType()->isPointerType())
        return IntRange::forValueOfType(C, E->getType());
      break;

    // The width of a division result is mostly determined by the size
    // of the LHS.
    case BO_Div: {
      // Don't 'pre-truncate' the operands.
      unsigned opWidth = C.getIntWidth(E->getType());
      IntRange L = GetExprRange(C, BO->getLHS(), opWidth);

      // If the divisor is constant, use that.
      llvm::APSInt divisor;
      if (BO->getRHS()->isIntegerConstantExpr(divisor, C)) {
        unsigned log2 = divisor.logBase2(); // floor(log_2(divisor))
        if (log2 >= L.Width)
          L.Width = (L.NonNegative ? 0 : 1);
        else
          L.Width = std::min(L.Width - log2, MaxWidth);
        return L;
      }

      // Otherwise, just use the LHS's width.
      IntRange R = GetExprRange(C, BO->getRHS(), opWidth);
      return IntRange(L.Width, L.NonNegative && R.NonNegative);
    }

    // The result of a remainder can't be larger than the result of
    // either side.
    case BO_Rem: {
      // Don't 'pre-truncate' the operands.
      unsigned opWidth = C.getIntWidth(E->getType());
      IntRange L = GetExprRange(C, BO->getLHS(), opWidth);
      IntRange R = GetExprRange(C, BO->getRHS(), opWidth);

      IntRange meet = IntRange::meet(L, R);
      meet.Width = std::min(meet.Width, MaxWidth);
      return meet;
    }

    // The default behavior is okay for these.
    case BO_Mul:
    case BO_Add:
    case BO_Xor:
    case BO_Or:
      break;
    }

    // The default case is to treat the operation as if it were closed
    // on the narrowest type that encompasses both operands.
    IntRange L = GetExprRange(C, BO->getLHS(), MaxWidth);
    IntRange R = GetExprRange(C, BO->getRHS(), MaxWidth);
    return IntRange::join(L, R);
  }

  if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    // Boolean-valued operations are white-listed.
    case UO_LNot:
      return IntRange::forBoolType();

    // Operations with opaque sources are black-listed.
    case UO_Deref:
    case UO_AddrOf: // should be impossible
      return IntRange::forValueOfType(C, E->getType());

    default:
      return GetExprRange(C, UO->getSubExpr(), MaxWidth);
    }
  }
  
  if (dyn_cast<OffsetOfExpr>(E)) {
    IntRange::forValueOfType(C, E->getType());
  }

  if (FieldDecl *BitField = E->getBitField())
    return IntRange(BitField->getBitWidthValue(C),
                    BitField->getType()->isUnsignedIntegerOrEnumerationType());

  return IntRange::forValueOfType(C, E->getType());
}

static IntRange GetExprRange(ASTContext &C, Expr *E) {
  return GetExprRange(C, E, C.getIntWidth(E->getType()));
}

/// Checks whether the given value, which currently has the given
/// source semantics, has the same value when coerced through the
/// target semantics.
static bool IsSameFloatAfterCast(const llvm::APFloat &value,
                                 const llvm::fltSemantics &Src,
                                 const llvm::fltSemantics &Tgt) {
  llvm::APFloat truncated = value;

  bool ignored;
  truncated.convert(Src, llvm::APFloat::rmNearestTiesToEven, &ignored);
  truncated.convert(Tgt, llvm::APFloat::rmNearestTiesToEven, &ignored);

  return truncated.bitwiseIsEqual(value);
}

/// Checks whether the given value, which currently has the given
/// source semantics, has the same value when coerced through the
/// target semantics.
///
/// The value might be a vector of floats (or a complex number).
static bool IsSameFloatAfterCast(const APValue &value,
                                 const llvm::fltSemantics &Src,
                                 const llvm::fltSemantics &Tgt) {
  if (value.isFloat())
    return IsSameFloatAfterCast(value.getFloat(), Src, Tgt);

  if (value.isVector()) {
    for (unsigned i = 0, e = value.getVectorLength(); i != e; ++i)
      if (!IsSameFloatAfterCast(value.getVectorElt(i), Src, Tgt))
        return false;
    return true;
  }

  assert(value.isComplexFloat());
  return (IsSameFloatAfterCast(value.getComplexFloatReal(), Src, Tgt) &&
          IsSameFloatAfterCast(value.getComplexFloatImag(), Src, Tgt));
}

static void AnalyzeImplicitConversions(Sema &S, Expr *E, SourceLocation CC);

static bool IsZero(Sema &S, Expr *E) {
  // Suppress cases where we are comparing against an enum constant.
  if (const DeclRefExpr *DR =
      dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts()))
    if (isa<EnumConstantDecl>(DR->getDecl()))
      return false;

  // Suppress cases where the '0' value is expanded from a macro.
  if (E->getLocStart().isMacroID())
    return false;

  llvm::APSInt Value;
  return E->isIntegerConstantExpr(Value, S.Context) && Value == 0;
}

static bool HasEnumType(Expr *E) {
  // Strip off implicit integral promotions.
  while (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() != CK_IntegralCast &&
        ICE->getCastKind() != CK_NoOp)
      break;
    E = ICE->getSubExpr();
  }

  return E->getType()->isEnumeralType();
}

static void CheckTrivialUnsignedComparison(Sema &S, BinaryOperator *E) {
  BinaryOperatorKind op = E->getOpcode();
  if (E->isValueDependent())
    return;

  if (op == BO_LT && IsZero(S, E->getRHS())) {
    S.Diag(E->getOperatorLoc(), diag::warn_lunsigned_always_true_comparison)
      << "< 0" << "false" << HasEnumType(E->getLHS())
      << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
  } else if (op == BO_GE && IsZero(S, E->getRHS())) {
    S.Diag(E->getOperatorLoc(), diag::warn_lunsigned_always_true_comparison)
      << ">= 0" << "true" << HasEnumType(E->getLHS())
      << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
  } else if (op == BO_GT && IsZero(S, E->getLHS())) {
    S.Diag(E->getOperatorLoc(), diag::warn_runsigned_always_true_comparison)
      << "0 >" << "false" << HasEnumType(E->getRHS())
      << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
  } else if (op == BO_LE && IsZero(S, E->getLHS())) {
    S.Diag(E->getOperatorLoc(), diag::warn_runsigned_always_true_comparison)
      << "0 <=" << "true" << HasEnumType(E->getRHS())
      << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
  }
}

static void DiagnoseOutOfRangeComparison(Sema &S, BinaryOperator *E,
                                         Expr *Constant, Expr *Other,
                                         llvm::APSInt Value,
                                         bool RhsConstant) {
  BinaryOperatorKind op = E->getOpcode();
  QualType OtherT = Other->getType();
  QualType ConstantT = Constant->getType();
  if (S.Context.hasSameUnqualifiedType(OtherT, ConstantT))
    return;
  assert((OtherT->isIntegerType() && ConstantT->isIntegerType())
         && "comparison with non-integer type");
  // FIXME. handle cases for signedness to catch (signed char)N == 200
  IntRange OtherRange = IntRange::forValueOfType(S.Context, OtherT);
  IntRange LitRange = GetValueRange(S.Context, Value, Value.getBitWidth());
  if (OtherRange.Width >= LitRange.Width)
    return;
  bool IsTrue = true;
  if (op == BO_EQ)
    IsTrue = false;
  else if (op == BO_NE)
    IsTrue = true;
  else if (RhsConstant) {
    if (op == BO_GT || op == BO_GE)
      IsTrue = !LitRange.NonNegative;
    else // op == BO_LT || op == BO_LE
      IsTrue = LitRange.NonNegative;
  } else {
    if (op == BO_LT || op == BO_LE)
      IsTrue = !LitRange.NonNegative;
    else // op == BO_GT || op == BO_GE
      IsTrue = LitRange.NonNegative;
  }
  SmallString<16> PrettySourceValue(Value.toString(10));
  S.Diag(E->getOperatorLoc(), diag::warn_out_of_range_compare)
  << PrettySourceValue << OtherT << IsTrue
  << E->getLHS()->getSourceRange() << E->getRHS()->getSourceRange();
}

/// Analyze the operands of the given comparison.  Implements the
/// fallback case from AnalyzeComparison.
static void AnalyzeImpConvsInComparison(Sema &S, BinaryOperator *E) {
  AnalyzeImplicitConversions(S, E->getLHS(), E->getOperatorLoc());
  AnalyzeImplicitConversions(S, E->getRHS(), E->getOperatorLoc());
}

/// \brief Implements -Wsign-compare.
///
/// \param E the binary operator to check for warnings
static void AnalyzeComparison(Sema &S, BinaryOperator *E) {
  // The type the comparison is being performed in.
  QualType T = E->getLHS()->getType();
  assert(S.Context.hasSameUnqualifiedType(T, E->getRHS()->getType())
         && "comparison with mismatched types");
  if (E->isValueDependent())
    return AnalyzeImpConvsInComparison(S, E);

  Expr *LHS = E->getLHS()->IgnoreParenImpCasts();
  Expr *RHS = E->getRHS()->IgnoreParenImpCasts();
  
  bool IsComparisonConstant = false;
  
  // Check whether an integer constant comparison results in a value
  // of 'true' or 'false'.
  if (T->isIntegralType(S.Context)) {
    llvm::APSInt RHSValue;
    bool IsRHSIntegralLiteral = 
      RHS->isIntegerConstantExpr(RHSValue, S.Context);
    llvm::APSInt LHSValue;
    bool IsLHSIntegralLiteral = 
      LHS->isIntegerConstantExpr(LHSValue, S.Context);
    if (IsRHSIntegralLiteral && !IsLHSIntegralLiteral)
        DiagnoseOutOfRangeComparison(S, E, RHS, LHS, RHSValue, true);
    else if (!IsRHSIntegralLiteral && IsLHSIntegralLiteral)
      DiagnoseOutOfRangeComparison(S, E, LHS, RHS, LHSValue, false);
    else
      IsComparisonConstant = 
        (IsRHSIntegralLiteral && IsLHSIntegralLiteral);
  } else if (!T->hasUnsignedIntegerRepresentation())
      IsComparisonConstant = E->isIntegerConstantExpr(S.Context);
  
  // We don't do anything special if this isn't an unsigned integral
  // comparison:  we're only interested in integral comparisons, and
  // signed comparisons only happen in cases we don't care to warn about.
  //
  // We also don't care about value-dependent expressions or expressions
  // whose result is a constant.
  if (!T->hasUnsignedIntegerRepresentation() || IsComparisonConstant)
    return AnalyzeImpConvsInComparison(S, E);
  
  // Check to see if one of the (unmodified) operands is of different
  // signedness.
  Expr *signedOperand, *unsignedOperand;
  if (LHS->getType()->hasSignedIntegerRepresentation()) {
    assert(!RHS->getType()->hasSignedIntegerRepresentation() &&
           "unsigned comparison between two signed integer expressions?");
    signedOperand = LHS;
    unsignedOperand = RHS;
  } else if (RHS->getType()->hasSignedIntegerRepresentation()) {
    signedOperand = RHS;
    unsignedOperand = LHS;
  } else {
    CheckTrivialUnsignedComparison(S, E);
    return AnalyzeImpConvsInComparison(S, E);
  }

  // Otherwise, calculate the effective range of the signed operand.
  IntRange signedRange = GetExprRange(S.Context, signedOperand);

  // Go ahead and analyze implicit conversions in the operands.  Note
  // that we skip the implicit conversions on both sides.
  AnalyzeImplicitConversions(S, LHS, E->getOperatorLoc());
  AnalyzeImplicitConversions(S, RHS, E->getOperatorLoc());

  // If the signed range is non-negative, -Wsign-compare won't fire,
  // but we should still check for comparisons which are always true
  // or false.
  if (signedRange.NonNegative)
    return CheckTrivialUnsignedComparison(S, E);

  // For (in)equality comparisons, if the unsigned operand is a
  // constant which cannot collide with a overflowed signed operand,
  // then reinterpreting the signed operand as unsigned will not
  // change the result of the comparison.
  if (E->isEqualityOp()) {
    unsigned comparisonWidth = S.Context.getIntWidth(T);
    IntRange unsignedRange = GetExprRange(S.Context, unsignedOperand);

    // We should never be unable to prove that the unsigned operand is
    // non-negative.
    assert(unsignedRange.NonNegative && "unsigned range includes negative?");

    if (unsignedRange.Width < comparisonWidth)
      return;
  }

  S.DiagRuntimeBehavior(E->getOperatorLoc(), E,
    S.PDiag(diag::warn_mixed_sign_comparison)
      << LHS->getType() << RHS->getType()
      << LHS->getSourceRange() << RHS->getSourceRange());
}

/// Analyzes an attempt to assign the given value to a bitfield.
///
/// Returns true if there was something fishy about the attempt.
static bool AnalyzeBitFieldAssignment(Sema &S, FieldDecl *Bitfield, Expr *Init,
                                      SourceLocation InitLoc) {
  assert(Bitfield->isBitField());
  if (Bitfield->isInvalidDecl())
    return false;

  // White-list bool bitfields.
  if (Bitfield->getType()->isBooleanType())
    return false;

  // Ignore value- or type-dependent expressions.
  if (Bitfield->getBitWidth()->isValueDependent() ||
      Bitfield->getBitWidth()->isTypeDependent() ||
      Init->isValueDependent() ||
      Init->isTypeDependent())
    return false;

  Expr *OriginalInit = Init->IgnoreParenImpCasts();

  llvm::APSInt Value;
  if (!OriginalInit->EvaluateAsInt(Value, S.Context, Expr::SE_AllowSideEffects))
    return false;

  unsigned OriginalWidth = Value.getBitWidth();
  unsigned FieldWidth = Bitfield->getBitWidthValue(S.Context);

  if (OriginalWidth <= FieldWidth)
    return false;

  // Compute the value which the bitfield will contain.
  llvm::APSInt TruncatedValue = Value.trunc(FieldWidth);
  TruncatedValue.setIsSigned(Bitfield->getType()->isSignedIntegerType());

  // Check whether the stored value is equal to the original value.
  TruncatedValue = TruncatedValue.extend(OriginalWidth);
  if (llvm::APSInt::isSameValue(Value, TruncatedValue))
    return false;

  // Special-case bitfields of width 1: booleans are naturally 0/1, and
  // therefore don't strictly fit into a signed bitfield of width 1.
  if (FieldWidth == 1 && Value == 1)
    return false;

  std::string PrettyValue = Value.toString(10);
  std::string PrettyTrunc = TruncatedValue.toString(10);

  S.Diag(InitLoc, diag::warn_impcast_bitfield_precision_constant)
    << PrettyValue << PrettyTrunc << OriginalInit->getType()
    << Init->getSourceRange();

  return true;
}

/// Analyze the given simple or compound assignment for warning-worthy
/// operations.
static void AnalyzeAssignment(Sema &S, BinaryOperator *E) {
  // Just recurse on the LHS.
  AnalyzeImplicitConversions(S, E->getLHS(), E->getOperatorLoc());

  // We want to recurse on the RHS as normal unless we're assigning to
  // a bitfield.
  if (FieldDecl *Bitfield = E->getLHS()->getBitField()) {
    if (AnalyzeBitFieldAssignment(S, Bitfield, E->getRHS(),
                                  E->getOperatorLoc())) {
      // Recurse, ignoring any implicit conversions on the RHS.
      return AnalyzeImplicitConversions(S, E->getRHS()->IgnoreParenImpCasts(),
                                        E->getOperatorLoc());
    }
  }

  AnalyzeImplicitConversions(S, E->getRHS(), E->getOperatorLoc());
}

/// Diagnose an implicit cast;  purely a helper for CheckImplicitConversion.
static void DiagnoseImpCast(Sema &S, Expr *E, QualType SourceType, QualType T, 
                            SourceLocation CContext, unsigned diag,
                            bool pruneControlFlow = false) {
  if (pruneControlFlow) {
    S.DiagRuntimeBehavior(E->getExprLoc(), E,
                          S.PDiag(diag)
                            << SourceType << T << E->getSourceRange()
                            << SourceRange(CContext));
    return;
  }
  S.Diag(E->getExprLoc(), diag)
    << SourceType << T << E->getSourceRange() << SourceRange(CContext);
}

/// Diagnose an implicit cast;  purely a helper for CheckImplicitConversion.
static void DiagnoseImpCast(Sema &S, Expr *E, QualType T,
                            SourceLocation CContext, unsigned diag,
                            bool pruneControlFlow = false) {
  DiagnoseImpCast(S, E, E->getType(), T, CContext, diag, pruneControlFlow);
}

/// Diagnose an implicit cast from a literal expression. Does not warn when the
/// cast wouldn't lose information.
void DiagnoseFloatingLiteralImpCast(Sema &S, FloatingLiteral *FL, QualType T,
                                    SourceLocation CContext) {
  // Try to convert the literal exactly to an integer. If we can, don't warn.
  bool isExact = false;
  const llvm::APFloat &Value = FL->getValue();
  llvm::APSInt IntegerValue(S.Context.getIntWidth(T),
                            T->hasUnsignedIntegerRepresentation());
  if (Value.convertToInteger(IntegerValue,
                             llvm::APFloat::rmTowardZero, &isExact)
      == llvm::APFloat::opOK && isExact)
    return;

  SmallString<16> PrettySourceValue;
  Value.toString(PrettySourceValue);
  SmallString<16> PrettyTargetValue;
  if (T->isSpecificBuiltinType(BuiltinType::Bool))
    PrettyTargetValue = IntegerValue == 0 ? "false" : "true";
  else
    IntegerValue.toString(PrettyTargetValue);

  S.Diag(FL->getExprLoc(), diag::warn_impcast_literal_float_to_integer)
    << FL->getType() << T.getUnqualifiedType() << PrettySourceValue
    << PrettyTargetValue << FL->getSourceRange() << SourceRange(CContext);
}

std::string PrettyPrintInRange(const llvm::APSInt &Value, IntRange Range) {
  if (!Range.Width) return "0";

  llvm::APSInt ValueInRange = Value;
  ValueInRange.setIsSigned(!Range.NonNegative);
  ValueInRange = ValueInRange.trunc(Range.Width);
  return ValueInRange.toString(10);
}

static bool IsImplicitBoolFloatConversion(Sema &S, Expr *Ex, bool ToBool) {
  if (!isa<ImplicitCastExpr>(Ex))
    return false;

  Expr *InnerE = Ex->IgnoreParenImpCasts();
  const Type *Target = S.Context.getCanonicalType(Ex->getType()).getTypePtr();
  const Type *Source =
    S.Context.getCanonicalType(InnerE->getType()).getTypePtr();
  if (Target->isDependentType())
    return false;

  const BuiltinType *FloatCandidateBT =
    dyn_cast<BuiltinType>(ToBool ? Source : Target);
  const Type *BoolCandidateType = ToBool ? Target : Source;

  return (BoolCandidateType->isSpecificBuiltinType(BuiltinType::Bool) &&
          FloatCandidateBT && (FloatCandidateBT->isFloatingPoint()));
}

void CheckImplicitArgumentConversions(Sema &S, CallExpr *TheCall,
                                      SourceLocation CC) {
  unsigned NumArgs = TheCall->getNumArgs();
  for (unsigned i = 0; i < NumArgs; ++i) {
    Expr *CurrA = TheCall->getArg(i);
    if (!IsImplicitBoolFloatConversion(S, CurrA, true))
      continue;

    bool IsSwapped = ((i > 0) &&
        IsImplicitBoolFloatConversion(S, TheCall->getArg(i - 1), false));
    IsSwapped |= ((i < (NumArgs - 1)) &&
        IsImplicitBoolFloatConversion(S, TheCall->getArg(i + 1), false));
    if (IsSwapped) {
      // Warn on this floating-point to bool conversion.
      DiagnoseImpCast(S, CurrA->IgnoreParenImpCasts(),
                      CurrA->getType(), CC,
                      diag::warn_impcast_floating_point_to_bool);
    }
  }
}

void CheckImplicitConversion(Sema &S, Expr *E, QualType T,
                             SourceLocation CC, bool *ICContext = 0) {
  if (E->isTypeDependent() || E->isValueDependent()) return;

  const Type *Source = S.Context.getCanonicalType(E->getType()).getTypePtr();
  const Type *Target = S.Context.getCanonicalType(T).getTypePtr();
  if (Source == Target) return;
  if (Target->isDependentType()) return;

  // If the conversion context location is invalid don't complain. We also
  // don't want to emit a warning if the issue occurs from the expansion of
  // a system macro. The problem is that 'getSpellingLoc()' is slow, so we
  // delay this check as long as possible. Once we detect we are in that
  // scenario, we just return.
  if (CC.isInvalid())
    return;

  // Diagnose implicit casts to bool.
  if (Target->isSpecificBuiltinType(BuiltinType::Bool)) {
    if (isa<StringLiteral>(E))
      // Warn on string literal to bool.  Checks for string literals in logical
      // expressions, for instances, assert(0 && "error here"), is prevented
      // by a check in AnalyzeImplicitConversions().
      return DiagnoseImpCast(S, E, T, CC,
                             diag::warn_impcast_string_literal_to_bool);
    if (Source->isFunctionType()) {
      // Warn on function to bool. Checks free functions and static member
      // functions. Weakly imported functions are excluded from the check,
      // since it's common to test their value to check whether the linker
      // found a definition for them.
      ValueDecl *D = 0;
      if (DeclRefExpr* R = dyn_cast<DeclRefExpr>(E)) {
        D = R->getDecl();
      } else if (MemberExpr *M = dyn_cast<MemberExpr>(E)) {
        D = M->getMemberDecl();
      }

      if (D && !D->isWeak()) {
        if (FunctionDecl* F = dyn_cast<FunctionDecl>(D)) {
          S.Diag(E->getExprLoc(), diag::warn_impcast_function_to_bool)
            << F << E->getSourceRange() << SourceRange(CC);
          S.Diag(E->getExprLoc(), diag::note_function_to_bool_silence)
            << FixItHint::CreateInsertion(E->getExprLoc(), "&");
          QualType ReturnType;
          UnresolvedSet<4> NonTemplateOverloads;
          S.isExprCallable(*E, ReturnType, NonTemplateOverloads);
          if (!ReturnType.isNull() 
              && ReturnType->isSpecificBuiltinType(BuiltinType::Bool))
            S.Diag(E->getExprLoc(), diag::note_function_to_bool_call)
              << FixItHint::CreateInsertion(
                 S.getPreprocessor().getLocForEndOfToken(E->getLocEnd()), "()");
          return;
        }
      }
    }
  }

  // Strip vector types.
  if (isa<VectorType>(Source)) {
    if (!isa<VectorType>(Target)) {
      if (S.SourceMgr.isInSystemMacro(CC))
        return;
      return DiagnoseImpCast(S, E, T, CC, diag::warn_impcast_vector_scalar);
    }
    
    // If the vector cast is cast between two vectors of the same size, it is
    // a bitcast, not a conversion.
    if (S.Context.getTypeSize(Source) == S.Context.getTypeSize(Target))
      return;

    Source = cast<VectorType>(Source)->getElementType().getTypePtr();
    Target = cast<VectorType>(Target)->getElementType().getTypePtr();
  }

  // Strip complex types.
  if (isa<ComplexType>(Source)) {
    if (!isa<ComplexType>(Target)) {
      if (S.SourceMgr.isInSystemMacro(CC))
        return;

      return DiagnoseImpCast(S, E, T, CC, diag::warn_impcast_complex_scalar);
    }

    Source = cast<ComplexType>(Source)->getElementType().getTypePtr();
    Target = cast<ComplexType>(Target)->getElementType().getTypePtr();
  }

  const BuiltinType *SourceBT = dyn_cast<BuiltinType>(Source);
  const BuiltinType *TargetBT = dyn_cast<BuiltinType>(Target);

  // If the source is floating point...
  if (SourceBT && SourceBT->isFloatingPoint()) {
    // ...and the target is floating point...
    if (TargetBT && TargetBT->isFloatingPoint()) {
      // ...then warn if we're dropping FP rank.

      // Builtin FP kinds are ordered by increasing FP rank.
      if (SourceBT->getKind() > TargetBT->getKind()) {
        // Don't warn about float constants that are precisely
        // representable in the target type.
        Expr::EvalResult result;
        if (E->EvaluateAsRValue(result, S.Context)) {
          // Value might be a float, a float vector, or a float complex.
          if (IsSameFloatAfterCast(result.Val,
                   S.Context.getFloatTypeSemantics(QualType(TargetBT, 0)),
                   S.Context.getFloatTypeSemantics(QualType(SourceBT, 0))))
            return;
        }

        if (S.SourceMgr.isInSystemMacro(CC))
          return;

        DiagnoseImpCast(S, E, T, CC, diag::warn_impcast_float_precision);
      }
      return;
    }

    // If the target is integral, always warn.    
    if (TargetBT && TargetBT->isInteger()) {
      if (S.SourceMgr.isInSystemMacro(CC))
        return;
      
      Expr *InnerE = E->IgnoreParenImpCasts();
      // We also want to warn on, e.g., "int i = -1.234"
      if (UnaryOperator *UOp = dyn_cast<UnaryOperator>(InnerE))
        if (UOp->getOpcode() == UO_Minus || UOp->getOpcode() == UO_Plus)
          InnerE = UOp->getSubExpr()->IgnoreParenImpCasts();

      if (FloatingLiteral *FL = dyn_cast<FloatingLiteral>(InnerE)) {
        DiagnoseFloatingLiteralImpCast(S, FL, T, CC);
      } else {
        DiagnoseImpCast(S, E, T, CC, diag::warn_impcast_float_integer);
      }
    }

    // If the target is bool, warn if expr is a function or method call.
    if (Target->isSpecificBuiltinType(BuiltinType::Bool) &&
        isa<CallExpr>(E)) {
      // Check last argument of function call to see if it is an
      // implicit cast from a type matching the type the result
      // is being cast to.
      CallExpr *CEx = cast<CallExpr>(E);
      unsigned NumArgs = CEx->getNumArgs();
      if (NumArgs > 0) {
        Expr *LastA = CEx->getArg(NumArgs - 1);
        Expr *InnerE = LastA->IgnoreParenImpCasts();
        const Type *InnerType =
          S.Context.getCanonicalType(InnerE->getType()).getTypePtr();
        if (isa<ImplicitCastExpr>(LastA) && (InnerType == Target)) {
          // Warn on this floating-point to bool conversion
          DiagnoseImpCast(S, E, T, CC,
                          diag::warn_impcast_floating_point_to_bool);
        }
      }
    }
    return;
  }

  if ((E->isNullPointerConstant(S.Context, Expr::NPC_ValueDependentIsNotNull)
           == Expr::NPCK_GNUNull) && !Target->isAnyPointerType()
      && !Target->isBlockPointerType() && !Target->isMemberPointerType()
      && Target->isScalarType()) {
    SourceLocation Loc = E->getSourceRange().getBegin();
    if (Loc.isMacroID())
      Loc = S.SourceMgr.getImmediateExpansionRange(Loc).first;
    if (!Loc.isMacroID() || CC.isMacroID())
      S.Diag(Loc, diag::warn_impcast_null_pointer_to_integer)
          << T << clang::SourceRange(CC)
          << FixItHint::CreateReplacement(Loc, S.getFixItZeroLiteralForType(T));
  }

  if (!Source->isIntegerType() || !Target->isIntegerType())
    return;

  // TODO: remove this early return once the false positives for constant->bool
  // in templates, macros, etc, are reduced or removed.
  if (Target->isSpecificBuiltinType(BuiltinType::Bool))
    return;

  IntRange SourceRange = GetExprRange(S.Context, E);
  IntRange TargetRange = IntRange::forTargetOfCanonicalType(S.Context, Target);

  if (SourceRange.Width > TargetRange.Width) {
    // If the source is a constant, use a default-on diagnostic.
    // TODO: this should happen for bitfield stores, too.
    llvm::APSInt Value(32);
    if (E->isIntegerConstantExpr(Value, S.Context)) {
      if (S.SourceMgr.isInSystemMacro(CC))
        return;

      std::string PrettySourceValue = Value.toString(10);
      std::string PrettyTargetValue = PrettyPrintInRange(Value, TargetRange);

      S.DiagRuntimeBehavior(E->getExprLoc(), E,
        S.PDiag(diag::warn_impcast_integer_precision_constant)
            << PrettySourceValue << PrettyTargetValue
            << E->getType() << T << E->getSourceRange()
            << clang::SourceRange(CC));
      return;
    }

    // People want to build with -Wshorten-64-to-32 and not -Wconversion.
    if (S.SourceMgr.isInSystemMacro(CC))
      return;
    
    if (TargetRange.Width == 32 && S.Context.getIntWidth(E->getType()) == 64)
      return DiagnoseImpCast(S, E, T, CC, diag::warn_impcast_integer_64_32,
                             /* pruneControlFlow */ true);
    return DiagnoseImpCast(S, E, T, CC, diag::warn_impcast_integer_precision);
  }

  if ((TargetRange.NonNegative && !SourceRange.NonNegative) ||
      (!TargetRange.NonNegative && SourceRange.NonNegative &&
       SourceRange.Width == TargetRange.Width)) {
        
    if (S.SourceMgr.isInSystemMacro(CC))
      return;

    unsigned DiagID = diag::warn_impcast_integer_sign;

    // Traditionally, gcc has warned about this under -Wsign-compare.
    // We also want to warn about it in -Wconversion.
    // So if -Wconversion is off, use a completely identical diagnostic
    // in the sign-compare group.
    // The conditional-checking code will 
    if (ICContext) {
      DiagID = diag::warn_impcast_integer_sign_conditional;
      *ICContext = true;
    }

    return DiagnoseImpCast(S, E, T, CC, DiagID);
  }

  // Diagnose conversions between different enumeration types.
  // In C, we pretend that the type of an EnumConstantDecl is its enumeration
  // type, to give us better diagnostics.
  QualType SourceType = E->getType();
  if (!S.getLangOpts().CPlusPlus) {
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
      if (EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        EnumDecl *Enum = cast<EnumDecl>(ECD->getDeclContext());
        SourceType = S.Context.getTypeDeclType(Enum);
        Source = S.Context.getCanonicalType(SourceType).getTypePtr();
      }
  }
  
  if (const EnumType *SourceEnum = Source->getAs<EnumType>())
    if (const EnumType *TargetEnum = Target->getAs<EnumType>())
      if ((SourceEnum->getDecl()->getIdentifier() || 
           SourceEnum->getDecl()->getTypedefNameForAnonDecl()) &&
          (TargetEnum->getDecl()->getIdentifier() ||
           TargetEnum->getDecl()->getTypedefNameForAnonDecl()) &&
          SourceEnum != TargetEnum) {
        if (S.SourceMgr.isInSystemMacro(CC))
          return;

        return DiagnoseImpCast(S, E, SourceType, T, CC, 
                               diag::warn_impcast_different_enum_types);
      }
  
  return;
}

void CheckConditionalOperator(Sema &S, ConditionalOperator *E,
                              SourceLocation CC, QualType T);

void CheckConditionalOperand(Sema &S, Expr *E, QualType T,
                             SourceLocation CC, bool &ICContext) {
  E = E->IgnoreParenImpCasts();

  if (isa<ConditionalOperator>(E))
    return CheckConditionalOperator(S, cast<ConditionalOperator>(E), CC, T);

  AnalyzeImplicitConversions(S, E, CC);
  if (E->getType() != T)
    return CheckImplicitConversion(S, E, T, CC, &ICContext);
  return;
}

void CheckConditionalOperator(Sema &S, ConditionalOperator *E,
                              SourceLocation CC, QualType T) {
  AnalyzeImplicitConversions(S, E->getCond(), CC);

  bool Suspicious = false;
  CheckConditionalOperand(S, E->getTrueExpr(), T, CC, Suspicious);
  CheckConditionalOperand(S, E->getFalseExpr(), T, CC, Suspicious);

  // If -Wconversion would have warned about either of the candidates
  // for a signedness conversion to the context type...
  if (!Suspicious) return;

  // ...but it's currently ignored...
  if (S.Diags.getDiagnosticLevel(diag::warn_impcast_integer_sign_conditional,
                                 CC))
    return;

  // ...then check whether it would have warned about either of the
  // candidates for a signedness conversion to the condition type.
  if (E->getType() == T) return;
 
  Suspicious = false;
  CheckImplicitConversion(S, E->getTrueExpr()->IgnoreParenImpCasts(),
                          E->getType(), CC, &Suspicious);
  if (!Suspicious)
    CheckImplicitConversion(S, E->getFalseExpr()->IgnoreParenImpCasts(),
                            E->getType(), CC, &Suspicious);
}

/// AnalyzeImplicitConversions - Find and report any interesting
/// implicit conversions in the given expression.  There are a couple
/// of competing diagnostics here, -Wconversion and -Wsign-compare.
void AnalyzeImplicitConversions(Sema &S, Expr *OrigE, SourceLocation CC) {
  QualType T = OrigE->getType();
  Expr *E = OrigE->IgnoreParenImpCasts();

  if (E->isTypeDependent() || E->isValueDependent())
    return;

  // For conditional operators, we analyze the arguments as if they
  // were being fed directly into the output.
  if (isa<ConditionalOperator>(E)) {
    ConditionalOperator *CO = cast<ConditionalOperator>(E);
    CheckConditionalOperator(S, CO, CC, T);
    return;
  }

  // Check implicit argument conversions for function calls.
  if (CallExpr *Call = dyn_cast<CallExpr>(E))
    CheckImplicitArgumentConversions(S, Call, CC);

  // Go ahead and check any implicit conversions we might have skipped.
  // The non-canonical typecheck is just an optimization;
  // CheckImplicitConversion will filter out dead implicit conversions.
  if (E->getType() != T)
    CheckImplicitConversion(S, E, T, CC);

  // Now continue drilling into this expression.

  // Skip past explicit casts.
  if (isa<ExplicitCastExpr>(E)) {
    E = cast<ExplicitCastExpr>(E)->getSubExpr()->IgnoreParenImpCasts();
    return AnalyzeImplicitConversions(S, E, CC);
  }

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    // Do a somewhat different check with comparison operators.
    if (BO->isComparisonOp())
      return AnalyzeComparison(S, BO);

    // And with simple assignments.
    if (BO->getOpcode() == BO_Assign)
      return AnalyzeAssignment(S, BO);
  }

  // These break the otherwise-useful invariant below.  Fortunately,
  // we don't really need to recurse into them, because any internal
  // expressions should have been analyzed already when they were
  // built into statements.
  if (isa<StmtExpr>(E)) return;

  // Don't descend into unevaluated contexts.
  if (isa<UnaryExprOrTypeTraitExpr>(E)) return;

  // Now just recurse over the expression's children.
  CC = E->getExprLoc();
  BinaryOperator *BO = dyn_cast<BinaryOperator>(E);
  bool IsLogicalOperator = BO && BO->isLogicalOp();
  for (Stmt::child_range I = E->children(); I; ++I) {
    Expr *ChildExpr = dyn_cast_or_null<Expr>(*I);
    if (!ChildExpr)
      continue;

    if (IsLogicalOperator &&
        isa<StringLiteral>(ChildExpr->IgnoreParenImpCasts()))
      // Ignore checking string literals that are in logical operators.
      continue;
    AnalyzeImplicitConversions(S, ChildExpr, CC);
  }
}

} // end anonymous namespace

/// Diagnoses "dangerous" implicit conversions within the given
/// expression (which is a full expression).  Implements -Wconversion
/// and -Wsign-compare.
///
/// \param CC the "context" location of the implicit conversion, i.e.
///   the most location of the syntactic entity requiring the implicit
///   conversion
void Sema::CheckImplicitConversions(Expr *E, SourceLocation CC) {
  // Don't diagnose in unevaluated contexts.
  if (isUnevaluatedContext())
    return;

  // Don't diagnose for value- or type-dependent expressions.
  if (E->isTypeDependent() || E->isValueDependent())
    return;

  // Check for array bounds violations in cases where the check isn't triggered
  // elsewhere for other Expr types (like BinaryOperators), e.g. when an
  // ArraySubscriptExpr is on the RHS of a variable initialization.
  CheckArrayAccess(E);

  // This is not the right CC for (e.g.) a variable initialization.
  AnalyzeImplicitConversions(*this, E, CC);
}

void Sema::CheckBitFieldInitialization(SourceLocation InitLoc,
                                       FieldDecl *BitField,
                                       Expr *Init) {
  (void) AnalyzeBitFieldAssignment(*this, BitField, Init, InitLoc);
}

/// CheckParmsForFunctionDef - Check that the parameters of the given
/// function are appropriate for the definition of a function. This
/// takes care of any checks that cannot be performed on the
/// declaration itself, e.g., that the types of each of the function
/// parameters are complete.
bool Sema::CheckParmsForFunctionDef(ParmVarDecl **P, ParmVarDecl **PEnd,
                                    bool CheckParameterNames) {
  bool HasInvalidParm = false;
  for (; P != PEnd; ++P) {
    ParmVarDecl *Param = *P;
    
    // C99 6.7.5.3p4: the parameters in a parameter type list in a
    // function declarator that is part of a function definition of
    // that function shall not have incomplete type.
    //
    // This is also C++ [dcl.fct]p6.
    if (!Param->isInvalidDecl() &&
        RequireCompleteType(Param->getLocation(), Param->getType(),
                            diag::err_typecheck_decl_incomplete_type)) {
      Param->setInvalidDecl();
      HasInvalidParm = true;
    }

    // C99 6.9.1p5: If the declarator includes a parameter type list, the
    // declaration of each parameter shall include an identifier.
    if (CheckParameterNames &&
        Param->getIdentifier() == 0 &&
        !Param->isImplicit() &&
        !getLangOpts().CPlusPlus)
      Diag(Param->getLocation(), diag::err_parameter_name_omitted);

    // C99 6.7.5.3p12:
    //   If the function declarator is not part of a definition of that
    //   function, parameters may have incomplete type and may use the [*]
    //   notation in their sequences of declarator specifiers to specify
    //   variable length array types.
    QualType PType = Param->getOriginalType();
    if (const ArrayType *AT = Context.getAsArrayType(PType)) {
      if (AT->getSizeModifier() == ArrayType::Star) {
        // FIXME: This diagnosic should point the '[*]' if source-location
        // information is added for it.
        Diag(Param->getLocation(), diag::err_array_star_in_function_definition);
      }
    }
  }

  return HasInvalidParm;
}

/// CheckCastAlign - Implements -Wcast-align, which warns when a
/// pointer cast increases the alignment requirements.
void Sema::CheckCastAlign(Expr *Op, QualType T, SourceRange TRange) {
  // This is actually a lot of work to potentially be doing on every
  // cast; don't do it if we're ignoring -Wcast_align (as is the default).
  if (getDiagnostics().getDiagnosticLevel(diag::warn_cast_align,
                                          TRange.getBegin())
        == DiagnosticsEngine::Ignored)
    return;

  // Ignore dependent types.
  if (T->isDependentType() || Op->getType()->isDependentType())
    return;

  // Require that the destination be a pointer type.
  const PointerType *DestPtr = T->getAs<PointerType>();
  if (!DestPtr) return;

  // If the destination has alignment 1, we're done.
  QualType DestPointee = DestPtr->getPointeeType();
  if (DestPointee->isIncompleteType()) return;
  CharUnits DestAlign = Context.getTypeAlignInChars(DestPointee);
  if (DestAlign.isOne()) return;

  // Require that the source be a pointer type.
  const PointerType *SrcPtr = Op->getType()->getAs<PointerType>();
  if (!SrcPtr) return;
  QualType SrcPointee = SrcPtr->getPointeeType();

  // Whitelist casts from cv void*.  We already implicitly
  // whitelisted casts to cv void*, since they have alignment 1.
  // Also whitelist casts involving incomplete types, which implicitly
  // includes 'void'.
  if (SrcPointee->isIncompleteType()) return;

  CharUnits SrcAlign = Context.getTypeAlignInChars(SrcPointee);
  if (SrcAlign >= DestAlign) return;

  Diag(TRange.getBegin(), diag::warn_cast_align)
    << Op->getType() << T
    << static_cast<unsigned>(SrcAlign.getQuantity())
    << static_cast<unsigned>(DestAlign.getQuantity())
    << TRange << Op->getSourceRange();
}

static const Type* getElementType(const Expr *BaseExpr) {
  const Type* EltType = BaseExpr->getType().getTypePtr();
  if (EltType->isAnyPointerType())
    return EltType->getPointeeType().getTypePtr();
  else if (EltType->isArrayType())
    return EltType->getBaseElementTypeUnsafe();
  return EltType;
}

/// \brief Check whether this array fits the idiom of a size-one tail padded
/// array member of a struct.
///
/// We avoid emitting out-of-bounds access warnings for such arrays as they are
/// commonly used to emulate flexible arrays in C89 code.
static bool IsTailPaddedMemberArray(Sema &S, llvm::APInt Size,
                                    const NamedDecl *ND) {
  if (Size != 1 || !ND) return false;

  const FieldDecl *FD = dyn_cast<FieldDecl>(ND);
  if (!FD) return false;

  // Don't consider sizes resulting from macro expansions or template argument
  // substitution to form C89 tail-padded arrays.

  TypeSourceInfo *TInfo = FD->getTypeSourceInfo();
  while (TInfo) {
    TypeLoc TL = TInfo->getTypeLoc();
    // Look through typedefs.
    const TypedefTypeLoc *TTL = dyn_cast<TypedefTypeLoc>(&TL);
    if (TTL) {
      const TypedefNameDecl *TDL = TTL->getTypedefNameDecl();
      TInfo = TDL->getTypeSourceInfo();
      continue;
    }
    ConstantArrayTypeLoc CTL = cast<ConstantArrayTypeLoc>(TL);
    const Expr *SizeExpr = dyn_cast<IntegerLiteral>(CTL.getSizeExpr());
    if (!SizeExpr || SizeExpr->getExprLoc().isMacroID())
      return false;
    break;
  }

  const RecordDecl *RD = dyn_cast<RecordDecl>(FD->getDeclContext());
  if (!RD) return false;
  if (RD->isUnion()) return false;
  if (const CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (!CRD->isStandardLayout()) return false;
  }

  // See if this is the last field decl in the record.
  const Decl *D = FD;
  while ((D = D->getNextDeclInContext()))
    if (isa<FieldDecl>(D))
      return false;
  return true;
}

void Sema::CheckArrayAccess(const Expr *BaseExpr, const Expr *IndexExpr,
                            const ArraySubscriptExpr *ASE,
                            bool AllowOnePastEnd, bool IndexNegated) {
  IndexExpr = IndexExpr->IgnoreParenImpCasts();
  if (IndexExpr->isValueDependent())
    return;

  const Type *EffectiveType = getElementType(BaseExpr);
  BaseExpr = BaseExpr->IgnoreParenCasts();
  const ConstantArrayType *ArrayTy =
    Context.getAsConstantArrayType(BaseExpr->getType());
  if (!ArrayTy)
    return;

  llvm::APSInt index;
  if (!IndexExpr->EvaluateAsInt(index, Context))
    return;
  if (IndexNegated)
    index = -index;

  const NamedDecl *ND = NULL;
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BaseExpr))
    ND = dyn_cast<NamedDecl>(DRE->getDecl());
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(BaseExpr))
    ND = dyn_cast<NamedDecl>(ME->getMemberDecl());

  if (index.isUnsigned() || !index.isNegative()) {
    llvm::APInt size = ArrayTy->getSize();
    if (!size.isStrictlyPositive())
      return;

    const Type* BaseType = getElementType(BaseExpr);
    if (BaseType != EffectiveType) {
      // Make sure we're comparing apples to apples when comparing index to size
      uint64_t ptrarith_typesize = Context.getTypeSize(EffectiveType);
      uint64_t array_typesize = Context.getTypeSize(BaseType);
      // Handle ptrarith_typesize being zero, such as when casting to void*
      if (!ptrarith_typesize) ptrarith_typesize = 1;
      if (ptrarith_typesize != array_typesize) {
        // There's a cast to a different size type involved
        uint64_t ratio = array_typesize / ptrarith_typesize;
        // TODO: Be smarter about handling cases where array_typesize is not a
        // multiple of ptrarith_typesize
        if (ptrarith_typesize * ratio == array_typesize)
          size *= llvm::APInt(size.getBitWidth(), ratio);
      }
    }

    if (size.getBitWidth() > index.getBitWidth())
      index = index.zext(size.getBitWidth());
    else if (size.getBitWidth() < index.getBitWidth())
      size = size.zext(index.getBitWidth());

    // For array subscripting the index must be less than size, but for pointer
    // arithmetic also allow the index (offset) to be equal to size since
    // computing the next address after the end of the array is legal and
    // commonly done e.g. in C++ iterators and range-based for loops.
    if (AllowOnePastEnd ? index.ule(size) : index.ult(size))
      return;

    // Also don't warn for arrays of size 1 which are members of some
    // structure. These are often used to approximate flexible arrays in C89
    // code.
    if (IsTailPaddedMemberArray(*this, size, ND))
      return;

    // Suppress the warning if the subscript expression (as identified by the
    // ']' location) and the index expression are both from macro expansions
    // within a system header.
    if (ASE) {
      SourceLocation RBracketLoc = SourceMgr.getSpellingLoc(
          ASE->getRBracketLoc());
      if (SourceMgr.isInSystemHeader(RBracketLoc)) {
        SourceLocation IndexLoc = SourceMgr.getSpellingLoc(
            IndexExpr->getLocStart());
        if (SourceMgr.isFromSameFile(RBracketLoc, IndexLoc))
          return;
      }
    }

    unsigned DiagID = diag::warn_ptr_arith_exceeds_bounds;
    if (ASE)
      DiagID = diag::warn_array_index_exceeds_bounds;

    DiagRuntimeBehavior(BaseExpr->getLocStart(), BaseExpr,
                        PDiag(DiagID) << index.toString(10, true)
                          << size.toString(10, true)
                          << (unsigned)size.getLimitedValue(~0U)
                          << IndexExpr->getSourceRange());
  } else {
    unsigned DiagID = diag::warn_array_index_precedes_bounds;
    if (!ASE) {
      DiagID = diag::warn_ptr_arith_precedes_bounds;
      if (index.isNegative()) index = -index;
    }

    DiagRuntimeBehavior(BaseExpr->getLocStart(), BaseExpr,
                        PDiag(DiagID) << index.toString(10, true)
                          << IndexExpr->getSourceRange());
  }

  if (!ND) {
    // Try harder to find a NamedDecl to point at in the note.
    while (const ArraySubscriptExpr *ASE =
           dyn_cast<ArraySubscriptExpr>(BaseExpr))
      BaseExpr = ASE->getBase()->IgnoreParenCasts();
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(BaseExpr))
      ND = dyn_cast<NamedDecl>(DRE->getDecl());
    if (const MemberExpr *ME = dyn_cast<MemberExpr>(BaseExpr))
      ND = dyn_cast<NamedDecl>(ME->getMemberDecl());
  }

  if (ND)
    DiagRuntimeBehavior(ND->getLocStart(), BaseExpr,
                        PDiag(diag::note_array_index_out_of_bounds)
                          << ND->getDeclName());
}

void Sema::CheckArrayAccess(const Expr *expr) {
  int AllowOnePastEnd = 0;
  while (expr) {
    expr = expr->IgnoreParenImpCasts();
    switch (expr->getStmtClass()) {
      case Stmt::ArraySubscriptExprClass: {
        const ArraySubscriptExpr *ASE = cast<ArraySubscriptExpr>(expr);
        CheckArrayAccess(ASE->getBase(), ASE->getIdx(), ASE,
                         AllowOnePastEnd > 0);
        return;
      }
      case Stmt::UnaryOperatorClass: {
        // Only unwrap the * and & unary operators
        const UnaryOperator *UO = cast<UnaryOperator>(expr);
        expr = UO->getSubExpr();
        switch (UO->getOpcode()) {
          case UO_AddrOf:
            AllowOnePastEnd++;
            break;
          case UO_Deref:
            AllowOnePastEnd--;
            break;
          default:
            return;
        }
        break;
      }
      case Stmt::ConditionalOperatorClass: {
        const ConditionalOperator *cond = cast<ConditionalOperator>(expr);
        if (const Expr *lhs = cond->getLHS())
          CheckArrayAccess(lhs);
        if (const Expr *rhs = cond->getRHS())
          CheckArrayAccess(rhs);
        return;
      }
      default:
        return;
    }
  }
}

//===--- CHECK: Objective-C retain cycles ----------------------------------//

namespace {
  struct RetainCycleOwner {
    RetainCycleOwner() : Variable(0), Indirect(false) {}
    VarDecl *Variable;
    SourceRange Range;
    SourceLocation Loc;
    bool Indirect;

    void setLocsFrom(Expr *e) {
      Loc = e->getExprLoc();
      Range = e->getSourceRange();
    }
  };
}

/// Consider whether capturing the given variable can possibly lead to
/// a retain cycle.
static bool considerVariable(VarDecl *var, Expr *ref, RetainCycleOwner &owner) {
  // In ARC, it's captured strongly iff the variable has __strong
  // lifetime.  In MRR, it's captured strongly if the variable is
  // __block and has an appropriate type.
  if (var->getType().getObjCLifetime() != Qualifiers::OCL_Strong)
    return false;

  owner.Variable = var;
  if (ref)
    owner.setLocsFrom(ref);
  return true;
}

static bool findRetainCycleOwner(Sema &S, Expr *e, RetainCycleOwner &owner) {
  while (true) {
    e = e->IgnoreParens();
    if (CastExpr *cast = dyn_cast<CastExpr>(e)) {
      switch (cast->getCastKind()) {
      case CK_BitCast:
      case CK_LValueBitCast:
      case CK_LValueToRValue:
      case CK_ARCReclaimReturnedObject:
        e = cast->getSubExpr();
        continue;

      default:
        return false;
      }
    }

    if (ObjCIvarRefExpr *ref = dyn_cast<ObjCIvarRefExpr>(e)) {
      ObjCIvarDecl *ivar = ref->getDecl();
      if (ivar->getType().getObjCLifetime() != Qualifiers::OCL_Strong)
        return false;

      // Try to find a retain cycle in the base.
      if (!findRetainCycleOwner(S, ref->getBase(), owner))
        return false;

      if (ref->isFreeIvar()) owner.setLocsFrom(ref);
      owner.Indirect = true;
      return true;
    }

    if (DeclRefExpr *ref = dyn_cast<DeclRefExpr>(e)) {
      VarDecl *var = dyn_cast<VarDecl>(ref->getDecl());
      if (!var) return false;
      return considerVariable(var, ref, owner);
    }

    if (MemberExpr *member = dyn_cast<MemberExpr>(e)) {
      if (member->isArrow()) return false;

      // Don't count this as an indirect ownership.
      e = member->getBase();
      continue;
    }

    if (PseudoObjectExpr *pseudo = dyn_cast<PseudoObjectExpr>(e)) {
      // Only pay attention to pseudo-objects on property references.
      ObjCPropertyRefExpr *pre
        = dyn_cast<ObjCPropertyRefExpr>(pseudo->getSyntacticForm()
                                              ->IgnoreParens());
      if (!pre) return false;
      if (pre->isImplicitProperty()) return false;
      ObjCPropertyDecl *property = pre->getExplicitProperty();
      if (!property->isRetaining() &&
          !(property->getPropertyIvarDecl() &&
            property->getPropertyIvarDecl()->getType()
              .getObjCLifetime() == Qualifiers::OCL_Strong))
          return false;

      owner.Indirect = true;
      if (pre->isSuperReceiver()) {
        owner.Variable = S.getCurMethodDecl()->getSelfDecl();
        if (!owner.Variable)
          return false;
        owner.Loc = pre->getLocation();
        owner.Range = pre->getSourceRange();
        return true;
      }
      e = const_cast<Expr*>(cast<OpaqueValueExpr>(pre->getBase())
                              ->getSourceExpr());
      continue;
    }

    // Array ivars?

    return false;
  }
}

namespace {
  struct FindCaptureVisitor : EvaluatedExprVisitor<FindCaptureVisitor> {
    FindCaptureVisitor(ASTContext &Context, VarDecl *variable)
      : EvaluatedExprVisitor<FindCaptureVisitor>(Context),
        Variable(variable), Capturer(0) {}

    VarDecl *Variable;
    Expr *Capturer;

    void VisitDeclRefExpr(DeclRefExpr *ref) {
      if (ref->getDecl() == Variable && !Capturer)
        Capturer = ref;
    }

    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *ref) {
      if (Capturer) return;
      Visit(ref->getBase());
      if (Capturer && ref->isFreeIvar())
        Capturer = ref;
    }

    void VisitBlockExpr(BlockExpr *block) {
      // Look inside nested blocks 
      if (block->getBlockDecl()->capturesVariable(Variable))
        Visit(block->getBlockDecl()->getBody());
    }
    
    void VisitOpaqueValueExpr(OpaqueValueExpr *OVE) {
      if (Capturer) return;
      if (OVE->getSourceExpr())
        Visit(OVE->getSourceExpr());
    }
  };
}

/// Check whether the given argument is a block which captures a
/// variable.
static Expr *findCapturingExpr(Sema &S, Expr *e, RetainCycleOwner &owner) {
  assert(owner.Variable && owner.Loc.isValid());

  e = e->IgnoreParenCasts();

  // Look through [^{...} copy] and Block_copy(^{...}).
  if (ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(e)) {
    Selector Cmd = ME->getSelector();
    if (Cmd.isUnarySelector() && Cmd.getNameForSlot(0) == "copy") {
      e = ME->getInstanceReceiver();
      if (!e)
        return 0;
      e = e->IgnoreParenCasts();
    }
  } else if (CallExpr *CE = dyn_cast<CallExpr>(e)) {
    if (CE->getNumArgs() == 1) {
      FunctionDecl *Fn = dyn_cast_or_null<FunctionDecl>(CE->getCalleeDecl());
      if (Fn) {
        const IdentifierInfo *FnI = Fn->getIdentifier();
        if (FnI && FnI->isStr("_Block_copy")) {
          e = CE->getArg(0)->IgnoreParenCasts();
        }
      }
    }
  }
  
  BlockExpr *block = dyn_cast<BlockExpr>(e);
  if (!block || !block->getBlockDecl()->capturesVariable(owner.Variable))
    return 0;

  FindCaptureVisitor visitor(S.Context, owner.Variable);
  visitor.Visit(block->getBlockDecl()->getBody());
  return visitor.Capturer;
}

static void diagnoseRetainCycle(Sema &S, Expr *capturer,
                                RetainCycleOwner &owner) {
  assert(capturer);
  assert(owner.Variable && owner.Loc.isValid());

  S.Diag(capturer->getExprLoc(), diag::warn_arc_retain_cycle)
    << owner.Variable << capturer->getSourceRange();
  S.Diag(owner.Loc, diag::note_arc_retain_cycle_owner)
    << owner.Indirect << owner.Range;
}

/// Check for a keyword selector that starts with the word 'add' or
/// 'set'.
static bool isSetterLikeSelector(Selector sel) {
  if (sel.isUnarySelector()) return false;

  StringRef str = sel.getNameForSlot(0);
  while (!str.empty() && str.front() == '_') str = str.substr(1);
  if (str.startswith("set"))
    str = str.substr(3);
  else if (str.startswith("add")) {
    // Specially whitelist 'addOperationWithBlock:'.
    if (sel.getNumArgs() == 1 && str.startswith("addOperationWithBlock"))
      return false;
    str = str.substr(3);
  }
  else
    return false;

  if (str.empty()) return true;
  return !islower(str.front());
}

/// Check a message send to see if it's likely to cause a retain cycle.
void Sema::checkRetainCycles(ObjCMessageExpr *msg) {
  // Only check instance methods whose selector looks like a setter.
  if (!msg->isInstanceMessage() || !isSetterLikeSelector(msg->getSelector()))
    return;

  // Try to find a variable that the receiver is strongly owned by.
  RetainCycleOwner owner;
  if (msg->getReceiverKind() == ObjCMessageExpr::Instance) {
    if (!findRetainCycleOwner(*this, msg->getInstanceReceiver(), owner))
      return;
  } else {
    assert(msg->getReceiverKind() == ObjCMessageExpr::SuperInstance);
    owner.Variable = getCurMethodDecl()->getSelfDecl();
    owner.Loc = msg->getSuperLoc();
    owner.Range = msg->getSuperLoc();
  }

  // Check whether the receiver is captured by any of the arguments.
  for (unsigned i = 0, e = msg->getNumArgs(); i != e; ++i)
    if (Expr *capturer = findCapturingExpr(*this, msg->getArg(i), owner))
      return diagnoseRetainCycle(*this, capturer, owner);
}

/// Check a property assign to see if it's likely to cause a retain cycle.
void Sema::checkRetainCycles(Expr *receiver, Expr *argument) {
  RetainCycleOwner owner;
  if (!findRetainCycleOwner(*this, receiver, owner))
    return;

  if (Expr *capturer = findCapturingExpr(*this, argument, owner))
    diagnoseRetainCycle(*this, capturer, owner);
}

void Sema::checkRetainCycles(VarDecl *Var, Expr *Init) {
  RetainCycleOwner Owner;
  if (!considerVariable(Var, /*DeclRefExpr=*/0, Owner))
    return;
  
  // Because we don't have an expression for the variable, we have to set the
  // location explicitly here.
  Owner.Loc = Var->getLocation();
  Owner.Range = Var->getSourceRange();
  
  if (Expr *Capturer = findCapturingExpr(*this, Init, Owner))
    diagnoseRetainCycle(*this, Capturer, Owner);
}

bool Sema::checkUnsafeAssigns(SourceLocation Loc,
                              QualType LHS, Expr *RHS) {
  Qualifiers::ObjCLifetime LT = LHS.getObjCLifetime();
  if (LT != Qualifiers::OCL_Weak && LT != Qualifiers::OCL_ExplicitNone)
    return false;
  // strip off any implicit cast added to get to the one arc-specific
  while (ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(RHS)) {
    if (cast->getCastKind() == CK_ARCConsumeObject) {
      Diag(Loc, diag::warn_arc_retained_assign)
        << (LT == Qualifiers::OCL_ExplicitNone) << 1
        << RHS->getSourceRange();
      return true;
    }
    RHS = cast->getSubExpr();
  }
  return false;
}

void Sema::checkUnsafeExprAssigns(SourceLocation Loc,
                              Expr *LHS, Expr *RHS) {
  QualType LHSType;
  // PropertyRef on LHS type need be directly obtained from
  // its declaration as it has a PsuedoType.
  ObjCPropertyRefExpr *PRE
    = dyn_cast<ObjCPropertyRefExpr>(LHS->IgnoreParens());
  if (PRE && !PRE->isImplicitProperty()) {
    const ObjCPropertyDecl *PD = PRE->getExplicitProperty();
    if (PD)
      LHSType = PD->getType();
  }
  
  if (LHSType.isNull())
    LHSType = LHS->getType();

  Qualifiers::ObjCLifetime LT = LHSType.getObjCLifetime();

  if (LT == Qualifiers::OCL_Weak) {
    DiagnosticsEngine::Level Level =
      Diags.getDiagnosticLevel(diag::warn_arc_repeated_use_of_weak, Loc);
    if (Level != DiagnosticsEngine::Ignored)
      getCurFunction()->markSafeWeakUse(LHS);
  }

  if (checkUnsafeAssigns(Loc, LHSType, RHS))
    return;

  // FIXME. Check for other life times.
  if (LT != Qualifiers::OCL_None)
    return;
  
  if (PRE) {
    if (PRE->isImplicitProperty())
      return;
    const ObjCPropertyDecl *PD = PRE->getExplicitProperty();
    if (!PD)
      return;
    
    unsigned Attributes = PD->getPropertyAttributes();
    if (Attributes & ObjCPropertyDecl::OBJC_PR_assign) {
      // when 'assign' attribute was not explicitly specified
      // by user, ignore it and rely on property type itself
      // for lifetime info.
      unsigned AsWrittenAttr = PD->getPropertyAttributesAsWritten();
      if (!(AsWrittenAttr & ObjCPropertyDecl::OBJC_PR_assign) &&
          LHSType->isObjCRetainableType())
        return;
        
      while (ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(RHS)) {
        if (cast->getCastKind() == CK_ARCConsumeObject) {
          Diag(Loc, diag::warn_arc_retained_property_assign)
          << RHS->getSourceRange();
          return;
        }
        RHS = cast->getSubExpr();
      }
    }
    else if (Attributes & ObjCPropertyDecl::OBJC_PR_weak) {
      while (ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(RHS)) {
        if (cast->getCastKind() == CK_ARCConsumeObject) {
          Diag(Loc, diag::warn_arc_retained_assign)
          << 0 << 0<< RHS->getSourceRange();
          return;
        }
        RHS = cast->getSubExpr();
      }
    }
  }
}

//===--- CHECK: Empty statement body (-Wempty-body) ---------------------===//

namespace {
bool ShouldDiagnoseEmptyStmtBody(const SourceManager &SourceMgr,
                                 SourceLocation StmtLoc,
                                 const NullStmt *Body) {
  // Do not warn if the body is a macro that expands to nothing, e.g:
  //
  // #define CALL(x)
  // if (condition)
  //   CALL(0);
  //
  if (Body->hasLeadingEmptyMacro())
    return false;

  // Get line numbers of statement and body.
  bool StmtLineInvalid;
  unsigned StmtLine = SourceMgr.getSpellingLineNumber(StmtLoc,
                                                      &StmtLineInvalid);
  if (StmtLineInvalid)
    return false;

  bool BodyLineInvalid;
  unsigned BodyLine = SourceMgr.getSpellingLineNumber(Body->getSemiLoc(),
                                                      &BodyLineInvalid);
  if (BodyLineInvalid)
    return false;

  // Warn if null statement and body are on the same line.
  if (StmtLine != BodyLine)
    return false;

  return true;
}
} // Unnamed namespace

void Sema::DiagnoseEmptyStmtBody(SourceLocation StmtLoc,
                                 const Stmt *Body,
                                 unsigned DiagID) {
  // Since this is a syntactic check, don't emit diagnostic for template
  // instantiations, this just adds noise.
  if (CurrentInstantiationScope)
    return;

  // The body should be a null statement.
  const NullStmt *NBody = dyn_cast<NullStmt>(Body);
  if (!NBody)
    return;

  // Do the usual checks.
  if (!ShouldDiagnoseEmptyStmtBody(SourceMgr, StmtLoc, NBody))
    return;

  Diag(NBody->getSemiLoc(), DiagID);
  Diag(NBody->getSemiLoc(), diag::note_empty_body_on_separate_line);
}

void Sema::DiagnoseEmptyLoopBody(const Stmt *S,
                                 const Stmt *PossibleBody) {
  assert(!CurrentInstantiationScope); // Ensured by caller

  SourceLocation StmtLoc;
  const Stmt *Body;
  unsigned DiagID;
  if (const ForStmt *FS = dyn_cast<ForStmt>(S)) {
    StmtLoc = FS->getRParenLoc();
    Body = FS->getBody();
    DiagID = diag::warn_empty_for_body;
  } else if (const WhileStmt *WS = dyn_cast<WhileStmt>(S)) {
    StmtLoc = WS->getCond()->getSourceRange().getEnd();
    Body = WS->getBody();
    DiagID = diag::warn_empty_while_body;
  } else
    return; // Neither `for' nor `while'.

  // The body should be a null statement.
  const NullStmt *NBody = dyn_cast<NullStmt>(Body);
  if (!NBody)
    return;

  // Skip expensive checks if diagnostic is disabled.
  if (Diags.getDiagnosticLevel(DiagID, NBody->getSemiLoc()) ==
          DiagnosticsEngine::Ignored)
    return;

  // Do the usual checks.
  if (!ShouldDiagnoseEmptyStmtBody(SourceMgr, StmtLoc, NBody))
    return;

  // `for(...);' and `while(...);' are popular idioms, so in order to keep
  // noise level low, emit diagnostics only if for/while is followed by a
  // CompoundStmt, e.g.:
  //    for (int i = 0; i < n; i++);
  //    {
  //      a(i);
  //    }
  // or if for/while is followed by a statement with more indentation
  // than for/while itself:
  //    for (int i = 0; i < n; i++);
  //      a(i);
  bool ProbableTypo = isa<CompoundStmt>(PossibleBody);
  if (!ProbableTypo) {
    bool BodyColInvalid;
    unsigned BodyCol = SourceMgr.getPresumedColumnNumber(
                             PossibleBody->getLocStart(),
                             &BodyColInvalid);
    if (BodyColInvalid)
      return;

    bool StmtColInvalid;
    unsigned StmtCol = SourceMgr.getPresumedColumnNumber(
                             S->getLocStart(),
                             &StmtColInvalid);
    if (StmtColInvalid)
      return;

    if (BodyCol > StmtCol)
      ProbableTypo = true;
  }

  if (ProbableTypo) {
    Diag(NBody->getSemiLoc(), DiagID);
    Diag(NBody->getSemiLoc(), diag::note_empty_body_on_separate_line);
  }
}

//===--- Layout compatibility ----------------------------------------------//

namespace {

bool isLayoutCompatible(ASTContext &C, QualType T1, QualType T2);

/// \brief Check if two enumeration types are layout-compatible.
bool isLayoutCompatible(ASTContext &C, EnumDecl *ED1, EnumDecl *ED2) {
  // C++11 [dcl.enum] p8:
  // Two enumeration types are layout-compatible if they have the same
  // underlying type.
  return ED1->isComplete() && ED2->isComplete() &&
         C.hasSameType(ED1->getIntegerType(), ED2->getIntegerType());
}

/// \brief Check if two fields are layout-compatible.
bool isLayoutCompatible(ASTContext &C, FieldDecl *Field1, FieldDecl *Field2) {
  if (!isLayoutCompatible(C, Field1->getType(), Field2->getType()))
    return false;

  if (Field1->isBitField() != Field2->isBitField())
    return false;

  if (Field1->isBitField()) {
    // Make sure that the bit-fields are the same length.
    unsigned Bits1 = Field1->getBitWidthValue(C);
    unsigned Bits2 = Field2->getBitWidthValue(C);

    if (Bits1 != Bits2)
      return false;
  }

  return true;
}

/// \brief Check if two standard-layout structs are layout-compatible.
/// (C++11 [class.mem] p17)
bool isLayoutCompatibleStruct(ASTContext &C,
                              RecordDecl *RD1,
                              RecordDecl *RD2) {
  // If both records are C++ classes, check that base classes match.
  if (const CXXRecordDecl *D1CXX = dyn_cast<CXXRecordDecl>(RD1)) {
    // If one of records is a CXXRecordDecl we are in C++ mode,
    // thus the other one is a CXXRecordDecl, too.
    const CXXRecordDecl *D2CXX = cast<CXXRecordDecl>(RD2);
    // Check number of base classes.
    if (D1CXX->getNumBases() != D2CXX->getNumBases())
      return false;

    // Check the base classes.
    for (CXXRecordDecl::base_class_const_iterator
               Base1 = D1CXX->bases_begin(),
           BaseEnd1 = D1CXX->bases_end(),
              Base2 = D2CXX->bases_begin();
         Base1 != BaseEnd1;
         ++Base1, ++Base2) {
      if (!isLayoutCompatible(C, Base1->getType(), Base2->getType()))
        return false;
    }
  } else if (const CXXRecordDecl *D2CXX = dyn_cast<CXXRecordDecl>(RD2)) {
    // If only RD2 is a C++ class, it should have zero base classes.
    if (D2CXX->getNumBases() > 0)
      return false;
  }

  // Check the fields.
  RecordDecl::field_iterator Field2 = RD2->field_begin(),
                             Field2End = RD2->field_end(),
                             Field1 = RD1->field_begin(),
                             Field1End = RD1->field_end();
  for ( ; Field1 != Field1End && Field2 != Field2End; ++Field1, ++Field2) {
    if (!isLayoutCompatible(C, *Field1, *Field2))
      return false;
  }
  if (Field1 != Field1End || Field2 != Field2End)
    return false;

  return true;
}

/// \brief Check if two standard-layout unions are layout-compatible.
/// (C++11 [class.mem] p18)
bool isLayoutCompatibleUnion(ASTContext &C,
                             RecordDecl *RD1,
                             RecordDecl *RD2) {
  llvm::SmallPtrSet<FieldDecl *, 8> UnmatchedFields;
  for (RecordDecl::field_iterator Field2 = RD2->field_begin(),
                                  Field2End = RD2->field_end();
       Field2 != Field2End; ++Field2) {
    UnmatchedFields.insert(*Field2);
  }

  for (RecordDecl::field_iterator Field1 = RD1->field_begin(),
                                  Field1End = RD1->field_end();
       Field1 != Field1End; ++Field1) {
    llvm::SmallPtrSet<FieldDecl *, 8>::iterator
        I = UnmatchedFields.begin(),
        E = UnmatchedFields.end();

    for ( ; I != E; ++I) {
      if (isLayoutCompatible(C, *Field1, *I)) {
        bool Result = UnmatchedFields.erase(*I);
        (void) Result;
        assert(Result);
        break;
      }
    }
    if (I == E)
      return false;
  }

  return UnmatchedFields.empty();
}

bool isLayoutCompatible(ASTContext &C, RecordDecl *RD1, RecordDecl *RD2) {
  if (RD1->isUnion() != RD2->isUnion())
    return false;

  if (RD1->isUnion())
    return isLayoutCompatibleUnion(C, RD1, RD2);
  else
    return isLayoutCompatibleStruct(C, RD1, RD2);
}

/// \brief Check if two types are layout-compatible in C++11 sense.
bool isLayoutCompatible(ASTContext &C, QualType T1, QualType T2) {
  if (T1.isNull() || T2.isNull())
    return false;

  // C++11 [basic.types] p11:
  // If two types T1 and T2 are the same type, then T1 and T2 are
  // layout-compatible types.
  if (C.hasSameType(T1, T2))
    return true;

  T1 = T1.getCanonicalType().getUnqualifiedType();
  T2 = T2.getCanonicalType().getUnqualifiedType();

  const Type::TypeClass TC1 = T1->getTypeClass();
  const Type::TypeClass TC2 = T2->getTypeClass();

  if (TC1 != TC2)
    return false;

  if (TC1 == Type::Enum) {
    return isLayoutCompatible(C,
                              cast<EnumType>(T1)->getDecl(),
                              cast<EnumType>(T2)->getDecl());
  } else if (TC1 == Type::Record) {
    if (!T1->isStandardLayoutType() || !T2->isStandardLayoutType())
      return false;

    return isLayoutCompatible(C,
                              cast<RecordType>(T1)->getDecl(),
                              cast<RecordType>(T2)->getDecl());
  }

  return false;
}
}

//===--- CHECK: pointer_with_type_tag attribute: datatypes should match ----//

namespace {
/// \brief Given a type tag expression find the type tag itself.
///
/// \param TypeExpr Type tag expression, as it appears in user's code.
///
/// \param VD Declaration of an identifier that appears in a type tag.
///
/// \param MagicValue Type tag magic value.
bool FindTypeTagExpr(const Expr *TypeExpr, const ASTContext &Ctx,
                     const ValueDecl **VD, uint64_t *MagicValue) {
  while(true) {
    if (!TypeExpr)
      return false;

    TypeExpr = TypeExpr->IgnoreParenImpCasts()->IgnoreParenCasts();

    switch (TypeExpr->getStmtClass()) {
    case Stmt::UnaryOperatorClass: {
      const UnaryOperator *UO = cast<UnaryOperator>(TypeExpr);
      if (UO->getOpcode() == UO_AddrOf || UO->getOpcode() == UO_Deref) {
        TypeExpr = UO->getSubExpr();
        continue;
      }
      return false;
    }

    case Stmt::DeclRefExprClass: {
      const DeclRefExpr *DRE = cast<DeclRefExpr>(TypeExpr);
      *VD = DRE->getDecl();
      return true;
    }

    case Stmt::IntegerLiteralClass: {
      const IntegerLiteral *IL = cast<IntegerLiteral>(TypeExpr);
      llvm::APInt MagicValueAPInt = IL->getValue();
      if (MagicValueAPInt.getActiveBits() <= 64) {
        *MagicValue = MagicValueAPInt.getZExtValue();
        return true;
      } else
        return false;
    }

    case Stmt::BinaryConditionalOperatorClass:
    case Stmt::ConditionalOperatorClass: {
      const AbstractConditionalOperator *ACO =
          cast<AbstractConditionalOperator>(TypeExpr);
      bool Result;
      if (ACO->getCond()->EvaluateAsBooleanCondition(Result, Ctx)) {
        if (Result)
          TypeExpr = ACO->getTrueExpr();
        else
          TypeExpr = ACO->getFalseExpr();
        continue;
      }
      return false;
    }

    case Stmt::BinaryOperatorClass: {
      const BinaryOperator *BO = cast<BinaryOperator>(TypeExpr);
      if (BO->getOpcode() == BO_Comma) {
        TypeExpr = BO->getRHS();
        continue;
      }
      return false;
    }

    default:
      return false;
    }
  }
}

/// \brief Retrieve the C type corresponding to type tag TypeExpr.
///
/// \param TypeExpr Expression that specifies a type tag.
///
/// \param MagicValues Registered magic values.
///
/// \param FoundWrongKind Set to true if a type tag was found, but of a wrong
///        kind.
///
/// \param TypeInfo Information about the corresponding C type.
///
/// \returns true if the corresponding C type was found.
bool GetMatchingCType(
        const IdentifierInfo *ArgumentKind,
        const Expr *TypeExpr, const ASTContext &Ctx,
        const llvm::DenseMap<Sema::TypeTagMagicValue,
                             Sema::TypeTagData> *MagicValues,
        bool &FoundWrongKind,
        Sema::TypeTagData &TypeInfo) {
  FoundWrongKind = false;

  // Variable declaration that has type_tag_for_datatype attribute.
  const ValueDecl *VD = NULL;

  uint64_t MagicValue;

  if (!FindTypeTagExpr(TypeExpr, Ctx, &VD, &MagicValue))
    return false;

  if (VD) {
    for (specific_attr_iterator<TypeTagForDatatypeAttr>
             I = VD->specific_attr_begin<TypeTagForDatatypeAttr>(),
             E = VD->specific_attr_end<TypeTagForDatatypeAttr>();
         I != E; ++I) {
      if (I->getArgumentKind() != ArgumentKind) {
        FoundWrongKind = true;
        return false;
      }
      TypeInfo.Type = I->getMatchingCType();
      TypeInfo.LayoutCompatible = I->getLayoutCompatible();
      TypeInfo.MustBeNull = I->getMustBeNull();
      return true;
    }
    return false;
  }

  if (!MagicValues)
    return false;

  llvm::DenseMap<Sema::TypeTagMagicValue,
                 Sema::TypeTagData>::const_iterator I =
      MagicValues->find(std::make_pair(ArgumentKind, MagicValue));
  if (I == MagicValues->end())
    return false;

  TypeInfo = I->second;
  return true;
}
} // unnamed namespace

void Sema::RegisterTypeTagForDatatype(const IdentifierInfo *ArgumentKind,
                                      uint64_t MagicValue, QualType Type,
                                      bool LayoutCompatible,
                                      bool MustBeNull) {
  if (!TypeTagForDatatypeMagicValues)
    TypeTagForDatatypeMagicValues.reset(
        new llvm::DenseMap<TypeTagMagicValue, TypeTagData>);

  TypeTagMagicValue Magic(ArgumentKind, MagicValue);
  (*TypeTagForDatatypeMagicValues)[Magic] =
      TypeTagData(Type, LayoutCompatible, MustBeNull);
}

namespace {
bool IsSameCharType(QualType T1, QualType T2) {
  const BuiltinType *BT1 = T1->getAs<BuiltinType>();
  if (!BT1)
    return false;

  const BuiltinType *BT2 = T2->getAs<BuiltinType>();
  if (!BT2)
    return false;

  BuiltinType::Kind T1Kind = BT1->getKind();
  BuiltinType::Kind T2Kind = BT2->getKind();

  return (T1Kind == BuiltinType::SChar  && T2Kind == BuiltinType::Char_S) ||
         (T1Kind == BuiltinType::UChar  && T2Kind == BuiltinType::Char_U) ||
         (T1Kind == BuiltinType::Char_U && T2Kind == BuiltinType::UChar) ||
         (T1Kind == BuiltinType::Char_S && T2Kind == BuiltinType::SChar);
}
} // unnamed namespace

void Sema::CheckArgumentWithTypeTag(const ArgumentWithTypeTagAttr *Attr,
                                    const Expr * const *ExprArgs) {
  const IdentifierInfo *ArgumentKind = Attr->getArgumentKind();
  bool IsPointerAttr = Attr->getIsPointer();

  const Expr *TypeTagExpr = ExprArgs[Attr->getTypeTagIdx()];
  bool FoundWrongKind;
  TypeTagData TypeInfo;
  if (!GetMatchingCType(ArgumentKind, TypeTagExpr, Context,
                        TypeTagForDatatypeMagicValues.get(),
                        FoundWrongKind, TypeInfo)) {
    if (FoundWrongKind)
      Diag(TypeTagExpr->getExprLoc(),
           diag::warn_type_tag_for_datatype_wrong_kind)
        << TypeTagExpr->getSourceRange();
    return;
  }

  const Expr *ArgumentExpr = ExprArgs[Attr->getArgumentIdx()];
  if (IsPointerAttr) {
    // Skip implicit cast of pointer to `void *' (as a function argument).
    if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(ArgumentExpr))
      if (ICE->getType()->isVoidPointerType() &&
          ICE->getCastKind() == CK_BitCast)
        ArgumentExpr = ICE->getSubExpr();
  }
  QualType ArgumentType = ArgumentExpr->getType();

  // Passing a `void*' pointer shouldn't trigger a warning.
  if (IsPointerAttr && ArgumentType->isVoidPointerType())
    return;

  if (TypeInfo.MustBeNull) {
    // Type tag with matching void type requires a null pointer.
    if (!ArgumentExpr->isNullPointerConstant(Context,
                                             Expr::NPC_ValueDependentIsNotNull)) {
      Diag(ArgumentExpr->getExprLoc(),
           diag::warn_type_safety_null_pointer_required)
          << ArgumentKind->getName()
          << ArgumentExpr->getSourceRange()
          << TypeTagExpr->getSourceRange();
    }
    return;
  }

  QualType RequiredType = TypeInfo.Type;
  if (IsPointerAttr)
    RequiredType = Context.getPointerType(RequiredType);

  bool mismatch = false;
  if (!TypeInfo.LayoutCompatible) {
    mismatch = !Context.hasSameType(ArgumentType, RequiredType);

    // C++11 [basic.fundamental] p1:
    // Plain char, signed char, and unsigned char are three distinct types.
    //
    // But we treat plain `char' as equivalent to `signed char' or `unsigned
    // char' depending on the current char signedness mode.
    if (mismatch)
      if ((IsPointerAttr && IsSameCharType(ArgumentType->getPointeeType(),
                                           RequiredType->getPointeeType())) ||
          (!IsPointerAttr && IsSameCharType(ArgumentType, RequiredType)))
        mismatch = false;
  } else
    if (IsPointerAttr)
      mismatch = !isLayoutCompatible(Context,
                                     ArgumentType->getPointeeType(),
                                     RequiredType->getPointeeType());
    else
      mismatch = !isLayoutCompatible(Context, ArgumentType, RequiredType);

  if (mismatch)
    Diag(ArgumentExpr->getExprLoc(), diag::warn_type_safety_type_mismatch)
        << ArgumentType << ArgumentKind->getName()
        << TypeInfo.LayoutCompatible << RequiredType
        << ArgumentExpr->getSourceRange()
        << TypeTagExpr->getSourceRange();
}
