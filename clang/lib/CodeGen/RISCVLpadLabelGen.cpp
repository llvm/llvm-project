//===--- RISCVLpadLabelGen.cpp - func-sig based lpad label gen ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements lpad label generation helpers from RISCVLpadLabelGen.h
//
//===----------------------------------------------------------------------===//

#include "RISCVLpadLabelGen.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/RISCVZicfilpHash.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::CodeGen;

namespace clang {
namespace CodeGen {
QualType getRISCVLpadCanonicalSignature(ASTContext &Ctx, QualType Ty,
                                        SourceLocation Loc = SourceLocation());

QualType substituteCovariantReturn(ASTContext &Ctx, QualType Ty);

QualType getSyntheticVType(ASTContext &Ctx);

QualType buildMainSigTy(ASTContext &Ctx);

} // namespace CodeGen
} // namespace clang

namespace {

enum class TemplateArgScanResult {
  Safe,
  ContainsWCharT,
  ContainsExceptionSpec,
  Unsupported,
};

/// FIXME: template arguments are not handled correctly (unsupported).
/// supports: pointers, references, arrays, member-pointer pointees,
/// function-proto/no-proto, ParenType, TypedefType, Block pointers
class SignatureNormalizer : public TypeVisitor<SignatureNormalizer, QualType> {
  ASTContext &Ctx;
  QualType UnderlyingWCharTy;

  static QualType computeUnderlyingWCharTy(ASTContext &Ctx) {
    switch (Ctx.getTargetInfo().getWCharType()) {
    case TargetInfo::SignedInt:
      return Ctx.IntTy;
    case TargetInfo::UnsignedInt:
      return Ctx.UnsignedIntTy;
    case TargetInfo::SignedLong:
      return Ctx.LongTy;
    case TargetInfo::UnsignedLong:
      return Ctx.UnsignedLongTy;
    case TargetInfo::SignedShort:
      return Ctx.ShortTy;
    case TargetInfo::UnsignedShort:
      return Ctx.UnsignedShortTy;
    default:
      llvm_unreachable("unexpected wchar_t underlying type");
    }
  }

public:
  explicit SignatureNormalizer(ASTContext &Ctx)
      : Ctx(Ctx), UnderlyingWCharTy(computeUnderlyingWCharTy(Ctx)) {}

  QualType Visit(QualType T) {
    Qualifiers Quals = T.getQualifiers();
    QualType Result = TypeVisitor::Visit(T.getTypePtr());
    return Ctx.getQualifiedType(Result, Quals);
  }

  QualType VisitType(const Type *T) { return QualType(T, 0); }

  QualType VisitBuiltinType(const BuiltinType *T) {
    if (T->getKind() == BuiltinType::WChar_S ||
        T->getKind() == BuiltinType::WChar_U)
      return UnderlyingWCharTy;
    return QualType(T, 0);
  }

  QualType VisitPointerType(const PointerType *T) {
    return Ctx.getPointerType(Visit(T->getPointeeType()));
  }
  QualType VisitLValueReferenceType(const LValueReferenceType *T) {
    return Ctx.getLValueReferenceType(Visit(T->getPointeeType()));
  }
  QualType VisitRValueReferenceType(const RValueReferenceType *T) {
    return Ctx.getRValueReferenceType(Visit(T->getPointeeType()));
  }

  // Pointer-to-array 
  QualType VisitConstantArrayType(const ConstantArrayType *T) {
    return Ctx.getConstantArrayType(Visit(T->getElementType()), T->getSize(),
                                    T->getSizeExpr(), T->getSizeModifier(),
                                    T->getIndexTypeCVRQualifiers());
  }
  QualType VisitIncompleteArrayType(const IncompleteArrayType *T) {
    return Ctx.getIncompleteArrayType(Visit(T->getElementType()),
                                      T->getSizeModifier(),
                                      T->getIndexTypeCVRQualifiers());
  }

  // Nested pointer-to-member
  QualType VisitMemberPointerType(const MemberPointerType *T) {
    QualType NewPointee = Visit(T->getPointeeType());
    return Ctx.getMemberPointerType(NewPointee, std::nullopt,
                                    T->getMostRecentCXXRecordDecl());
  }

  // Unprototyped (C-style) empty parameter list -> explicit (void).
  QualType VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.ExtInfo = T->getExtInfo();
    return Ctx.getFunctionType(Visit(T->getReturnType()), {}, EPI);
  }

  QualType VisitFunctionProtoType(const FunctionProtoType *T) {
    SmallVector<QualType, 8> Params;
    for (QualType P : T->getParamTypes())
      Params.push_back(Visit(P));

    FunctionProtoType::ExtProtoInfo EPI = T->getExtProtoInfo();
    EPI.ExceptionSpec =
        FunctionProtoType::ExceptionSpecInfo(); // strip, at every level

    return Ctx.getFunctionType(Visit(T->getReturnType()), Params, EPI);
  }

  QualType VisitParenType(const ParenType *T) {
    return Visit(T->getInnerType());
  }

  QualType VisitTypedefType(const TypedefType *T) {
    return Visit(T->desugar());
  }

  QualType VisitBlockPointerType(const BlockPointerType *T) {
    return Ctx.getBlockPointerType(Visit(T->getPointeeType()));
  }

  QualType VisitVariableArrayType(const VariableArrayType *T) {
    return Ctx.getVariableArrayType(Visit(T->getElementType()),
                                    T->getSizeExpr(), T->getSizeModifier(),
                                    T->getIndexTypeCVRQualifiers());
  }
};

} // end anonymous namespace

QualType clang::CodeGen::buildMainSigTy(ASTContext &Ctx) {
  QualType CharPtrPtr = Ctx.getPointerType(Ctx.getPointerType(Ctx.CharTy));
  FunctionProtoType::ExtProtoInfo
      EPI; // defaults are fine: no exceptions, normal CC
  return Ctx.getFunctionType(Ctx.IntTy, {Ctx.IntTy, CharPtrPtr}, EPI);
}

QualType clang::CodeGen::buildBlockCopyHelperSigTy(ASTContext &Ctx) {
  QualType VoidPtr = Ctx.getPointerType(Ctx.VoidTy);
  FunctionProtoType::ExtProtoInfo EPI;
  return Ctx.getFunctionType(Ctx.VoidTy, {VoidPtr, VoidPtr}, EPI);
}

QualType clang::CodeGen::buildDtorSigTy(ASTContext &Ctx) {
  QualType VoidPtr = Ctx.getPointerType(Ctx.VoidTy);
  FunctionProtoType::ExtProtoInfo EPI;
  return Ctx.getFunctionType(Ctx.VoidTy, {VoidPtr}, EPI);
}

QualType clang::CodeGen::buildVoidPtrToVoidPtrSigTy(ASTContext &Ctx) {
  QualType VoidPtr = Ctx.getPointerType(Ctx.VoidTy);
  FunctionProtoType::ExtProtoInfo EPI;
  return Ctx.getFunctionType(VoidPtr, {VoidPtr}, EPI);
}

uint32_t
clang::CodeGen::computeRISCVLpadLabel(const llvm::MD5::MD5Result &Result) {
  return llvm::RISCVCFI::reduceZicfilpFuncSigLabel(Result);
}

uint32_t clang::CodeGen::computeRISCVLpadLabel(llvm::StringRef Str) {
  return llvm::RISCVCFI::getZicfilpFuncSigLabel(Str);
}

// apply signature normalization according to psABI draft
// https://github.com/riscv-non-isa/riscv-elf-psabi-doc/pull/434, then delegate
// to Itanium Mangler.
std::string
clang::CodeGen::getRISCVLpadSignatureString(ASTContext &Ctx,
                                            ItaniumMangleContext &MC,
                                            QualType Ty, SourceLocation Loc) {
  Ty = getRISCVLpadCanonicalSignature(Ctx, Ty, Loc);

  std::string MangledName;
  llvm::raw_string_ostream Out(MangledName);
  MC.mangleCXXRTTIName(Ty, Out);

  assert(llvm::StringRef(MangledName).starts_with("_ZTS") &&
         "unexpected RTTI name mangling format");
  return MangledName.substr(4);
}

std::string clang::CodeGen::getRISCVLpadSignatureString(
    ASTContext &Ctx, ItaniumMangleContext &MC, const FunctionDecl *FD) {
  // special cases
  if (FD->isMain())
    return getRISCVLpadSignatureString(Ctx, MC, buildMainSigTy(Ctx));
  if (isa<CXXDestructorDecl>(FD))
    return getRISCVLpadSignatureString(Ctx, MC, buildDtorSigTy(Ctx));

  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (!MD->isStatic()) {
      QualType MPTy = Ctx.getMemberPointerType(MD->getType(), std::nullopt,
                                               MD->getParent());
      return getRISCVLpadSignatureString(
          Ctx, MC, MPTy, FD->getLocation()); // v-substitution happens
    }
    // later, during canonicalization
    // Static member: falls through, treated as non-member below.
  }

  // Free function / static member function: use the FunctionProtoType
  // directly (exception-spec stripping, wchar_t normalization, empty
  // param list -> void, all applied before/inside the type overload).
  return getRISCVLpadSignatureString(Ctx, MC, FD->getType(), FD->getLocation());
}

uint32_t clang::CodeGen::createRISCVEntryLpadTypeId(ASTContext &Ctx,
                                                    ItaniumMangleContext &MC,
                                                    const FunctionDecl *FD) {
  if (FD->getIdentifier() && FD->getName() == "_dl_runtime_resolve" &&
      FD->getDeclContext()->getRedeclContext()->isTranslationUnit() &&
      FD->isExternC() && FD->isExternallyVisible())
    return 0;
  return createRISCVCallSiteLpadTypeId(Ctx, MC, FD);
}

uint32_t clang::CodeGen::createRISCVCallSiteLpadTypeId(ASTContext &Ctx,
                                                       ItaniumMangleContext &MC,
                                                       const FunctionDecl *FD) {
  return computeRISCVLpadLabel(
      llvm::StringRef(getRISCVLpadSignatureString(Ctx, MC, FD)));
}

uint32_t clang::CodeGen::createRISCVLpadTypeId(ASTContext &Ctx,
                                               ItaniumMangleContext &MC,
                                               QualType Ty,
                                               SourceLocation Loc) {
  return computeRISCVLpadLabel(
      llvm::StringRef(getRISCVLpadSignatureString(Ctx, MC, Ty, Loc)));
}

static bool isWCharT(QualType T) {
  if (const auto *BT = T->getAs<BuiltinType>())
    return BT->getKind() == BuiltinType::WChar_S ||
           BT->getKind() == BuiltinType::WChar_U;
  return false;
}

static TemplateArgScanResult
combineTemplateArgScanResults(TemplateArgScanResult LHS,
                              TemplateArgScanResult RHS) {
  // Precedence below is an arbitrary but deterministic tie-break for
  // inputs triggering more than one condition; only one diagnostic can
  // be reported per call.
  if (LHS == TemplateArgScanResult::ContainsWCharT ||
      RHS == TemplateArgScanResult::ContainsWCharT)
    return TemplateArgScanResult::ContainsWCharT;
  if (LHS == TemplateArgScanResult::ContainsExceptionSpec ||
      RHS == TemplateArgScanResult::ContainsExceptionSpec)
    return TemplateArgScanResult::ContainsExceptionSpec;
  if (LHS == TemplateArgScanResult::Unsupported ||
      RHS == TemplateArgScanResult::Unsupported)
    return TemplateArgScanResult::Unsupported;
  return TemplateArgScanResult::Safe;
}

static TemplateArgScanResult
scanTemplateArgumentForWCharT(const TemplateArgument &Arg);

static TemplateArgScanResult
scanTypeForWCharTInTemplateArgs(QualType T, bool WithinTemplateArgs = false) {
  if (WithinTemplateArgs && isWCharT(T))
    return TemplateArgScanResult::ContainsWCharT;
  if (const auto *RD = T->getAsCXXRecordDecl()) {
    if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
      TemplateArgScanResult Result = TemplateArgScanResult::Safe;

      for (const TemplateArgument &Arg : Spec->getTemplateArgs().asArray()) {
        Result = combineTemplateArgScanResults(
            Result, scanTemplateArgumentForWCharT(Arg));
        // ContainsWCharT outranks all other results; no need to keep going if
        // found.
        if (Result == TemplateArgScanResult::ContainsWCharT)
          return Result;
      }

      if (Result != TemplateArgScanResult::Safe)
        return Result;
    }
  }
  
  if (const auto *PT = T->getAs<PointerType>())
    return scanTypeForWCharTInTemplateArgs(PT->getPointeeType(),
                                           WithinTemplateArgs);
  if (const auto *RT = T->getAs<ReferenceType>())
    return scanTypeForWCharTInTemplateArgs(RT->getPointeeType(),
                                           WithinTemplateArgs);
  if (const auto *AT = T->getAsArrayTypeUnsafe())
    return scanTypeForWCharTInTemplateArgs(AT->getElementType(),
                                           WithinTemplateArgs);
  if (const auto *FPT = T->getAs<FunctionProtoType>()) {
    TemplateArgScanResult Result = TemplateArgScanResult::Safe;

    if (WithinTemplateArgs && FPT->hasExceptionSpec()) {
      Result = TemplateArgScanResult::ContainsExceptionSpec;
    }

    Result = combineTemplateArgScanResults(
        Result, scanTypeForWCharTInTemplateArgs(FPT->getReturnType(),
                                                WithinTemplateArgs));
    for (QualType ParamTy : FPT->getParamTypes())
      Result = combineTemplateArgScanResults(
          Result, scanTypeForWCharTInTemplateArgs(ParamTy, WithinTemplateArgs));
    return Result;
  }

  if (const auto *FNPT = T->getAs<FunctionNoProtoType>())
    return scanTypeForWCharTInTemplateArgs(FNPT->getReturnType(),
                                           WithinTemplateArgs);

  if (const auto *MPT = T->getAs<MemberPointerType>())
    return scanTypeForWCharTInTemplateArgs(MPT->getPointeeType(),
                                           WithinTemplateArgs);

  if (const auto *BPT = T->getAs<BlockPointerType>())
    return scanTypeForWCharTInTemplateArgs(BPT->getPointeeType(),
                                           WithinTemplateArgs);

  if (const auto *PT = dyn_cast<ParenType>(T.getTypePtr()))
    return scanTypeForWCharTInTemplateArgs(PT->getInnerType(),
                                           WithinTemplateArgs);

  if (const auto *TT = dyn_cast<TypedefType>(T.getTypePtr()))
    return scanTypeForWCharTInTemplateArgs(TT->desugar(), WithinTemplateArgs);

  return TemplateArgScanResult::Safe;
}

static TemplateArgScanResult
scanTemplateArgumentForWCharT(const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Type:
    return scanTypeForWCharTInTemplateArgs(Arg.getAsType(),
                                           /*WithinTemplateArgument=*/true);

  case TemplateArgument::Integral:
    return scanTypeForWCharTInTemplateArgs(Arg.getIntegralType(),
                                           /*WithinTemplateArgument=*/true);

  case TemplateArgument::NullPtr:
    return scanTypeForWCharTInTemplateArgs(Arg.getNullPtrType(),
                                           /*WithinTemplateArgument=*/true);

  case TemplateArgument::Pack: {
    TemplateArgScanResult Result = TemplateArgScanResult::Safe;
    for (const TemplateArgument &Element : Arg.pack_elements()) {
      Result = combineTemplateArgScanResults(
          Result, scanTemplateArgumentForWCharT(Element));
    }
    return Result;
  }

  case TemplateArgument::Null:
    return TemplateArgScanResult::Safe;

  case TemplateArgument::Expression:
  case TemplateArgument::Declaration:
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
  case TemplateArgument::StructuralValue:
    return TemplateArgScanResult::Unsupported;
  }
  llvm_unreachable("unknown TemplateArgument kind");
}

QualType clang::CodeGen::getRISCVLpadCanonicalSignature(ASTContext &Ctx,
                                                        QualType Ty,
                                                        SourceLocation Loc) {
  TemplateArgScanResult ScanResult =
      scanTypeForWCharTInTemplateArgs(Ty,
                                      /*WithinTemplateArgument=*/false);

  unsigned UnsupportedTemplateArgDiagID = Ctx.getDiagnostics().getCustomDiagID(
      DiagnosticsEngine::Error,
      "RISC-V landing pad label generation does not yet support "
      "this template argument form");

  unsigned WCharTInTemplateArgDiagID = Ctx.getDiagnostics().getCustomDiagID(
      DiagnosticsEngine::Error,
      "RISC-V landing pad label generation does not yet support "
      "template arguments containing wchar_t");

  unsigned ExceptionSpecTInTemplateArgDiagID =
      Ctx.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "RISC-V landing pad label generation does not yet support "
          "template arguments containing exception spec");

  switch (ScanResult) {
  case TemplateArgScanResult::Safe:
    break;

  case TemplateArgScanResult::ContainsWCharT:
    Ctx.getDiagnostics().Report(Loc, WCharTInTemplateArgDiagID);
    break;

  case TemplateArgScanResult::ContainsExceptionSpec:
    Ctx.getDiagnostics().Report(Loc, ExceptionSpecTInTemplateArgDiagID);
    break;

  case TemplateArgScanResult::Unsupported:
    Ctx.getDiagnostics().Report(Loc, UnsupportedTemplateArgDiagID);
    break;
  }

  SignatureNormalizer Normalizer(Ctx);

  if (const auto *MPT = Ty->getAs<MemberPointerType>()) {
    QualType VType = getSyntheticVType(Ctx);

    QualType PointeeTy =
        Normalizer.Visit(MPT->getPointeeType()); // generic, recursive
    PointeeTy =
        substituteCovariantReturn(Ctx, PointeeTy); // narrow, top-level only

    return Ctx.getMemberPointerType(PointeeTy, std::nullopt,
                                    VType->getAsCXXRecordDecl());
  }

  return Normalizer.Visit(Ty);
}

QualType clang::CodeGen::substituteCovariantReturn(ASTContext &Ctx,
                                                   QualType FnTy) {
  const auto *FPT = FnTy->getAs<FunctionProtoType>();
  if (!FPT)
    return FnTy;

  QualType Ret = FPT->getReturnType();
  QualType NewRet = Ret;

  if (const auto *PT = Ret->getAs<PointerType>()) {
    if (PT->getPointeeType()->isRecordType())
      NewRet = Ctx.getPointerType(Ctx.VoidTy);
  } else if (const auto *RT = Ret->getAs<LValueReferenceType>()) {
    if (RT->getPointeeType()->isRecordType())
      NewRet = Ctx.getLValueReferenceType(Ctx.VoidTy);
  } else if (const auto *RRT = Ret->getAs<RValueReferenceType>()) {
    if (RRT->getPointeeType()->isRecordType())
      NewRet = Ctx.getRValueReferenceType(Ctx.VoidTy);
  }

  if (NewRet == Ret)
    return FnTy;
  FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
  return Ctx.getFunctionType(NewRet, FPT->getParamTypes(), EPI);
}

QualType clang::CodeGen::getSyntheticVType(ASTContext &Ctx) {
  auto *RD = CXXRecordDecl::Create(
      Ctx, TagTypeKind::Class, Ctx.getTranslationUnitDecl(), SourceLocation(),
      SourceLocation(), &Ctx.Idents.get("v"));
  RD->startDefinition();
  RD->completeDefinition();
  return Ctx.getTagType(ElaboratedTypeKeyword::None, std::nullopt, RD,
                        /*OwnsTag=*/false);
}