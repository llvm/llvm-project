//===--- MicrosoftMangle.cpp - Microsoft Visual C++ Name Mangling ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ name mangling targeting the Microsoft Visual C++ ABI.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/JamCRC.h"

using namespace clang;

namespace {

/// \brief Retrieve the declaration context that should be used when mangling
/// the given declaration.
static const DeclContext *getEffectiveDeclContext(const Decl *D) {
  // The ABI assumes that lambda closure types that occur within
  // default arguments live in the context of the function. However, due to
  // the way in which Clang parses and creates function declarations, this is
  // not the case: the lambda closure type ends up living in the context
  // where the function itself resides, because the function declaration itself
  // had not yet been created. Fix the context here.
  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    if (RD->isLambda())
      if (ParmVarDecl *ContextParam =
              dyn_cast_or_null<ParmVarDecl>(RD->getLambdaContextDecl()))
        return ContextParam->getDeclContext();
  }

  // Perform the same check for block literals.
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    if (ParmVarDecl *ContextParam =
            dyn_cast_or_null<ParmVarDecl>(BD->getBlockManglingContextDecl()))
      return ContextParam->getDeclContext();
  }

  const DeclContext *DC = D->getDeclContext();
  if (const CapturedDecl *CD = dyn_cast<CapturedDecl>(DC))
    return getEffectiveDeclContext(CD);

  return DC;
}

static const DeclContext *getEffectiveParentContext(const DeclContext *DC) {
  return getEffectiveDeclContext(cast<Decl>(DC));
}

static const FunctionDecl *getStructor(const NamedDecl *ND) {
  if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(ND))
    return FTD->getTemplatedDecl();

  const auto *FD = cast<FunctionDecl>(ND);
  if (const auto *FTD = FD->getPrimaryTemplate())
    return FTD->getTemplatedDecl();

  return FD;
}

static bool isLambda(const NamedDecl *ND) {
  const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(ND);
  if (!Record)
    return false;

  return Record->isLambda();
}

/// MicrosoftMangleContextImpl - Overrides the default MangleContext for the
/// Microsoft Visual C++ ABI.
class MicrosoftMangleContextImpl : public MicrosoftMangleContext {
  typedef std::pair<const DeclContext *, IdentifierInfo *> DiscriminatorKeyTy;
  llvm::DenseMap<DiscriminatorKeyTy, unsigned> Discriminator;
  llvm::DenseMap<const NamedDecl *, unsigned> Uniquifier;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> LambdaIds;
  llvm::DenseMap<const NamedDecl *, unsigned> SEHFilterIds;
  llvm::DenseMap<const NamedDecl *, unsigned> SEHFinallyIds;

public:
  MicrosoftMangleContextImpl(ASTContext &Context, DiagnosticsEngine &Diags)
      : MicrosoftMangleContext(Context, Diags) {}
  bool shouldMangleCXXName(const NamedDecl *D) override;
  bool shouldMangleStringLiteral(const StringLiteral *SL) override;
  void mangleCXXName(const NamedDecl *D, raw_ostream &Out) override;
  void mangleVirtualMemPtrThunk(const CXXMethodDecl *MD,
                                raw_ostream &) override;
  void mangleThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk,
                   raw_ostream &) override;
  void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                          const ThisAdjustment &ThisAdjustment,
                          raw_ostream &) override;
  void mangleCXXVFTable(const CXXRecordDecl *Derived,
                        ArrayRef<const CXXRecordDecl *> BasePath,
                        raw_ostream &Out) override;
  void mangleCXXVBTable(const CXXRecordDecl *Derived,
                        ArrayRef<const CXXRecordDecl *> BasePath,
                        raw_ostream &Out) override;
  void mangleCXXVirtualDisplacementMap(const CXXRecordDecl *SrcRD,
                                       const CXXRecordDecl *DstRD,
                                       raw_ostream &Out) override;
  void mangleCXXThrowInfo(QualType T, bool IsConst, bool IsVolatile,
                          uint32_t NumEntries, raw_ostream &Out) override;
  void mangleCXXCatchableTypeArray(QualType T, uint32_t NumEntries,
                                   raw_ostream &Out) override;
  void mangleCXXCatchableType(QualType T, const CXXConstructorDecl *CD,
                              CXXCtorType CT, uint32_t Size, uint32_t NVOffset,
                              int32_t VBPtrOffset, uint32_t VBIndex,
                              raw_ostream &Out) override;
  void mangleCXXCatchHandlerType(QualType T, uint32_t Flags,
                                 raw_ostream &Out) override;
  void mangleCXXRTTI(QualType T, raw_ostream &Out) override;
  void mangleCXXRTTIName(QualType T, raw_ostream &Out) override;
  void mangleCXXRTTIBaseClassDescriptor(const CXXRecordDecl *Derived,
                                        uint32_t NVOffset, int32_t VBPtrOffset,
                                        uint32_t VBTableOffset, uint32_t Flags,
                                        raw_ostream &Out) override;
  void mangleCXXRTTIBaseClassArray(const CXXRecordDecl *Derived,
                                   raw_ostream &Out) override;
  void mangleCXXRTTIClassHierarchyDescriptor(const CXXRecordDecl *Derived,
                                             raw_ostream &Out) override;
  void
  mangleCXXRTTICompleteObjectLocator(const CXXRecordDecl *Derived,
                                     ArrayRef<const CXXRecordDecl *> BasePath,
                                     raw_ostream &Out) override;
  void mangleTypeName(QualType T, raw_ostream &) override;
  void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                     raw_ostream &) override;
  void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                     raw_ostream &) override;
  void mangleReferenceTemporary(const VarDecl *, unsigned ManglingNumber,
                                raw_ostream &) override;
  void mangleStaticGuardVariable(const VarDecl *D, raw_ostream &Out) override;
  void mangleThreadSafeStaticGuardVariable(const VarDecl *D, unsigned GuardNum,
                                           raw_ostream &Out) override;
  void mangleDynamicInitializer(const VarDecl *D, raw_ostream &Out) override;
  void mangleDynamicAtExitDestructor(const VarDecl *D,
                                     raw_ostream &Out) override;
  void mangleSEHFilterExpression(const NamedDecl *EnclosingDecl,
                                 raw_ostream &Out) override;
  void mangleSEHFinallyBlock(const NamedDecl *EnclosingDecl,
                             raw_ostream &Out) override;
  void mangleStringLiteral(const StringLiteral *SL, raw_ostream &Out) override;
  bool getNextDiscriminator(const NamedDecl *ND, unsigned &disc) {
    // Lambda closure types are already numbered.
    if (isLambda(ND))
      return false;

    const DeclContext *DC = getEffectiveDeclContext(ND);
    if (!DC->isFunctionOrMethod())
      return false;

    // Use the canonical number for externally visible decls.
    if (ND->isExternallyVisible()) {
      disc = getASTContext().getManglingNumber(ND);
      return true;
    }

    // Anonymous tags are already numbered.
    if (const TagDecl *Tag = dyn_cast<TagDecl>(ND)) {
      if (!Tag->hasNameForLinkage() &&
          !getASTContext().getDeclaratorForUnnamedTagDecl(Tag) &&
          !getASTContext().getTypedefNameForUnnamedTagDecl(Tag))
        return false;
    }

    // Make up a reasonable number for internal decls.
    unsigned &discriminator = Uniquifier[ND];
    if (!discriminator)
      discriminator = ++Discriminator[std::make_pair(DC, ND->getIdentifier())];
    disc = discriminator + 1;
    return true;
  }

  unsigned getLambdaId(const CXXRecordDecl *RD) {
    assert(RD->isLambda() && "RD must be a lambda!");
    assert(!RD->isExternallyVisible() && "RD must not be visible!");
    assert(RD->getLambdaManglingNumber() == 0 &&
           "RD must not have a mangling number!");
    std::pair<llvm::DenseMap<const CXXRecordDecl *, unsigned>::iterator, bool>
        Result = LambdaIds.insert(std::make_pair(RD, LambdaIds.size()));
    return Result.first->second;
  }

private:
  void mangleInitFiniStub(const VarDecl *D, raw_ostream &Out, char CharCode);
};

/// MicrosoftCXXNameMangler - Manage the mangling of a single name for the
/// Microsoft Visual C++ ABI.
class MicrosoftCXXNameMangler {
  MicrosoftMangleContextImpl &Context;
  raw_ostream &Out;

  /// The "structor" is the top-level declaration being mangled, if
  /// that's not a template specialization; otherwise it's the pattern
  /// for that specialization.
  const NamedDecl *Structor;
  unsigned StructorType;

  typedef llvm::SmallVector<std::string, 10> BackRefVec;
  BackRefVec NameBackReferences;

  typedef llvm::DenseMap<void *, unsigned> ArgBackRefMap;
  ArgBackRefMap TypeBackReferences;

  ASTContext &getASTContext() const { return Context.getASTContext(); }

  // FIXME: If we add support for __ptr32/64 qualifiers, then we should push
  // this check into mangleQualifiers().
  const bool PointersAre64Bit;

public:
  enum QualifierMangleMode { QMM_Drop, QMM_Mangle, QMM_Escape, QMM_Result };

  MicrosoftCXXNameMangler(MicrosoftMangleContextImpl &C, raw_ostream &Out_)
      : Context(C), Out(Out_), Structor(nullptr), StructorType(-1),
        PointersAre64Bit(C.getASTContext().getTargetInfo().getPointerWidth(0) ==
                         64) {}

  MicrosoftCXXNameMangler(MicrosoftMangleContextImpl &C, raw_ostream &Out_,
                          const CXXConstructorDecl *D, CXXCtorType Type)
      : Context(C), Out(Out_), Structor(getStructor(D)), StructorType(Type),
        PointersAre64Bit(C.getASTContext().getTargetInfo().getPointerWidth(0) ==
                         64) {}

  MicrosoftCXXNameMangler(MicrosoftMangleContextImpl &C, raw_ostream &Out_,
                          const CXXDestructorDecl *D, CXXDtorType Type)
      : Context(C), Out(Out_), Structor(getStructor(D)), StructorType(Type),
        PointersAre64Bit(C.getASTContext().getTargetInfo().getPointerWidth(0) ==
                         64) {}

  raw_ostream &getStream() const { return Out; }

  void mangle(const NamedDecl *D, StringRef Prefix = "\01?");
  void mangleName(const NamedDecl *ND);
  void mangleFunctionEncoding(const FunctionDecl *FD, bool ShouldMangle);
  void mangleVariableEncoding(const VarDecl *VD);
  void mangleMemberDataPointer(const CXXRecordDecl *RD, const ValueDecl *VD);
  void mangleMemberFunctionPointer(const CXXRecordDecl *RD,
                                   const CXXMethodDecl *MD);
  void mangleVirtualMemPtrThunk(
      const CXXMethodDecl *MD,
      const MicrosoftVTableContext::MethodVFTableLocation &ML);
  void mangleNumber(int64_t Number);
  void mangleType(QualType T, SourceRange Range,
                  QualifierMangleMode QMM = QMM_Mangle);
  void mangleFunctionType(const FunctionType *T,
                          const FunctionDecl *D = nullptr,
                          bool ForceThisQuals = false);
  void mangleNestedName(const NamedDecl *ND);

private:
  void mangleUnqualifiedName(const NamedDecl *ND) {
    mangleUnqualifiedName(ND, ND->getDeclName());
  }
  void mangleUnqualifiedName(const NamedDecl *ND, DeclarationName Name);
  void mangleSourceName(StringRef Name);
  void mangleOperatorName(OverloadedOperatorKind OO, SourceLocation Loc);
  void mangleCXXDtorType(CXXDtorType T);
  void mangleQualifiers(Qualifiers Quals, bool IsMember);
  void mangleRefQualifier(RefQualifierKind RefQualifier);
  void manglePointerCVQualifiers(Qualifiers Quals);
  void manglePointerExtQualifiers(Qualifiers Quals, QualType PointeeType);

  void mangleUnscopedTemplateName(const TemplateDecl *ND);
  void
  mangleTemplateInstantiationName(const TemplateDecl *TD,
                                  const TemplateArgumentList &TemplateArgs);
  void mangleObjCMethodName(const ObjCMethodDecl *MD);

  void mangleArgumentType(QualType T, SourceRange Range);

  // Declare manglers for every type class.
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) void mangleType(const CLASS##Type *T, \
                                            Qualifiers Quals, \
                                            SourceRange Range);
#include "clang/AST/TypeNodes.def"
#undef ABSTRACT_TYPE
#undef NON_CANONICAL_TYPE
#undef TYPE

  void mangleType(const TagDecl *TD);
  void mangleDecayedArrayType(const ArrayType *T);
  void mangleArrayType(const ArrayType *T);
  void mangleFunctionClass(const FunctionDecl *FD);
  void mangleCallingConvention(CallingConv CC);
  void mangleCallingConvention(const FunctionType *T);
  void mangleIntegerLiteral(const llvm::APSInt &Number, bool IsBoolean);
  void mangleExpression(const Expr *E);
  void mangleThrowSpecification(const FunctionProtoType *T);

  void mangleTemplateArgs(const TemplateDecl *TD,
                          const TemplateArgumentList &TemplateArgs);
  void mangleTemplateArg(const TemplateDecl *TD, const TemplateArgument &TA,
                         const NamedDecl *Parm);
};
}

bool MicrosoftMangleContextImpl::shouldMangleCXXName(const NamedDecl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    LanguageLinkage L = FD->getLanguageLinkage();
    // Overloadable functions need mangling.
    if (FD->hasAttr<OverloadableAttr>())
      return true;

    // The ABI expects that we would never mangle "typical" user-defined entry
    // points regardless of visibility or freestanding-ness.
    //
    // N.B. This is distinct from asking about "main".  "main" has a lot of
    // special rules associated with it in the standard while these
    // user-defined entry points are outside of the purview of the standard.
    // For example, there can be only one definition for "main" in a standards
    // compliant program; however nothing forbids the existence of wmain and
    // WinMain in the same translation unit.
    if (FD->isMSVCRTEntryPoint())
      return false;

    // C++ functions and those whose names are not a simple identifier need
    // mangling.
    if (!FD->getDeclName().isIdentifier() || L == CXXLanguageLinkage)
      return true;

    // C functions are not mangled.
    if (L == CLanguageLinkage)
      return false;
  }

  // Otherwise, no mangling is done outside C++ mode.
  if (!getASTContext().getLangOpts().CPlusPlus)
    return false;

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    // C variables are not mangled.
    if (VD->isExternC())
      return false;

    // Variables at global scope with non-internal linkage are not mangled.
    const DeclContext *DC = getEffectiveDeclContext(D);
    // Check for extern variable declared locally.
    if (DC->isFunctionOrMethod() && D->hasLinkage())
      while (!DC->isNamespace() && !DC->isTranslationUnit())
        DC = getEffectiveParentContext(DC);

    if (DC->isTranslationUnit() && D->getFormalLinkage() == InternalLinkage &&
        !isa<VarTemplateSpecializationDecl>(D) &&
        D->getIdentifier() != nullptr)
      return false;
  }

  return true;
}

bool
MicrosoftMangleContextImpl::shouldMangleStringLiteral(const StringLiteral *SL) {
  return true;
}

void MicrosoftCXXNameMangler::mangle(const NamedDecl *D, StringRef Prefix) {
  // MSVC doesn't mangle C++ names the same way it mangles extern "C" names.
  // Therefore it's really important that we don't decorate the
  // name with leading underscores or leading/trailing at signs. So, by
  // default, we emit an asm marker at the start so we get the name right.
  // Callers can override this with a custom prefix.

  // <mangled-name> ::= ? <name> <type-encoding>
  Out << Prefix;
  mangleName(D);
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    mangleFunctionEncoding(FD, Context.shouldMangleDeclName(FD));
  else if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    mangleVariableEncoding(VD);
  else {
    // TODO: Fields? Can MSVC even mangle them?
    // Issue a diagnostic for now.
    DiagnosticsEngine &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(
        DiagnosticsEngine::Error, "cannot mangle this declaration yet");
    Diags.Report(D->getLocation(), DiagID) << D->getSourceRange();
  }
}

void MicrosoftCXXNameMangler::mangleFunctionEncoding(const FunctionDecl *FD,
                                                     bool ShouldMangle) {
  // <type-encoding> ::= <function-class> <function-type>

  // Since MSVC operates on the type as written and not the canonical type, it
  // actually matters which decl we have here.  MSVC appears to choose the
  // first, since it is most likely to be the declaration in a header file.
  FD = FD->getFirstDecl();

  // We should never ever see a FunctionNoProtoType at this point.
  // We don't even know how to mangle their types anyway :).
  const FunctionProtoType *FT = FD->getType()->castAs<FunctionProtoType>();

  // extern "C" functions can hold entities that must be mangled.
  // As it stands, these functions still need to get expressed in the full
  // external name.  They have their class and type omitted, replaced with '9'.
  if (ShouldMangle) {
    // We would like to mangle all extern "C" functions using this additional
    // component but this would break compatibility with MSVC's behavior.
    // Instead, do this when we know that compatibility isn't important (in
    // other words, when it is an overloaded extern "C" funciton).
    if (FD->isExternC() && FD->hasAttr<OverloadableAttr>())
      Out << "$$J0";

    mangleFunctionClass(FD);

    mangleFunctionType(FT, FD);
  } else {
    Out << '9';
  }
}

void MicrosoftCXXNameMangler::mangleVariableEncoding(const VarDecl *VD) {
  // <type-encoding> ::= <storage-class> <variable-type>
  // <storage-class> ::= 0  # private static member
  //                 ::= 1  # protected static member
  //                 ::= 2  # public static member
  //                 ::= 3  # global
  //                 ::= 4  # static local

  // The first character in the encoding (after the name) is the storage class.
  if (VD->isStaticDataMember()) {
    // If it's a static member, it also encodes the access level.
    switch (VD->getAccess()) {
      default:
      case AS_private: Out << '0'; break;
      case AS_protected: Out << '1'; break;
      case AS_public: Out << '2'; break;
    }
  }
  else if (!VD->isStaticLocal())
    Out << '3';
  else
    Out << '4';
  // Now mangle the type.
  // <variable-type> ::= <type> <cvr-qualifiers>
  //                 ::= <type> <pointee-cvr-qualifiers> # pointers, references
  // Pointers and references are odd. The type of 'int * const foo;' gets
  // mangled as 'QAHA' instead of 'PAHB', for example.
  SourceRange SR = VD->getSourceRange();
  QualType Ty = VD->getType();
  if (Ty->isPointerType() || Ty->isReferenceType() ||
      Ty->isMemberPointerType()) {
    mangleType(Ty, SR, QMM_Drop);
    manglePointerExtQualifiers(
        Ty.getDesugaredType(getASTContext()).getLocalQualifiers(), QualType());
    if (const MemberPointerType *MPT = Ty->getAs<MemberPointerType>()) {
      mangleQualifiers(MPT->getPointeeType().getQualifiers(), true);
      // Member pointers are suffixed with a back reference to the member
      // pointer's class name.
      mangleName(MPT->getClass()->getAsCXXRecordDecl());
    } else
      mangleQualifiers(Ty->getPointeeType().getQualifiers(), false);
  } else if (const ArrayType *AT = getASTContext().getAsArrayType(Ty)) {
    // Global arrays are funny, too.
    mangleDecayedArrayType(AT);
    if (AT->getElementType()->isArrayType())
      Out << 'A';
    else
      mangleQualifiers(Ty.getQualifiers(), false);
  } else {
    mangleType(Ty, SR, QMM_Drop);
    mangleQualifiers(Ty.getQualifiers(), false);
  }
}

void MicrosoftCXXNameMangler::mangleMemberDataPointer(const CXXRecordDecl *RD,
                                                      const ValueDecl *VD) {
  // <member-data-pointer> ::= <integer-literal>
  //                       ::= $F <number> <number>
  //                       ::= $G <number> <number> <number>

  int64_t FieldOffset;
  int64_t VBTableOffset;
  MSInheritanceAttr::Spelling IM = RD->getMSInheritanceModel();
  if (VD) {
    FieldOffset = getASTContext().getFieldOffset(VD);
    assert(FieldOffset % getASTContext().getCharWidth() == 0 &&
           "cannot take address of bitfield");
    FieldOffset /= getASTContext().getCharWidth();

    VBTableOffset = 0;

    if (IM == MSInheritanceAttr::Keyword_virtual_inheritance)
      FieldOffset -= getASTContext().getOffsetOfBaseWithVBPtr(RD).getQuantity();
  } else {
    FieldOffset = RD->nullFieldOffsetIsZero() ? 0 : -1;

    VBTableOffset = -1;
  }

  char Code = '\0';
  switch (IM) {
  case MSInheritanceAttr::Keyword_single_inheritance:      Code = '0'; break;
  case MSInheritanceAttr::Keyword_multiple_inheritance:    Code = '0'; break;
  case MSInheritanceAttr::Keyword_virtual_inheritance:     Code = 'F'; break;
  case MSInheritanceAttr::Keyword_unspecified_inheritance: Code = 'G'; break;
  }

  Out << '$' << Code;

  mangleNumber(FieldOffset);

  // The C++ standard doesn't allow base-to-derived member pointer conversions
  // in template parameter contexts, so the vbptr offset of data member pointers
  // is always zero.
  if (MSInheritanceAttr::hasVBPtrOffsetField(IM))
    mangleNumber(0);
  if (MSInheritanceAttr::hasVBTableOffsetField(IM))
    mangleNumber(VBTableOffset);
}

void
MicrosoftCXXNameMangler::mangleMemberFunctionPointer(const CXXRecordDecl *RD,
                                                     const CXXMethodDecl *MD) {
  // <member-function-pointer> ::= $1? <name>
  //                           ::= $H? <name> <number>
  //                           ::= $I? <name> <number> <number>
  //                           ::= $J? <name> <number> <number> <number>

  MSInheritanceAttr::Spelling IM = RD->getMSInheritanceModel();

  char Code = '\0';
  switch (IM) {
  case MSInheritanceAttr::Keyword_single_inheritance:      Code = '1'; break;
  case MSInheritanceAttr::Keyword_multiple_inheritance:    Code = 'H'; break;
  case MSInheritanceAttr::Keyword_virtual_inheritance:     Code = 'I'; break;
  case MSInheritanceAttr::Keyword_unspecified_inheritance: Code = 'J'; break;
  }

  // If non-virtual, mangle the name.  If virtual, mangle as a virtual memptr
  // thunk.
  uint64_t NVOffset = 0;
  uint64_t VBTableOffset = 0;
  uint64_t VBPtrOffset = 0;
  if (MD) {
    Out << '$' << Code << '?';
    if (MD->isVirtual()) {
      MicrosoftVTableContext *VTContext =
          cast<MicrosoftVTableContext>(getASTContext().getVTableContext());
      const MicrosoftVTableContext::MethodVFTableLocation &ML =
          VTContext->getMethodVFTableLocation(GlobalDecl(MD));
      mangleVirtualMemPtrThunk(MD, ML);
      NVOffset = ML.VFPtrOffset.getQuantity();
      VBTableOffset = ML.VBTableIndex * 4;
      if (ML.VBase) {
        const ASTRecordLayout &Layout = getASTContext().getASTRecordLayout(RD);
        VBPtrOffset = Layout.getVBPtrOffset().getQuantity();
      }
    } else {
      mangleName(MD);
      mangleFunctionEncoding(MD, /*ShouldMangle=*/true);
    }

    if (VBTableOffset == 0 &&
        IM == MSInheritanceAttr::Keyword_virtual_inheritance)
      NVOffset -= getASTContext().getOffsetOfBaseWithVBPtr(RD).getQuantity();
  } else {
    // Null single inheritance member functions are encoded as a simple nullptr.
    if (IM == MSInheritanceAttr::Keyword_single_inheritance) {
      Out << "$0A@";
      return;
    }
    if (IM == MSInheritanceAttr::Keyword_unspecified_inheritance)
      VBTableOffset = -1;
    Out << '$' << Code;
  }

  if (MSInheritanceAttr::hasNVOffsetField(/*IsMemberFunction=*/true, IM))
    mangleNumber(static_cast<uint32_t>(NVOffset));
  if (MSInheritanceAttr::hasVBPtrOffsetField(IM))
    mangleNumber(VBPtrOffset);
  if (MSInheritanceAttr::hasVBTableOffsetField(IM))
    mangleNumber(VBTableOffset);
}

void MicrosoftCXXNameMangler::mangleVirtualMemPtrThunk(
    const CXXMethodDecl *MD,
    const MicrosoftVTableContext::MethodVFTableLocation &ML) {
  // Get the vftable offset.
  CharUnits PointerWidth = getASTContext().toCharUnitsFromBits(
      getASTContext().getTargetInfo().getPointerWidth(0));
  uint64_t OffsetInVFTable = ML.Index * PointerWidth.getQuantity();

  Out << "?_9";
  mangleName(MD->getParent());
  Out << "$B";
  mangleNumber(OffsetInVFTable);
  Out << 'A';
  mangleCallingConvention(MD->getType()->getAs<FunctionProtoType>());
}

void MicrosoftCXXNameMangler::mangleName(const NamedDecl *ND) {
  // <name> ::= <unscoped-name> {[<named-scope>]+ | [<nested-name>]}? @

  // Always start with the unqualified name.
  mangleUnqualifiedName(ND);

  mangleNestedName(ND);

  // Terminate the whole name with an '@'.
  Out << '@';
}

void MicrosoftCXXNameMangler::mangleNumber(int64_t Number) {
  // <non-negative integer> ::= A@              # when Number == 0
  //                        ::= <decimal digit> # when 1 <= Number <= 10
  //                        ::= <hex digit>+ @  # when Number >= 10
  //
  // <number>               ::= [?] <non-negative integer>

  uint64_t Value = static_cast<uint64_t>(Number);
  if (Number < 0) {
    Value = -Value;
    Out << '?';
  }

  if (Value == 0)
    Out << "A@";
  else if (Value >= 1 && Value <= 10)
    Out << (Value - 1);
  else {
    // Numbers that are not encoded as decimal digits are represented as nibbles
    // in the range of ASCII characters 'A' to 'P'.
    // The number 0x123450 would be encoded as 'BCDEFA'
    char EncodedNumberBuffer[sizeof(uint64_t) * 2];
    MutableArrayRef<char> BufferRef(EncodedNumberBuffer);
    MutableArrayRef<char>::reverse_iterator I = BufferRef.rbegin();
    for (; Value != 0; Value >>= 4)
      *I++ = 'A' + (Value & 0xf);
    Out.write(I.base(), I - BufferRef.rbegin());
    Out << '@';
  }
}

static const TemplateDecl *
isTemplate(const NamedDecl *ND, const TemplateArgumentList *&TemplateArgs) {
  // Check if we have a function template.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    if (const TemplateDecl *TD = FD->getPrimaryTemplate()) {
      TemplateArgs = FD->getTemplateSpecializationArgs();
      return TD;
    }
  }

  // Check if we have a class template.
  if (const ClassTemplateSpecializationDecl *Spec =
          dyn_cast<ClassTemplateSpecializationDecl>(ND)) {
    TemplateArgs = &Spec->getTemplateArgs();
    return Spec->getSpecializedTemplate();
  }

  // Check if we have a variable template.
  if (const VarTemplateSpecializationDecl *Spec =
          dyn_cast<VarTemplateSpecializationDecl>(ND)) {
    TemplateArgs = &Spec->getTemplateArgs();
    return Spec->getSpecializedTemplate();
  }

  return nullptr;
}

void MicrosoftCXXNameMangler::mangleUnqualifiedName(const NamedDecl *ND,
                                                    DeclarationName Name) {
  //  <unqualified-name> ::= <operator-name>
  //                     ::= <ctor-dtor-name>
  //                     ::= <source-name>
  //                     ::= <template-name>

  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = nullptr;
  if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
    // Function templates aren't considered for name back referencing.  This
    // makes sense since function templates aren't likely to occur multiple
    // times in a symbol.
    if (!isa<ClassTemplateDecl>(TD)) {
      mangleTemplateInstantiationName(TD, *TemplateArgs);
      Out << '@';
      return;
    }

    // Here comes the tricky thing: if we need to mangle something like
    //   void foo(A::X<Y>, B::X<Y>),
    // the X<Y> part is aliased. However, if you need to mangle
    //   void foo(A::X<A::Y>, A::X<B::Y>),
    // the A::X<> part is not aliased.
    // That said, from the mangler's perspective we have a structure like this:
    //   namespace[s] -> type[ -> template-parameters]
    // but from the Clang perspective we have
    //   type [ -> template-parameters]
    //      \-> namespace[s]
    // What we do is we create a new mangler, mangle the same type (without
    // a namespace suffix) to a string using the extra mangler and then use 
    // the mangled type name as a key to check the mangling of different types
    // for aliasing.

    llvm::SmallString<64> TemplateMangling;
    llvm::raw_svector_ostream Stream(TemplateMangling);
    MicrosoftCXXNameMangler Extra(Context, Stream);
    Extra.mangleTemplateInstantiationName(TD, *TemplateArgs);

    mangleSourceName(TemplateMangling);
    return;
  }

  switch (Name.getNameKind()) {
    case DeclarationName::Identifier: {
      if (const IdentifierInfo *II = Name.getAsIdentifierInfo()) {
        mangleSourceName(II->getName());
        break;
      }

      // Otherwise, an anonymous entity.  We must have a declaration.
      assert(ND && "mangling empty name without declaration");

      if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
        if (NS->isAnonymousNamespace()) {
          Out << "?A@";
          break;
        }
      }

      if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
        // We must have an anonymous union or struct declaration.
        const CXXRecordDecl *RD = VD->getType()->getAsCXXRecordDecl();
        assert(RD && "expected variable decl to have a record type");
        // Anonymous types with no tag or typedef get the name of their
        // declarator mangled in.  If they have no declarator, number them with
        // a $S prefix.
        llvm::SmallString<64> Name("$S");
        // Get a unique id for the anonymous struct.
        Name += llvm::utostr(Context.getAnonymousStructId(RD) + 1);
        mangleSourceName(Name.str());
        break;
      }

      // We must have an anonymous struct.
      const TagDecl *TD = cast<TagDecl>(ND);
      if (const TypedefNameDecl *D = TD->getTypedefNameForAnonDecl()) {
        assert(TD->getDeclContext() == D->getDeclContext() &&
               "Typedef should not be in another decl context!");
        assert(D->getDeclName().getAsIdentifierInfo() &&
               "Typedef was not named!");
        mangleSourceName(D->getDeclName().getAsIdentifierInfo()->getName());
        break;
      }

      if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(TD)) {
        if (Record->isLambda()) {
          llvm::SmallString<10> Name("<lambda_");
          unsigned LambdaId;
          if (Record->getLambdaManglingNumber())
            LambdaId = Record->getLambdaManglingNumber();
          else
            LambdaId = Context.getLambdaId(Record);

          Name += llvm::utostr(LambdaId);
          Name += ">";

          mangleSourceName(Name);
          break;
        }
      }

      llvm::SmallString<64> Name("<unnamed-type-");
      if (DeclaratorDecl *DD =
              Context.getASTContext().getDeclaratorForUnnamedTagDecl(TD)) {
        // Anonymous types without a name for linkage purposes have their
        // declarator mangled in if they have one.
        Name += DD->getName();
      } else if (TypedefNameDecl *TND =
                     Context.getASTContext().getTypedefNameForUnnamedTagDecl(
                         TD)) {
        // Anonymous types without a name for linkage purposes have their
        // associate typedef mangled in if they have one.
        Name += TND->getName();
      } else {
        // Otherwise, number the types using a $S prefix.
        Name += "$S";
        Name += llvm::utostr(Context.getAnonymousStructId(TD) + 1);
      }
      Name += ">";
      mangleSourceName(Name.str());
      break;
    }

    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      llvm_unreachable("Can't mangle Objective-C selector names here!");

    case DeclarationName::CXXConstructorName:
      if (Structor == getStructor(ND)) {
        if (StructorType == Ctor_CopyingClosure) {
          Out << "?_O";
          return;
        }
        if (StructorType == Ctor_DefaultClosure) {
          Out << "?_F";
          return;
        }
      }
      Out << "?0";
      return;

    case DeclarationName::CXXDestructorName:
      if (ND == Structor)
        // If the named decl is the C++ destructor we're mangling,
        // use the type we were given.
        mangleCXXDtorType(static_cast<CXXDtorType>(StructorType));
      else
        // Otherwise, use the base destructor name. This is relevant if a
        // class with a destructor is declared within a destructor.
        mangleCXXDtorType(Dtor_Base);
      break;

    case DeclarationName::CXXConversionFunctionName:
      // <operator-name> ::= ?B # (cast)
      // The target type is encoded as the return type.
      Out << "?B";
      break;

    case DeclarationName::CXXOperatorName:
      mangleOperatorName(Name.getCXXOverloadedOperator(), ND->getLocation());
      break;

    case DeclarationName::CXXLiteralOperatorName: {
      Out << "?__K";
      mangleSourceName(Name.getCXXLiteralIdentifier()->getName());
      break;
    }

    case DeclarationName::CXXUsingDirective:
      llvm_unreachable("Can't mangle a using directive name!");
  }
}

void MicrosoftCXXNameMangler::mangleNestedName(const NamedDecl *ND) {
  // <postfix> ::= <unqualified-name> [<postfix>]
  //           ::= <substitution> [<postfix>]
  const DeclContext *DC = getEffectiveDeclContext(ND);

  while (!DC->isTranslationUnit()) {
    if (isa<TagDecl>(ND) || isa<VarDecl>(ND)) {
      unsigned Disc;
      if (Context.getNextDiscriminator(ND, Disc)) {
        Out << '?';
        mangleNumber(Disc);
        Out << '?';
      }
    }

    if (const BlockDecl *BD = dyn_cast<BlockDecl>(DC)) {
      DiagnosticsEngine &Diags = Context.getDiags();
      unsigned DiagID =
          Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                "cannot mangle a local inside this block yet");
      Diags.Report(BD->getLocation(), DiagID);

      // FIXME: This is completely, utterly, wrong; see ItaniumMangle
      // for how this should be done.
      Out << "__block_invoke" << Context.getBlockId(BD, false);
      Out << '@';
      continue;
    } else if (const ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(DC)) {
      mangleObjCMethodName(Method);
    } else if (isa<NamedDecl>(DC)) {
      ND = cast<NamedDecl>(DC);
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
        mangle(FD, "?");
        break;
      } else
        mangleUnqualifiedName(ND);
    }
    DC = DC->getParent();
  }
}

void MicrosoftCXXNameMangler::mangleCXXDtorType(CXXDtorType T) {
  // Microsoft uses the names on the case labels for these dtor variants.  Clang
  // uses the Itanium terminology internally.  Everything in this ABI delegates
  // towards the base dtor.
  switch (T) {
  // <operator-name> ::= ?1  # destructor
  case Dtor_Base: Out << "?1"; return;
  // <operator-name> ::= ?_D # vbase destructor
  case Dtor_Complete: Out << "?_D"; return;
  // <operator-name> ::= ?_G # scalar deleting destructor
  case Dtor_Deleting: Out << "?_G"; return;
  // <operator-name> ::= ?_E # vector deleting destructor
  // FIXME: Add a vector deleting dtor type.  It goes in the vtable, so we need
  // it.
  case Dtor_Comdat:
    llvm_unreachable("not expecting a COMDAT");
  }
  llvm_unreachable("Unsupported dtor type?");
}

void MicrosoftCXXNameMangler::mangleOperatorName(OverloadedOperatorKind OO,
                                                 SourceLocation Loc) {
  switch (OO) {
  //                     ?0 # constructor
  //                     ?1 # destructor
  // <operator-name> ::= ?2 # new
  case OO_New: Out << "?2"; break;
  // <operator-name> ::= ?3 # delete
  case OO_Delete: Out << "?3"; break;
  // <operator-name> ::= ?4 # =
  case OO_Equal: Out << "?4"; break;
  // <operator-name> ::= ?5 # >>
  case OO_GreaterGreater: Out << "?5"; break;
  // <operator-name> ::= ?6 # <<
  case OO_LessLess: Out << "?6"; break;
  // <operator-name> ::= ?7 # !
  case OO_Exclaim: Out << "?7"; break;
  // <operator-name> ::= ?8 # ==
  case OO_EqualEqual: Out << "?8"; break;
  // <operator-name> ::= ?9 # !=
  case OO_ExclaimEqual: Out << "?9"; break;
  // <operator-name> ::= ?A # []
  case OO_Subscript: Out << "?A"; break;
  //                     ?B # conversion
  // <operator-name> ::= ?C # ->
  case OO_Arrow: Out << "?C"; break;
  // <operator-name> ::= ?D # *
  case OO_Star: Out << "?D"; break;
  // <operator-name> ::= ?E # ++
  case OO_PlusPlus: Out << "?E"; break;
  // <operator-name> ::= ?F # --
  case OO_MinusMinus: Out << "?F"; break;
  // <operator-name> ::= ?G # -
  case OO_Minus: Out << "?G"; break;
  // <operator-name> ::= ?H # +
  case OO_Plus: Out << "?H"; break;
  // <operator-name> ::= ?I # &
  case OO_Amp: Out << "?I"; break;
  // <operator-name> ::= ?J # ->*
  case OO_ArrowStar: Out << "?J"; break;
  // <operator-name> ::= ?K # /
  case OO_Slash: Out << "?K"; break;
  // <operator-name> ::= ?L # %
  case OO_Percent: Out << "?L"; break;
  // <operator-name> ::= ?M # <
  case OO_Less: Out << "?M"; break;
  // <operator-name> ::= ?N # <=
  case OO_LessEqual: Out << "?N"; break;
  // <operator-name> ::= ?O # >
  case OO_Greater: Out << "?O"; break;
  // <operator-name> ::= ?P # >=
  case OO_GreaterEqual: Out << "?P"; break;
  // <operator-name> ::= ?Q # ,
  case OO_Comma: Out << "?Q"; break;
  // <operator-name> ::= ?R # ()
  case OO_Call: Out << "?R"; break;
  // <operator-name> ::= ?S # ~
  case OO_Tilde: Out << "?S"; break;
  // <operator-name> ::= ?T # ^
  case OO_Caret: Out << "?T"; break;
  // <operator-name> ::= ?U # |
  case OO_Pipe: Out << "?U"; break;
  // <operator-name> ::= ?V # &&
  case OO_AmpAmp: Out << "?V"; break;
  // <operator-name> ::= ?W # ||
  case OO_PipePipe: Out << "?W"; break;
  // <operator-name> ::= ?X # *=
  case OO_StarEqual: Out << "?X"; break;
  // <operator-name> ::= ?Y # +=
  case OO_PlusEqual: Out << "?Y"; break;
  // <operator-name> ::= ?Z # -=
  case OO_MinusEqual: Out << "?Z"; break;
  // <operator-name> ::= ?_0 # /=
  case OO_SlashEqual: Out << "?_0"; break;
  // <operator-name> ::= ?_1 # %=
  case OO_PercentEqual: Out << "?_1"; break;
  // <operator-name> ::= ?_2 # >>=
  case OO_GreaterGreaterEqual: Out << "?_2"; break;
  // <operator-name> ::= ?_3 # <<=
  case OO_LessLessEqual: Out << "?_3"; break;
  // <operator-name> ::= ?_4 # &=
  case OO_AmpEqual: Out << "?_4"; break;
  // <operator-name> ::= ?_5 # |=
  case OO_PipeEqual: Out << "?_5"; break;
  // <operator-name> ::= ?_6 # ^=
  case OO_CaretEqual: Out << "?_6"; break;
  //                     ?_7 # vftable
  //                     ?_8 # vbtable
  //                     ?_9 # vcall
  //                     ?_A # typeof
  //                     ?_B # local static guard
  //                     ?_C # string
  //                     ?_D # vbase destructor
  //                     ?_E # vector deleting destructor
  //                     ?_F # default constructor closure
  //                     ?_G # scalar deleting destructor
  //                     ?_H # vector constructor iterator
  //                     ?_I # vector destructor iterator
  //                     ?_J # vector vbase constructor iterator
  //                     ?_K # virtual displacement map
  //                     ?_L # eh vector constructor iterator
  //                     ?_M # eh vector destructor iterator
  //                     ?_N # eh vector vbase constructor iterator
  //                     ?_O # copy constructor closure
  //                     ?_P<name> # udt returning <name>
  //                     ?_Q # <unknown>
  //                     ?_R0 # RTTI Type Descriptor
  //                     ?_R1 # RTTI Base Class Descriptor at (a,b,c,d)
  //                     ?_R2 # RTTI Base Class Array
  //                     ?_R3 # RTTI Class Hierarchy Descriptor
  //                     ?_R4 # RTTI Complete Object Locator
  //                     ?_S # local vftable
  //                     ?_T # local vftable constructor closure
  // <operator-name> ::= ?_U # new[]
  case OO_Array_New: Out << "?_U"; break;
  // <operator-name> ::= ?_V # delete[]
  case OO_Array_Delete: Out << "?_V"; break;

  case OO_Conditional: {
    DiagnosticsEngine &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
      "cannot mangle this conditional operator yet");
    Diags.Report(Loc, DiagID);
    break;
  }

  case OO_Coawait: {
    DiagnosticsEngine &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
      "cannot mangle this operator co_await yet");
    Diags.Report(Loc, DiagID);
    break;
  }

  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Not an overloaded operator");
  }
}

void MicrosoftCXXNameMangler::mangleSourceName(StringRef Name) {
  // <source name> ::= <identifier> @
  BackRefVec::iterator Found =
      std::find(NameBackReferences.begin(), NameBackReferences.end(), Name);
  if (Found == NameBackReferences.end()) {
    if (NameBackReferences.size() < 10)
      NameBackReferences.push_back(Name);
    Out << Name << '@';
  } else {
    Out << (Found - NameBackReferences.begin());
  }
}

void MicrosoftCXXNameMangler::mangleObjCMethodName(const ObjCMethodDecl *MD) {
  Context.mangleObjCMethodName(MD, Out);
}

void MicrosoftCXXNameMangler::mangleTemplateInstantiationName(
    const TemplateDecl *TD, const TemplateArgumentList &TemplateArgs) {
  // <template-name> ::= <unscoped-template-name> <template-args>
  //                 ::= <substitution>
  // Always start with the unqualified name.

  // Templates have their own context for back references.
  ArgBackRefMap OuterArgsContext;
  BackRefVec OuterTemplateContext;
  NameBackReferences.swap(OuterTemplateContext);
  TypeBackReferences.swap(OuterArgsContext);

  mangleUnscopedTemplateName(TD);
  mangleTemplateArgs(TD, TemplateArgs);

  // Restore the previous back reference contexts.
  NameBackReferences.swap(OuterTemplateContext);
  TypeBackReferences.swap(OuterArgsContext);
}

void
MicrosoftCXXNameMangler::mangleUnscopedTemplateName(const TemplateDecl *TD) {
  // <unscoped-template-name> ::= ?$ <unqualified-name>
  Out << "?$";
  mangleUnqualifiedName(TD);
}

void MicrosoftCXXNameMangler::mangleIntegerLiteral(const llvm::APSInt &Value,
                                                   bool IsBoolean) {
  // <integer-literal> ::= $0 <number>
  Out << "$0";
  // Make sure booleans are encoded as 0/1.
  if (IsBoolean && Value.getBoolValue())
    mangleNumber(1);
  else if (Value.isSigned())
    mangleNumber(Value.getSExtValue());
  else
    mangleNumber(Value.getZExtValue());
}

void MicrosoftCXXNameMangler::mangleExpression(const Expr *E) {
  // See if this is a constant expression.
  llvm::APSInt Value;
  if (E->isIntegerConstantExpr(Value, Context.getASTContext())) {
    mangleIntegerLiteral(Value, E->getType()->isBooleanType());
    return;
  }

  // Look through no-op casts like template parameter substitutions.
  E = E->IgnoreParenNoopCasts(Context.getASTContext());

  const CXXUuidofExpr *UE = nullptr;
  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_AddrOf)
      UE = dyn_cast<CXXUuidofExpr>(UO->getSubExpr());
  } else
    UE = dyn_cast<CXXUuidofExpr>(E);

  if (UE) {
    // This CXXUuidofExpr is mangled as-if it were actually a VarDecl from
    // const __s_GUID _GUID_{lower case UUID with underscores}
    StringRef Uuid = UE->getUuidAsStringRef(Context.getASTContext());
    std::string Name = "_GUID_" + Uuid.lower();
    std::replace(Name.begin(), Name.end(), '-', '_');

    // If we had to peek through an address-of operator, treat this like we are
    // dealing with a pointer type.  Otherwise, treat it like a const reference.
    //
    // N.B. This matches up with the handling of TemplateArgument::Declaration
    // in mangleTemplateArg
    if (UE == E)
      Out << "$E?";
    else
      Out << "$1?";
    Out << Name << "@@3U__s_GUID@@B";
    return;
  }

  // As bad as this diagnostic is, it's better than crashing.
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(
      DiagnosticsEngine::Error, "cannot yet mangle expression type %0");
  Diags.Report(E->getExprLoc(), DiagID) << E->getStmtClassName()
                                        << E->getSourceRange();
}

void MicrosoftCXXNameMangler::mangleTemplateArgs(
    const TemplateDecl *TD, const TemplateArgumentList &TemplateArgs) {
  // <template-args> ::= <template-arg>+
  const TemplateParameterList *TPL = TD->getTemplateParameters();
  assert(TPL->size() == TemplateArgs.size() &&
         "size mismatch between args and parms!");

  unsigned Idx = 0;
  for (const TemplateArgument &TA : TemplateArgs.asArray())
    mangleTemplateArg(TD, TA, TPL->getParam(Idx++));
}

void MicrosoftCXXNameMangler::mangleTemplateArg(const TemplateDecl *TD,
                                                const TemplateArgument &TA,
                                                const NamedDecl *Parm) {
  // <template-arg> ::= <type>
  //                ::= <integer-literal>
  //                ::= <member-data-pointer>
  //                ::= <member-function-pointer>
  //                ::= $E? <name> <type-encoding>
  //                ::= $1? <name> <type-encoding>
  //                ::= $0A@
  //                ::= <template-args>

  switch (TA.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Can't mangle null template arguments!");
  case TemplateArgument::TemplateExpansion:
    llvm_unreachable("Can't mangle template expansion arguments!");
  case TemplateArgument::Type: {
    QualType T = TA.getAsType();
    mangleType(T, SourceRange(), QMM_Escape);
    break;
  }
  case TemplateArgument::Declaration: {
    const NamedDecl *ND = cast<NamedDecl>(TA.getAsDecl());
    if (isa<FieldDecl>(ND) || isa<IndirectFieldDecl>(ND)) {
      mangleMemberDataPointer(
          cast<CXXRecordDecl>(ND->getDeclContext())->getMostRecentDecl(),
          cast<ValueDecl>(ND));
    } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
      const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
      if (MD && MD->isInstance()) {
        mangleMemberFunctionPointer(MD->getParent()->getMostRecentDecl(), MD);
      } else {
        Out << "$1?";
        mangleName(FD);
        mangleFunctionEncoding(FD, /*ShouldMangle=*/true);
      }
    } else {
      mangle(ND, TA.getParamTypeForDecl()->isReferenceType() ? "$E?" : "$1?");
    }
    break;
  }
  case TemplateArgument::Integral:
    mangleIntegerLiteral(TA.getAsIntegral(),
                         TA.getIntegralType()->isBooleanType());
    break;
  case TemplateArgument::NullPtr: {
    QualType T = TA.getNullPtrType();
    if (const MemberPointerType *MPT = T->getAs<MemberPointerType>()) {
      const CXXRecordDecl *RD = MPT->getMostRecentCXXRecordDecl();
      if (MPT->isMemberFunctionPointerType() &&
          !isa<FunctionTemplateDecl>(TD)) {
        mangleMemberFunctionPointer(RD, nullptr);
        return;
      }
      if (MPT->isMemberDataPointer()) {
        if (!isa<FunctionTemplateDecl>(TD)) {
          mangleMemberDataPointer(RD, nullptr);
          return;
        }
        // nullptr data pointers are always represented with a single field
        // which is initialized with either 0 or -1.  Why -1?  Well, we need to
        // distinguish the case where the data member is at offset zero in the
        // record.
        // However, we are free to use 0 *if* we would use multiple fields for
        // non-nullptr member pointers.
        if (!RD->nullFieldOffsetIsZero()) {
          mangleIntegerLiteral(llvm::APSInt::get(-1), /*IsBoolean=*/false);
          return;
        }
      }
    }
    mangleIntegerLiteral(llvm::APSInt::getUnsigned(0), /*IsBoolean=*/false);
    break;
  }
  case TemplateArgument::Expression:
    mangleExpression(TA.getAsExpr());
    break;
  case TemplateArgument::Pack: {
    ArrayRef<TemplateArgument> TemplateArgs = TA.getPackAsArray();
    if (TemplateArgs.empty()) {
      if (isa<TemplateTypeParmDecl>(Parm) ||
          isa<TemplateTemplateParmDecl>(Parm))
        // MSVC 2015 changed the mangling for empty expanded template packs,
        // use the old mangling for link compatibility for old versions.
        Out << (Context.getASTContext().getLangOpts().isCompatibleWithMSVC(
                    LangOptions::MSVC2015)
                    ? "$$V"
                    : "$$$V");
      else if (isa<NonTypeTemplateParmDecl>(Parm))
        Out << "$S";
      else
        llvm_unreachable("unexpected template parameter decl!");
    } else {
      for (const TemplateArgument &PA : TemplateArgs)
        mangleTemplateArg(TD, PA, Parm);
    }
    break;
  }
  case TemplateArgument::Template: {
    const NamedDecl *ND =
        TA.getAsTemplate().getAsTemplateDecl()->getTemplatedDecl();
    if (const auto *TD = dyn_cast<TagDecl>(ND)) {
      mangleType(TD);
    } else if (isa<TypeAliasDecl>(ND)) {
      Out << "$$Y";
      mangleName(ND);
    } else {
      llvm_unreachable("unexpected template template NamedDecl!");
    }
    break;
  }
  }
}

void MicrosoftCXXNameMangler::mangleQualifiers(Qualifiers Quals,
                                               bool IsMember) {
  // <cvr-qualifiers> ::= [E] [F] [I] <base-cvr-qualifiers>
  // 'E' means __ptr64 (32-bit only); 'F' means __unaligned (32/64-bit only);
  // 'I' means __restrict (32/64-bit).
  // Note that the MSVC __restrict keyword isn't the same as the C99 restrict
  // keyword!
  // <base-cvr-qualifiers> ::= A  # near
  //                       ::= B  # near const
  //                       ::= C  # near volatile
  //                       ::= D  # near const volatile
  //                       ::= E  # far (16-bit)
  //                       ::= F  # far const (16-bit)
  //                       ::= G  # far volatile (16-bit)
  //                       ::= H  # far const volatile (16-bit)
  //                       ::= I  # huge (16-bit)
  //                       ::= J  # huge const (16-bit)
  //                       ::= K  # huge volatile (16-bit)
  //                       ::= L  # huge const volatile (16-bit)
  //                       ::= M <basis> # based
  //                       ::= N <basis> # based const
  //                       ::= O <basis> # based volatile
  //                       ::= P <basis> # based const volatile
  //                       ::= Q  # near member
  //                       ::= R  # near const member
  //                       ::= S  # near volatile member
  //                       ::= T  # near const volatile member
  //                       ::= U  # far member (16-bit)
  //                       ::= V  # far const member (16-bit)
  //                       ::= W  # far volatile member (16-bit)
  //                       ::= X  # far const volatile member (16-bit)
  //                       ::= Y  # huge member (16-bit)
  //                       ::= Z  # huge const member (16-bit)
  //                       ::= 0  # huge volatile member (16-bit)
  //                       ::= 1  # huge const volatile member (16-bit)
  //                       ::= 2 <basis> # based member
  //                       ::= 3 <basis> # based const member
  //                       ::= 4 <basis> # based volatile member
  //                       ::= 5 <basis> # based const volatile member
  //                       ::= 6  # near function (pointers only)
  //                       ::= 7  # far function (pointers only)
  //                       ::= 8  # near method (pointers only)
  //                       ::= 9  # far method (pointers only)
  //                       ::= _A <basis> # based function (pointers only)
  //                       ::= _B <basis> # based function (far?) (pointers only)
  //                       ::= _C <basis> # based method (pointers only)
  //                       ::= _D <basis> # based method (far?) (pointers only)
  //                       ::= _E # block (Clang)
  // <basis> ::= 0 # __based(void)
  //         ::= 1 # __based(segment)?
  //         ::= 2 <name> # __based(name)
  //         ::= 3 # ?
  //         ::= 4 # ?
  //         ::= 5 # not really based
  bool HasConst = Quals.hasConst(),
       HasVolatile = Quals.hasVolatile();

  if (!IsMember) {
    if (HasConst && HasVolatile) {
      Out << 'D';
    } else if (HasVolatile) {
      Out << 'C';
    } else if (HasConst) {
      Out << 'B';
    } else {
      Out << 'A';
    }
  } else {
    if (HasConst && HasVolatile) {
      Out << 'T';
    } else if (HasVolatile) {
      Out << 'S';
    } else if (HasConst) {
      Out << 'R';
    } else {
      Out << 'Q';
    }
  }

  // FIXME: For now, just drop all extension qualifiers on the floor.
}

void
MicrosoftCXXNameMangler::mangleRefQualifier(RefQualifierKind RefQualifier) {
  // <ref-qualifier> ::= G                # lvalue reference
  //                 ::= H                # rvalue-reference
  switch (RefQualifier) {
  case RQ_None:
    break;

  case RQ_LValue:
    Out << 'G';
    break;

  case RQ_RValue:
    Out << 'H';
    break;
  }
}

void MicrosoftCXXNameMangler::manglePointerExtQualifiers(Qualifiers Quals,
                                                         QualType PointeeType) {
  bool HasRestrict = Quals.hasRestrict();
  if (PointersAre64Bit &&
      (PointeeType.isNull() || !PointeeType->isFunctionType()))
    Out << 'E';

  if (HasRestrict)
    Out << 'I';
}

void MicrosoftCXXNameMangler::manglePointerCVQualifiers(Qualifiers Quals) {
  // <pointer-cv-qualifiers> ::= P  # no qualifiers
  //                         ::= Q  # const
  //                         ::= R  # volatile
  //                         ::= S  # const volatile
  bool HasConst = Quals.hasConst(),
       HasVolatile = Quals.hasVolatile();

  if (HasConst && HasVolatile) {
    Out << 'S';
  } else if (HasVolatile) {
    Out << 'R';
  } else if (HasConst) {
    Out << 'Q';
  } else {
    Out << 'P';
  }
}

void MicrosoftCXXNameMangler::mangleArgumentType(QualType T,
                                                 SourceRange Range) {
  // MSVC will backreference two canonically equivalent types that have slightly
  // different manglings when mangled alone.

  // Decayed types do not match up with non-decayed versions of the same type.
  //
  // e.g.
  // void (*x)(void) will not form a backreference with void x(void)
  void *TypePtr;
  if (const auto *DT = T->getAs<DecayedType>()) {
    QualType OriginalType = DT->getOriginalType();
    // All decayed ArrayTypes should be treated identically; as-if they were
    // a decayed IncompleteArrayType.
    if (const auto *AT = getASTContext().getAsArrayType(OriginalType))
      OriginalType = getASTContext().getIncompleteArrayType(
          AT->getElementType(), AT->getSizeModifier(),
          AT->getIndexTypeCVRQualifiers());

    TypePtr = OriginalType.getCanonicalType().getAsOpaquePtr();
    // If the original parameter was textually written as an array,
    // instead treat the decayed parameter like it's const.
    //
    // e.g.
    // int [] -> int * const
    if (OriginalType->isArrayType())
      T = T.withConst();
  } else {
    TypePtr = T.getCanonicalType().getAsOpaquePtr();
  }

  ArgBackRefMap::iterator Found = TypeBackReferences.find(TypePtr);

  if (Found == TypeBackReferences.end()) {
    size_t OutSizeBefore = Out.tell();

    mangleType(T, Range, QMM_Drop);

    // See if it's worth creating a back reference.
    // Only types longer than 1 character are considered
    // and only 10 back references slots are available:
    bool LongerThanOneChar = (Out.tell() - OutSizeBefore > 1);
    if (LongerThanOneChar && TypeBackReferences.size() < 10) {
      size_t Size = TypeBackReferences.size();
      TypeBackReferences[TypePtr] = Size;
    }
  } else {
    Out << Found->second;
  }
}

void MicrosoftCXXNameMangler::mangleType(QualType T, SourceRange Range,
                                         QualifierMangleMode QMM) {
  // Don't use the canonical types.  MSVC includes things like 'const' on
  // pointer arguments to function pointers that canonicalization strips away.
  T = T.getDesugaredType(getASTContext());
  Qualifiers Quals = T.getLocalQualifiers();
  if (const ArrayType *AT = getASTContext().getAsArrayType(T)) {
    // If there were any Quals, getAsArrayType() pushed them onto the array
    // element type.
    if (QMM == QMM_Mangle)
      Out << 'A';
    else if (QMM == QMM_Escape || QMM == QMM_Result)
      Out << "$$B";
    mangleArrayType(AT);
    return;
  }

  bool IsPointer = T->isAnyPointerType() || T->isMemberPointerType() ||
                   T->isReferenceType() || T->isBlockPointerType();

  switch (QMM) {
  case QMM_Drop:
    break;
  case QMM_Mangle:
    if (const FunctionType *FT = dyn_cast<FunctionType>(T)) {
      Out << '6';
      mangleFunctionType(FT);
      return;
    }
    mangleQualifiers(Quals, false);
    break;
  case QMM_Escape:
    if (!IsPointer && Quals) {
      Out << "$$C";
      mangleQualifiers(Quals, false);
    }
    break;
  case QMM_Result:
    if ((!IsPointer && Quals) || isa<TagType>(T)) {
      Out << '?';
      mangleQualifiers(Quals, false);
    }
    break;
  }

  const Type *ty = T.getTypePtr();

  switch (ty->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT) \
  case Type::CLASS: \
    llvm_unreachable("can't mangle non-canonical type " #CLASS "Type"); \
    return;
#define TYPE(CLASS, PARENT) \
  case Type::CLASS: \
    mangleType(cast<CLASS##Type>(ty), Quals, Range); \
    break;
#include "clang/AST/TypeNodes.def"
#undef ABSTRACT_TYPE
#undef NON_CANONICAL_TYPE
#undef TYPE
  }
}

void MicrosoftCXXNameMangler::mangleType(const BuiltinType *T, Qualifiers,
                                         SourceRange Range) {
  //  <type>         ::= <builtin-type>
  //  <builtin-type> ::= X  # void
  //                 ::= C  # signed char
  //                 ::= D  # char
  //                 ::= E  # unsigned char
  //                 ::= F  # short
  //                 ::= G  # unsigned short (or wchar_t if it's not a builtin)
  //                 ::= H  # int
  //                 ::= I  # unsigned int
  //                 ::= J  # long
  //                 ::= K  # unsigned long
  //                     L  # <none>
  //                 ::= M  # float
  //                 ::= N  # double
  //                 ::= O  # long double (__float80 is mangled differently)
  //                 ::= _J # long long, __int64
  //                 ::= _K # unsigned long long, __int64
  //                 ::= _L # __int128
  //                 ::= _M # unsigned __int128
  //                 ::= _N # bool
  //                     _O # <array in parameter>
  //                 ::= _T # __float80 (Intel)
  //                 ::= _W # wchar_t
  //                 ::= _Z # __float80 (Digital Mars)
  switch (T->getKind()) {
  case BuiltinType::Void:
    Out << 'X';
    break;
  case BuiltinType::SChar:
    Out << 'C';
    break;
  case BuiltinType::Char_U:
  case BuiltinType::Char_S:
    Out << 'D';
    break;
  case BuiltinType::UChar:
    Out << 'E';
    break;
  case BuiltinType::Short:
    Out << 'F';
    break;
  case BuiltinType::UShort:
    Out << 'G';
    break;
  case BuiltinType::Int:
    Out << 'H';
    break;
  case BuiltinType::UInt:
    Out << 'I';
    break;
  case BuiltinType::Long:
    Out << 'J';
    break;
  case BuiltinType::ULong:
    Out << 'K';
    break;
  case BuiltinType::Float:
    Out << 'M';
    break;
  case BuiltinType::Double:
    Out << 'N';
    break;
  // TODO: Determine size and mangle accordingly
  case BuiltinType::LongDouble:
    Out << 'O';
    break;
  case BuiltinType::LongLong:
    Out << "_J";
    break;
  case BuiltinType::ULongLong:
    Out << "_K";
    break;
  case BuiltinType::Int128:
    Out << "_L";
    break;
  case BuiltinType::UInt128:
    Out << "_M";
    break;
  case BuiltinType::Bool:
    Out << "_N";
    break;
  case BuiltinType::Char16:
    Out << "_S";
    break;
  case BuiltinType::Char32:
    Out << "_U";
    break;
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
    Out << "_W";
    break;

#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) \
  case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
  case BuiltinType::Dependent:
    llvm_unreachable("placeholder types shouldn't get to name mangling");

  case BuiltinType::ObjCId:
    Out << "PAUobjc_object@@";
    break;
  case BuiltinType::ObjCClass:
    Out << "PAUobjc_class@@";
    break;
  case BuiltinType::ObjCSel:
    Out << "PAUobjc_selector@@";
    break;

  case BuiltinType::OCLImage1d:
    Out << "PAUocl_image1d@@";
    break;
  case BuiltinType::OCLImage1dArray:
    Out << "PAUocl_image1darray@@";
    break;
  case BuiltinType::OCLImage1dBuffer:
    Out << "PAUocl_image1dbuffer@@";
    break;
  case BuiltinType::OCLImage2d:
    Out << "PAUocl_image2d@@";
    break;
  case BuiltinType::OCLImage2dArray:
    Out << "PAUocl_image2darray@@";
    break;
  case BuiltinType::OCLImage2dDepth:
    Out << "PAUocl_image2ddepth@@";
    break;
  case BuiltinType::OCLImage2dArrayDepth:
    Out << "PAUocl_image2darraydepth@@";
    break;
  case BuiltinType::OCLImage2dMSAA:
    Out << "PAUocl_image2dmsaa@@";
    break;
  case BuiltinType::OCLImage2dArrayMSAA:
    Out << "PAUocl_image2darraymsaa@@";
    break;
  case BuiltinType::OCLImage2dMSAADepth:
    Out << "PAUocl_image2dmsaadepth@@";
    break;
  case BuiltinType::OCLImage2dArrayMSAADepth:
    Out << "PAUocl_image2darraymsaadepth@@";
    break;
  case BuiltinType::OCLImage3d:
    Out << "PAUocl_image3d@@";
    break;
  case BuiltinType::OCLSampler:
    Out << "PAUocl_sampler@@";
    break;
  case BuiltinType::OCLEvent:
    Out << "PAUocl_event@@";
    break;
  case BuiltinType::OCLClkEvent:
    Out << "PAUocl_clkevent@@";
    break;
  case BuiltinType::OCLQueue:
    Out << "PAUocl_queue@@";
    break;
  case BuiltinType::OCLNDRange:
    Out << "PAUocl_ndrange@@";
    break;
  case BuiltinType::OCLReserveID:
    Out << "PAUocl_reserveid@@";
    break;

  case BuiltinType::NullPtr:
    Out << "$$T";
    break;

  case BuiltinType::Half: {
    DiagnosticsEngine &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(
        DiagnosticsEngine::Error, "cannot mangle this built-in %0 type yet");
    Diags.Report(Range.getBegin(), DiagID)
        << T->getName(Context.getASTContext().getPrintingPolicy()) << Range;
    break;
  }
  }
}

// <type>          ::= <function-type>
void MicrosoftCXXNameMangler::mangleType(const FunctionProtoType *T, Qualifiers,
                                         SourceRange) {
  // Structors only appear in decls, so at this point we know it's not a
  // structor type.
  // FIXME: This may not be lambda-friendly.
  if (T->getTypeQuals() || T->getRefQualifier() != RQ_None) {
    Out << "$$A8@@";
    mangleFunctionType(T, /*D=*/nullptr, /*ForceThisQuals=*/true);
  } else {
    Out << "$$A6";
    mangleFunctionType(T);
  }
}
void MicrosoftCXXNameMangler::mangleType(const FunctionNoProtoType *T,
                                         Qualifiers, SourceRange) {
  Out << "$$A6";
  mangleFunctionType(T);
}

void MicrosoftCXXNameMangler::mangleFunctionType(const FunctionType *T,
                                                 const FunctionDecl *D,
                                                 bool ForceThisQuals) {
  // <function-type> ::= <this-cvr-qualifiers> <calling-convention>
  //                     <return-type> <argument-list> <throw-spec>
  const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(T);

  SourceRange Range;
  if (D) Range = D->getSourceRange();

  bool IsStructor = false, HasThisQuals = ForceThisQuals, IsCtorClosure = false;
  CallingConv CC = T->getCallConv();
  if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(D)) {
    if (MD->isInstance())
      HasThisQuals = true;
    if (isa<CXXDestructorDecl>(MD)) {
      IsStructor = true;
    } else if (isa<CXXConstructorDecl>(MD)) {
      IsStructor = true;
      IsCtorClosure = (StructorType == Ctor_CopyingClosure ||
                       StructorType == Ctor_DefaultClosure) &&
                      getStructor(MD) == Structor;
      if (IsCtorClosure)
        CC = getASTContext().getDefaultCallingConvention(
            /*IsVariadic=*/false, /*IsCXXMethod=*/true);
    }
  }

  // If this is a C++ instance method, mangle the CVR qualifiers for the
  // this pointer.
  if (HasThisQuals) {
    Qualifiers Quals = Qualifiers::fromCVRMask(Proto->getTypeQuals());
    manglePointerExtQualifiers(Quals, /*PointeeType=*/QualType());
    mangleRefQualifier(Proto->getRefQualifier());
    mangleQualifiers(Quals, /*IsMember=*/false);
  }

  mangleCallingConvention(CC);

  // <return-type> ::= <type>
  //               ::= @ # structors (they have no declared return type)
  if (IsStructor) {
    if (isa<CXXDestructorDecl>(D) && D == Structor &&
        StructorType == Dtor_Deleting) {
      // The scalar deleting destructor takes an extra int argument.
      // However, the FunctionType generated has 0 arguments.
      // FIXME: This is a temporary hack.
      // Maybe should fix the FunctionType creation instead?
      Out << (PointersAre64Bit ? "PEAXI@Z" : "PAXI@Z");
      return;
    }
    if (IsCtorClosure) {
      // Default constructor closure and copy constructor closure both return
      // void.
      Out << 'X';

      if (StructorType == Ctor_DefaultClosure) {
        // Default constructor closure always has no arguments.
        Out << 'X';
      } else if (StructorType == Ctor_CopyingClosure) {
        // Copy constructor closure always takes an unqualified reference.
        mangleArgumentType(getASTContext().getLValueReferenceType(
                               Proto->getParamType(0)
                                   ->getAs<LValueReferenceType>()
                                   ->getPointeeType(),
                               /*SpelledAsLValue=*/true),
                           Range);
        Out << '@';
      } else {
        llvm_unreachable("unexpected constructor closure!");
      }
      Out << 'Z';
      return;
    }
    Out << '@';
  } else {
    QualType ResultType = T->getReturnType();
    if (const auto *AT =
            dyn_cast_or_null<AutoType>(ResultType->getContainedAutoType())) {
      Out << '?';
      mangleQualifiers(ResultType.getLocalQualifiers(), /*IsMember=*/false);
      Out << '?';
      assert(AT->getKeyword() != AutoTypeKeyword::GNUAutoType &&
             "shouldn't need to mangle __auto_type!");
      mangleSourceName(AT->isDecltypeAuto() ? "<decltype-auto>" : "<auto>");
      Out << '@';
    } else {
      if (ResultType->isVoidType())
        ResultType = ResultType.getUnqualifiedType();
      mangleType(ResultType, Range, QMM_Result);
    }
  }

  // <argument-list> ::= X # void
  //                 ::= <type>+ @
  //                 ::= <type>* Z # varargs
  if (!Proto) {
    // Function types without prototypes can arise when mangling a function type
    // within an overloadable function in C. We mangle these as the absence of
    // any parameter types (not even an empty parameter list).
    Out << '@';
  } else if (Proto->getNumParams() == 0 && !Proto->isVariadic()) {
    Out << 'X';
  } else {
    // Happens for function pointer type arguments for example.
    for (const QualType &Arg : Proto->param_types())
      mangleArgumentType(Arg, Range);
    // <builtin-type>      ::= Z  # ellipsis
    if (Proto->isVariadic())
      Out << 'Z';
    else
      Out << '@';
  }

  mangleThrowSpecification(Proto);
}

void MicrosoftCXXNameMangler::mangleFunctionClass(const FunctionDecl *FD) {
  // <function-class>  ::= <member-function> E? # E designates a 64-bit 'this'
  //                                            # pointer. in 64-bit mode *all*
  //                                            # 'this' pointers are 64-bit.
  //                   ::= <global-function>
  // <member-function> ::= A # private: near
  //                   ::= B # private: far
  //                   ::= C # private: static near
  //                   ::= D # private: static far
  //                   ::= E # private: virtual near
  //                   ::= F # private: virtual far
  //                   ::= I # protected: near
  //                   ::= J # protected: far
  //                   ::= K # protected: static near
  //                   ::= L # protected: static far
  //                   ::= M # protected: virtual near
  //                   ::= N # protected: virtual far
  //                   ::= Q # public: near
  //                   ::= R # public: far
  //                   ::= S # public: static near
  //                   ::= T # public: static far
  //                   ::= U # public: virtual near
  //                   ::= V # public: virtual far
  // <global-function> ::= Y # global near
  //                   ::= Z # global far
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
    switch (MD->getAccess()) {
      case AS_none:
        llvm_unreachable("Unsupported access specifier");
      case AS_private:
        if (MD->isStatic())
          Out << 'C';
        else if (MD->isVirtual())
          Out << 'E';
        else
          Out << 'A';
        break;
      case AS_protected:
        if (MD->isStatic())
          Out << 'K';
        else if (MD->isVirtual())
          Out << 'M';
        else
          Out << 'I';
        break;
      case AS_public:
        if (MD->isStatic())
          Out << 'S';
        else if (MD->isVirtual())
          Out << 'U';
        else
          Out << 'Q';
    }
  } else {
    Out << 'Y';
  }
}
void MicrosoftCXXNameMangler::mangleCallingConvention(CallingConv CC) {
  // <calling-convention> ::= A # __cdecl
  //                      ::= B # __export __cdecl
  //                      ::= C # __pascal
  //                      ::= D # __export __pascal
  //                      ::= E # __thiscall
  //                      ::= F # __export __thiscall
  //                      ::= G # __stdcall
  //                      ::= H # __export __stdcall
  //                      ::= I # __fastcall
  //                      ::= J # __export __fastcall
  //                      ::= Q # __vectorcall
  // The 'export' calling conventions are from a bygone era
  // (*cough*Win16*cough*) when functions were declared for export with
  // that keyword. (It didn't actually export them, it just made them so
  // that they could be in a DLL and somebody from another module could call
  // them.)

  switch (CC) {
    default:
      llvm_unreachable("Unsupported CC for mangling");
    case CC_X86_64Win64:
    case CC_X86_64SysV:
    case CC_C: Out << 'A'; break;
    case CC_X86Pascal: Out << 'C'; break;
    case CC_X86ThisCall: Out << 'E'; break;
    case CC_X86StdCall: Out << 'G'; break;
    case CC_X86FastCall: Out << 'I'; break;
    case CC_X86VectorCall: Out << 'Q'; break;
  }
}
void MicrosoftCXXNameMangler::mangleCallingConvention(const FunctionType *T) {
  mangleCallingConvention(T->getCallConv());
}
void MicrosoftCXXNameMangler::mangleThrowSpecification(
                                                const FunctionProtoType *FT) {
  // <throw-spec> ::= Z # throw(...) (default)
  //              ::= @ # throw() or __declspec/__attribute__((nothrow))
  //              ::= <type>+
  // NOTE: Since the Microsoft compiler ignores throw specifications, they are
  // all actually mangled as 'Z'. (They're ignored because their associated
  // functionality isn't implemented, and probably never will be.)
  Out << 'Z';
}

void MicrosoftCXXNameMangler::mangleType(const UnresolvedUsingType *T,
                                         Qualifiers, SourceRange Range) {
  // Probably should be mangled as a template instantiation; need to see what
  // VC does first.
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this unresolved dependent type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

// <type>        ::= <union-type> | <struct-type> | <class-type> | <enum-type>
// <union-type>  ::= T <name>
// <struct-type> ::= U <name>
// <class-type>  ::= V <name>
// <enum-type>   ::= W4 <name>
void MicrosoftCXXNameMangler::mangleType(const EnumType *T, Qualifiers,
                                         SourceRange) {
  mangleType(cast<TagType>(T)->getDecl());
}
void MicrosoftCXXNameMangler::mangleType(const RecordType *T, Qualifiers,
                                         SourceRange) {
  mangleType(cast<TagType>(T)->getDecl());
}
void MicrosoftCXXNameMangler::mangleType(const TagDecl *TD) {
  switch (TD->getTagKind()) {
    case TTK_Union:
      Out << 'T';
      break;
    case TTK_Struct:
    case TTK_Interface:
      Out << 'U';
      break;
    case TTK_Class:
      Out << 'V';
      break;
    case TTK_Enum:
      Out << "W4";
      break;
  }
  mangleName(TD);
}

// <type>       ::= <array-type>
// <array-type> ::= <pointer-cvr-qualifiers> <cvr-qualifiers>
//                  [Y <dimension-count> <dimension>+]
//                  <element-type> # as global, E is never required
// It's supposed to be the other way around, but for some strange reason, it
// isn't. Today this behavior is retained for the sole purpose of backwards
// compatibility.
void MicrosoftCXXNameMangler::mangleDecayedArrayType(const ArrayType *T) {
  // This isn't a recursive mangling, so now we have to do it all in this
  // one call.
  manglePointerCVQualifiers(T->getElementType().getQualifiers());
  mangleType(T->getElementType(), SourceRange());
}
void MicrosoftCXXNameMangler::mangleType(const ConstantArrayType *T, Qualifiers,
                                         SourceRange) {
  llvm_unreachable("Should have been special cased");
}
void MicrosoftCXXNameMangler::mangleType(const VariableArrayType *T, Qualifiers,
                                         SourceRange) {
  llvm_unreachable("Should have been special cased");
}
void MicrosoftCXXNameMangler::mangleType(const DependentSizedArrayType *T,
                                         Qualifiers, SourceRange) {
  llvm_unreachable("Should have been special cased");
}
void MicrosoftCXXNameMangler::mangleType(const IncompleteArrayType *T,
                                         Qualifiers, SourceRange) {
  llvm_unreachable("Should have been special cased");
}
void MicrosoftCXXNameMangler::mangleArrayType(const ArrayType *T) {
  QualType ElementTy(T, 0);
  SmallVector<llvm::APInt, 3> Dimensions;
  for (;;) {
    if (ElementTy->isConstantArrayType()) {
      const ConstantArrayType *CAT =
          getASTContext().getAsConstantArrayType(ElementTy);
      Dimensions.push_back(CAT->getSize());
      ElementTy = CAT->getElementType();
    } else if (ElementTy->isIncompleteArrayType()) {
      const IncompleteArrayType *IAT =
          getASTContext().getAsIncompleteArrayType(ElementTy);
      Dimensions.push_back(llvm::APInt(32, 0));
      ElementTy = IAT->getElementType();
    } else if (ElementTy->isVariableArrayType()) {
      const VariableArrayType *VAT =
        getASTContext().getAsVariableArrayType(ElementTy);
      Dimensions.push_back(llvm::APInt(32, 0));
      ElementTy = VAT->getElementType();
    } else if (ElementTy->isDependentSizedArrayType()) {
      // The dependent expression has to be folded into a constant (TODO).
      const DependentSizedArrayType *DSAT =
        getASTContext().getAsDependentSizedArrayType(ElementTy);
      DiagnosticsEngine &Diags = Context.getDiags();
      unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
        "cannot mangle this dependent-length array yet");
      Diags.Report(DSAT->getSizeExpr()->getExprLoc(), DiagID)
        << DSAT->getBracketsRange();
      return;
    } else {
      break;
    }
  }
  Out << 'Y';
  // <dimension-count> ::= <number> # number of extra dimensions
  mangleNumber(Dimensions.size());
  for (const llvm::APInt &Dimension : Dimensions)
    mangleNumber(Dimension.getLimitedValue());
  mangleType(ElementTy, SourceRange(), QMM_Escape);
}

// <type>                   ::= <pointer-to-member-type>
// <pointer-to-member-type> ::= <pointer-cvr-qualifiers> <cvr-qualifiers>
//                                                          <class name> <type>
void MicrosoftCXXNameMangler::mangleType(const MemberPointerType *T, Qualifiers Quals,
                                         SourceRange Range) {
  QualType PointeeType = T->getPointeeType();
  manglePointerCVQualifiers(Quals);
  manglePointerExtQualifiers(Quals, PointeeType);
  if (const FunctionProtoType *FPT = PointeeType->getAs<FunctionProtoType>()) {
    Out << '8';
    mangleName(T->getClass()->castAs<RecordType>()->getDecl());
    mangleFunctionType(FPT, nullptr, true);
  } else {
    mangleQualifiers(PointeeType.getQualifiers(), true);
    mangleName(T->getClass()->castAs<RecordType>()->getDecl());
    mangleType(PointeeType, Range, QMM_Drop);
  }
}

void MicrosoftCXXNameMangler::mangleType(const TemplateTypeParmType *T,
                                         Qualifiers, SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this template type parameter type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const SubstTemplateTypeParmPackType *T,
                                         Qualifiers, SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this substituted parameter pack yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

// <type> ::= <pointer-type>
// <pointer-type> ::= E? <pointer-cvr-qualifiers> <cvr-qualifiers> <type>
//                       # the E is required for 64-bit non-static pointers
void MicrosoftCXXNameMangler::mangleType(const PointerType *T, Qualifiers Quals,
                                         SourceRange Range) {
  QualType PointeeType = T->getPointeeType();
  manglePointerCVQualifiers(Quals);
  manglePointerExtQualifiers(Quals, PointeeType);
  mangleType(PointeeType, Range);
}
void MicrosoftCXXNameMangler::mangleType(const ObjCObjectPointerType *T,
                                         Qualifiers Quals, SourceRange Range) {
  QualType PointeeType = T->getPointeeType();
  manglePointerCVQualifiers(Quals);
  manglePointerExtQualifiers(Quals, PointeeType);
  // Object pointers never have qualifiers.
  Out << 'A';
  mangleType(PointeeType, Range);
}

// <type> ::= <reference-type>
// <reference-type> ::= A E? <cvr-qualifiers> <type>
//                 # the E is required for 64-bit non-static lvalue references
void MicrosoftCXXNameMangler::mangleType(const LValueReferenceType *T,
                                         Qualifiers Quals, SourceRange Range) {
  QualType PointeeType = T->getPointeeType();
  Out << (Quals.hasVolatile() ? 'B' : 'A');
  manglePointerExtQualifiers(Quals, PointeeType);
  mangleType(PointeeType, Range);
}

// <type> ::= <r-value-reference-type>
// <r-value-reference-type> ::= $$Q E? <cvr-qualifiers> <type>
//                 # the E is required for 64-bit non-static rvalue references
void MicrosoftCXXNameMangler::mangleType(const RValueReferenceType *T,
                                         Qualifiers Quals, SourceRange Range) {
  QualType PointeeType = T->getPointeeType();
  Out << (Quals.hasVolatile() ? "$$R" : "$$Q");
  manglePointerExtQualifiers(Quals, PointeeType);
  mangleType(PointeeType, Range);
}

void MicrosoftCXXNameMangler::mangleType(const ComplexType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this complex number type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const VectorType *T, Qualifiers Quals,
                                         SourceRange Range) {
  const BuiltinType *ET = T->getElementType()->getAs<BuiltinType>();
  assert(ET && "vectors with non-builtin elements are unsupported");
  uint64_t Width = getASTContext().getTypeSize(T);
  // Pattern match exactly the typedefs in our intrinsic headers.  Anything that
  // doesn't match the Intel types uses a custom mangling below.
  bool IsBuiltin = true;
  llvm::Triple::ArchType AT =
      getASTContext().getTargetInfo().getTriple().getArch();
  if (AT == llvm::Triple::x86 || AT == llvm::Triple::x86_64) {
    if (Width == 64 && ET->getKind() == BuiltinType::LongLong) {
      Out << "T__m64";
    } else if (Width >= 128) {
      if (ET->getKind() == BuiltinType::Float)
        Out << "T__m" << Width;
      else if (ET->getKind() == BuiltinType::LongLong)
        Out << "T__m" << Width << 'i';
      else if (ET->getKind() == BuiltinType::Double)
        Out << "U__m" << Width << 'd';
      else
        IsBuiltin = false;
    } else {
      IsBuiltin = false;
    }
  } else {
    IsBuiltin = false;
  }

  if (!IsBuiltin) {
    // The MS ABI doesn't have a special mangling for vector types, so we define
    // our own mangling to handle uses of __vector_size__ on user-specified
    // types, and for extensions like __v4sf.
    Out << "T__clang_vec" << T->getNumElements() << '_';
    mangleType(ET, Quals, Range);
  }

  Out << "@@";
}

void MicrosoftCXXNameMangler::mangleType(const ExtVectorType *T,
                                         Qualifiers Quals, SourceRange Range) {
  mangleType(static_cast<const VectorType *>(T), Quals, Range);
}
void MicrosoftCXXNameMangler::mangleType(const DependentSizedExtVectorType *T,
                                         Qualifiers, SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this dependent-sized extended vector type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const ObjCInterfaceType *T, Qualifiers,
                                         SourceRange) {
  // ObjC interfaces have structs underlying them.
  Out << 'U';
  mangleName(T->getDecl());
}

void MicrosoftCXXNameMangler::mangleType(const ObjCObjectType *T, Qualifiers,
                                         SourceRange Range) {
  // We don't allow overloading by different protocol qualification,
  // so mangling them isn't necessary.
  mangleType(T->getBaseType(), Range);
}

void MicrosoftCXXNameMangler::mangleType(const BlockPointerType *T,
                                         Qualifiers Quals, SourceRange Range) {
  QualType PointeeType = T->getPointeeType();
  manglePointerCVQualifiers(Quals);
  manglePointerExtQualifiers(Quals, PointeeType);

  Out << "_E";

  mangleFunctionType(PointeeType->castAs<FunctionProtoType>());
}

void MicrosoftCXXNameMangler::mangleType(const InjectedClassNameType *,
                                         Qualifiers, SourceRange) {
  llvm_unreachable("Cannot mangle injected class name type.");
}

void MicrosoftCXXNameMangler::mangleType(const TemplateSpecializationType *T,
                                         Qualifiers, SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this template specialization type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const DependentNameType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this dependent name type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(
    const DependentTemplateSpecializationType *T, Qualifiers,
    SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this dependent template specialization type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const PackExpansionType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this pack expansion yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const TypeOfType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this typeof(type) yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const TypeOfExprType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this typeof(expression) yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const DecltypeType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this decltype() yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const UnaryTransformType *T,
                                         Qualifiers, SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this unary transform type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const AutoType *T, Qualifiers,
                                         SourceRange Range) {
  assert(T->getDeducedType().isNull() && "expecting a dependent type!");

  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this 'auto' type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftCXXNameMangler::mangleType(const AtomicType *T, Qualifiers,
                                         SourceRange Range) {
  DiagnosticsEngine &Diags = Context.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this C11 atomic type yet");
  Diags.Report(Range.getBegin(), DiagID)
    << Range;
}

void MicrosoftMangleContextImpl::mangleCXXName(const NamedDecl *D,
                                               raw_ostream &Out) {
  assert((isa<FunctionDecl>(D) || isa<VarDecl>(D)) &&
         "Invalid mangleName() call, argument is not a variable or function!");
  assert(!isa<CXXConstructorDecl>(D) && !isa<CXXDestructorDecl>(D) &&
         "Invalid mangleName() call on 'structor decl!");

  PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                 getASTContext().getSourceManager(),
                                 "Mangling declaration");

  MicrosoftCXXNameMangler Mangler(*this, Out);
  return Mangler.mangle(D);
}

// <this-adjustment> ::= <no-adjustment> | <static-adjustment> |
//                       <virtual-adjustment>
// <no-adjustment>      ::= A # private near
//                      ::= B # private far
//                      ::= I # protected near
//                      ::= J # protected far
//                      ::= Q # public near
//                      ::= R # public far
// <static-adjustment>  ::= G <static-offset> # private near
//                      ::= H <static-offset> # private far
//                      ::= O <static-offset> # protected near
//                      ::= P <static-offset> # protected far
//                      ::= W <static-offset> # public near
//                      ::= X <static-offset> # public far
// <virtual-adjustment> ::= $0 <virtual-shift> <static-offset> # private near
//                      ::= $1 <virtual-shift> <static-offset> # private far
//                      ::= $2 <virtual-shift> <static-offset> # protected near
//                      ::= $3 <virtual-shift> <static-offset> # protected far
//                      ::= $4 <virtual-shift> <static-offset> # public near
//                      ::= $5 <virtual-shift> <static-offset> # public far
// <virtual-shift>      ::= <vtordisp-shift> | <vtordispex-shift>
// <vtordisp-shift>     ::= <offset-to-vtordisp>
// <vtordispex-shift>   ::= <offset-to-vbptr> <vbase-offset-offset>
//                          <offset-to-vtordisp>
static void mangleThunkThisAdjustment(const CXXMethodDecl *MD,
                                      const ThisAdjustment &Adjustment,
                                      MicrosoftCXXNameMangler &Mangler,
                                      raw_ostream &Out) {
  if (!Adjustment.Virtual.isEmpty()) {
    Out << '$';
    char AccessSpec;
    switch (MD->getAccess()) {
    case AS_none:
      llvm_unreachable("Unsupported access specifier");
    case AS_private:
      AccessSpec = '0';
      break;
    case AS_protected:
      AccessSpec = '2';
      break;
    case AS_public:
      AccessSpec = '4';
    }
    if (Adjustment.Virtual.Microsoft.VBPtrOffset) {
      Out << 'R' << AccessSpec;
      Mangler.mangleNumber(
          static_cast<uint32_t>(Adjustment.Virtual.Microsoft.VBPtrOffset));
      Mangler.mangleNumber(
          static_cast<uint32_t>(Adjustment.Virtual.Microsoft.VBOffsetOffset));
      Mangler.mangleNumber(
          static_cast<uint32_t>(Adjustment.Virtual.Microsoft.VtordispOffset));
      Mangler.mangleNumber(static_cast<uint32_t>(Adjustment.NonVirtual));
    } else {
      Out << AccessSpec;
      Mangler.mangleNumber(
          static_cast<uint32_t>(Adjustment.Virtual.Microsoft.VtordispOffset));
      Mangler.mangleNumber(-static_cast<uint32_t>(Adjustment.NonVirtual));
    }
  } else if (Adjustment.NonVirtual != 0) {
    switch (MD->getAccess()) {
    case AS_none:
      llvm_unreachable("Unsupported access specifier");
    case AS_private:
      Out << 'G';
      break;
    case AS_protected:
      Out << 'O';
      break;
    case AS_public:
      Out << 'W';
    }
    Mangler.mangleNumber(-static_cast<uint32_t>(Adjustment.NonVirtual));
  } else {
    switch (MD->getAccess()) {
    case AS_none:
      llvm_unreachable("Unsupported access specifier");
    case AS_private:
      Out << 'A';
      break;
    case AS_protected:
      Out << 'I';
      break;
    case AS_public:
      Out << 'Q';
    }
  }
}

void
MicrosoftMangleContextImpl::mangleVirtualMemPtrThunk(const CXXMethodDecl *MD,
                                                     raw_ostream &Out) {
  MicrosoftVTableContext *VTContext =
      cast<MicrosoftVTableContext>(getASTContext().getVTableContext());
  const MicrosoftVTableContext::MethodVFTableLocation &ML =
      VTContext->getMethodVFTableLocation(GlobalDecl(MD));

  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01?";
  Mangler.mangleVirtualMemPtrThunk(MD, ML);
}

void MicrosoftMangleContextImpl::mangleThunk(const CXXMethodDecl *MD,
                                             const ThunkInfo &Thunk,
                                             raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Out << "\01?";
  Mangler.mangleName(MD);
  mangleThunkThisAdjustment(MD, Thunk.This, Mangler, Out);
  if (!Thunk.Return.isEmpty())
    assert(Thunk.Method != nullptr &&
           "Thunk info should hold the overridee decl");

  const CXXMethodDecl *DeclForFPT = Thunk.Method ? Thunk.Method : MD;
  Mangler.mangleFunctionType(
      DeclForFPT->getType()->castAs<FunctionProtoType>(), MD);
}

void MicrosoftMangleContextImpl::mangleCXXDtorThunk(
    const CXXDestructorDecl *DD, CXXDtorType Type,
    const ThisAdjustment &Adjustment, raw_ostream &Out) {
  // FIXME: Actually, the dtor thunk should be emitted for vector deleting
  // dtors rather than scalar deleting dtors. Just use the vector deleting dtor
  // mangling manually until we support both deleting dtor types.
  assert(Type == Dtor_Deleting);
  MicrosoftCXXNameMangler Mangler(*this, Out, DD, Type);
  Out << "\01??_E";
  Mangler.mangleName(DD->getParent());
  mangleThunkThisAdjustment(DD, Adjustment, Mangler, Out);
  Mangler.mangleFunctionType(DD->getType()->castAs<FunctionProtoType>(), DD);
}

void MicrosoftMangleContextImpl::mangleCXXVFTable(
    const CXXRecordDecl *Derived, ArrayRef<const CXXRecordDecl *> BasePath,
    raw_ostream &Out) {
  // <mangled-name> ::= ?_7 <class-name> <storage-class>
  //                    <cvr-qualifiers> [<name>] @
  // NOTE: <cvr-qualifiers> here is always 'B' (const). <storage-class>
  // is always '6' for vftables.
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_7";
  Mangler.mangleName(Derived);
  Mangler.getStream() << "6B"; // '6' for vftable, 'B' for const.
  for (const CXXRecordDecl *RD : BasePath)
    Mangler.mangleName(RD);
  Mangler.getStream() << '@';
}

void MicrosoftMangleContextImpl::mangleCXXVBTable(
    const CXXRecordDecl *Derived, ArrayRef<const CXXRecordDecl *> BasePath,
    raw_ostream &Out) {
  // <mangled-name> ::= ?_8 <class-name> <storage-class>
  //                    <cvr-qualifiers> [<name>] @
  // NOTE: <cvr-qualifiers> here is always 'B' (const). <storage-class>
  // is always '7' for vbtables.
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_8";
  Mangler.mangleName(Derived);
  Mangler.getStream() << "7B";  // '7' for vbtable, 'B' for const.
  for (const CXXRecordDecl *RD : BasePath)
    Mangler.mangleName(RD);
  Mangler.getStream() << '@';
}

void MicrosoftMangleContextImpl::mangleCXXRTTI(QualType T, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_R0";
  Mangler.mangleType(T, SourceRange(), MicrosoftCXXNameMangler::QMM_Result);
  Mangler.getStream() << "@8";
}

void MicrosoftMangleContextImpl::mangleCXXRTTIName(QualType T,
                                                   raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << '.';
  Mangler.mangleType(T, SourceRange(), MicrosoftCXXNameMangler::QMM_Result);
}

void MicrosoftMangleContextImpl::mangleCXXCatchHandlerType(QualType T,
                                                           uint32_t Flags,
                                                           raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "llvm.eh.handlertype.";
  Mangler.mangleType(T, SourceRange(), MicrosoftCXXNameMangler::QMM_Result);
  Mangler.getStream() << '.' << Flags;
}

void MicrosoftMangleContextImpl::mangleCXXVirtualDisplacementMap(
    const CXXRecordDecl *SrcRD, const CXXRecordDecl *DstRD, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_K";
  Mangler.mangleName(SrcRD);
  Mangler.getStream() << "$C";
  Mangler.mangleName(DstRD);
}

void MicrosoftMangleContextImpl::mangleCXXThrowInfo(QualType T,
                                                    bool IsConst,
                                                    bool IsVolatile,
                                                    uint32_t NumEntries,
                                                    raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_TI";
  if (IsConst)
    Mangler.getStream() << 'C';
  if (IsVolatile)
    Mangler.getStream() << 'V';
  Mangler.getStream() << NumEntries;
  Mangler.mangleType(T, SourceRange(), MicrosoftCXXNameMangler::QMM_Result);
}

void MicrosoftMangleContextImpl::mangleCXXCatchableTypeArray(
    QualType T, uint32_t NumEntries, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_CTA";
  Mangler.getStream() << NumEntries;
  Mangler.mangleType(T, SourceRange(), MicrosoftCXXNameMangler::QMM_Result);
}

void MicrosoftMangleContextImpl::mangleCXXCatchableType(
    QualType T, const CXXConstructorDecl *CD, CXXCtorType CT, uint32_t Size,
    uint32_t NVOffset, int32_t VBPtrOffset, uint32_t VBIndex,
    raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "_CT";

  llvm::SmallString<64> RTTIMangling;
  {
    llvm::raw_svector_ostream Stream(RTTIMangling);
    mangleCXXRTTI(T, Stream);
  }
  Mangler.getStream() << RTTIMangling.substr(1);

  // VS2015 CTP6 omits the copy-constructor in the mangled name.  This name is,
  // in fact, superfluous but I'm not sure the change was made consciously.
  // TODO: Revisit this when VS2015 gets released.
  llvm::SmallString<64> CopyCtorMangling;
  if (CD) {
    llvm::raw_svector_ostream Stream(CopyCtorMangling);
    mangleCXXCtor(CD, CT, Stream);
  }
  Mangler.getStream() << CopyCtorMangling.substr(1);

  Mangler.getStream() << Size;
  if (VBPtrOffset == -1) {
    if (NVOffset) {
      Mangler.getStream() << NVOffset;
    }
  } else {
    Mangler.getStream() << NVOffset;
    Mangler.getStream() << VBPtrOffset;
    Mangler.getStream() << VBIndex;
  }
}

void MicrosoftMangleContextImpl::mangleCXXRTTIBaseClassDescriptor(
    const CXXRecordDecl *Derived, uint32_t NVOffset, int32_t VBPtrOffset,
    uint32_t VBTableOffset, uint32_t Flags, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_R1";
  Mangler.mangleNumber(NVOffset);
  Mangler.mangleNumber(VBPtrOffset);
  Mangler.mangleNumber(VBTableOffset);
  Mangler.mangleNumber(Flags);
  Mangler.mangleName(Derived);
  Mangler.getStream() << "8";
}

void MicrosoftMangleContextImpl::mangleCXXRTTIBaseClassArray(
    const CXXRecordDecl *Derived, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_R2";
  Mangler.mangleName(Derived);
  Mangler.getStream() << "8";
}

void MicrosoftMangleContextImpl::mangleCXXRTTIClassHierarchyDescriptor(
    const CXXRecordDecl *Derived, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_R3";
  Mangler.mangleName(Derived);
  Mangler.getStream() << "8";
}

void MicrosoftMangleContextImpl::mangleCXXRTTICompleteObjectLocator(
    const CXXRecordDecl *Derived, ArrayRef<const CXXRecordDecl *> BasePath,
    raw_ostream &Out) {
  // <mangled-name> ::= ?_R4 <class-name> <storage-class>
  //                    <cvr-qualifiers> [<name>] @
  // NOTE: <cvr-qualifiers> here is always 'B' (const). <storage-class>
  // is always '6' for vftables.
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_R4";
  Mangler.mangleName(Derived);
  Mangler.getStream() << "6B"; // '6' for vftable, 'B' for const.
  for (const CXXRecordDecl *RD : BasePath)
    Mangler.mangleName(RD);
  Mangler.getStream() << '@';
}

void MicrosoftMangleContextImpl::mangleSEHFilterExpression(
    const NamedDecl *EnclosingDecl, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  // The function body is in the same comdat as the function with the handler,
  // so the numbering here doesn't have to be the same across TUs.
  //
  // <mangled-name> ::= ?filt$ <filter-number> @0
  Mangler.getStream() << "\01?filt$" << SEHFilterIds[EnclosingDecl]++ << "@0@";
  Mangler.mangleName(EnclosingDecl);
}

void MicrosoftMangleContextImpl::mangleSEHFinallyBlock(
    const NamedDecl *EnclosingDecl, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  // The function body is in the same comdat as the function with the handler,
  // so the numbering here doesn't have to be the same across TUs.
  //
  // <mangled-name> ::= ?fin$ <filter-number> @0
  Mangler.getStream() << "\01?fin$" << SEHFinallyIds[EnclosingDecl]++ << "@0@";
  Mangler.mangleName(EnclosingDecl);
}

void MicrosoftMangleContextImpl::mangleTypeName(QualType T, raw_ostream &Out) {
  // This is just a made up unique string for the purposes of tbaa.  undname
  // does *not* know how to demangle it.
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << '?';
  Mangler.mangleType(T, SourceRange());
}

void MicrosoftMangleContextImpl::mangleCXXCtor(const CXXConstructorDecl *D,
                                               CXXCtorType Type,
                                               raw_ostream &Out) {
  MicrosoftCXXNameMangler mangler(*this, Out, D, Type);
  mangler.mangle(D);
}

void MicrosoftMangleContextImpl::mangleCXXDtor(const CXXDestructorDecl *D,
                                               CXXDtorType Type,
                                               raw_ostream &Out) {
  MicrosoftCXXNameMangler mangler(*this, Out, D, Type);
  mangler.mangle(D);
}

void MicrosoftMangleContextImpl::mangleReferenceTemporary(const VarDecl *VD,
                                                          unsigned,
                                                          raw_ostream &) {
  unsigned DiagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
    "cannot mangle this reference temporary yet");
  getDiags().Report(VD->getLocation(), DiagID);
}

void MicrosoftMangleContextImpl::mangleThreadSafeStaticGuardVariable(
    const VarDecl *VD, unsigned GuardNum, raw_ostream &Out) {
  MicrosoftCXXNameMangler Mangler(*this, Out);

  Mangler.getStream() << "\01?$TSS" << GuardNum << '@';
  Mangler.mangleNestedName(VD);
}

void MicrosoftMangleContextImpl::mangleStaticGuardVariable(const VarDecl *VD,
                                                           raw_ostream &Out) {
  // <guard-name> ::= ?_B <postfix> @5 <scope-depth>
  //              ::= ?__J <postfix> @5 <scope-depth>
  //              ::= ?$S <guard-num> @ <postfix> @4IA

  // The first mangling is what MSVC uses to guard static locals in inline
  // functions.  It uses a different mangling in external functions to support
  // guarding more than 32 variables.  MSVC rejects inline functions with more
  // than 32 static locals.  We don't fully implement the second mangling
  // because those guards are not externally visible, and instead use LLVM's
  // default renaming when creating a new guard variable.
  MicrosoftCXXNameMangler Mangler(*this, Out);

  bool Visible = VD->isExternallyVisible();
  if (Visible) {
    Mangler.getStream() << (VD->getTLSKind() ? "\01??__J" : "\01??_B");
  } else {
    Mangler.getStream() << "\01?$S1@";
  }
  unsigned ScopeDepth = 0;
  if (Visible && !getNextDiscriminator(VD, ScopeDepth))
    // If we do not have a discriminator and are emitting a guard variable for
    // use at global scope, then mangling the nested name will not be enough to
    // remove ambiguities.
    Mangler.mangle(VD, "");
  else
    Mangler.mangleNestedName(VD);
  Mangler.getStream() << (Visible ? "@5" : "@4IA");
  if (ScopeDepth)
    Mangler.mangleNumber(ScopeDepth);
}

void MicrosoftMangleContextImpl::mangleInitFiniStub(const VarDecl *D,
                                                    raw_ostream &Out,
                                                    char CharCode) {
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??__" << CharCode;
  Mangler.mangleName(D);
  if (D->isStaticDataMember()) {
    Mangler.mangleVariableEncoding(D);
    Mangler.getStream() << '@';
  }
  // This is the function class mangling.  These stubs are global, non-variadic,
  // cdecl functions that return void and take no args.
  Mangler.getStream() << "YAXXZ";
}

void MicrosoftMangleContextImpl::mangleDynamicInitializer(const VarDecl *D,
                                                          raw_ostream &Out) {
  // <initializer-name> ::= ?__E <name> YAXXZ
  mangleInitFiniStub(D, Out, 'E');
}

void
MicrosoftMangleContextImpl::mangleDynamicAtExitDestructor(const VarDecl *D,
                                                          raw_ostream &Out) {
  // <destructor-name> ::= ?__F <name> YAXXZ
  mangleInitFiniStub(D, Out, 'F');
}

void MicrosoftMangleContextImpl::mangleStringLiteral(const StringLiteral *SL,
                                                     raw_ostream &Out) {
  // <char-type> ::= 0   # char
  //             ::= 1   # wchar_t
  //             ::= ??? # char16_t/char32_t will need a mangling too...
  //
  // <literal-length> ::= <non-negative integer>  # the length of the literal
  //
  // <encoded-crc>    ::= <hex digit>+ @          # crc of the literal including
  //                                              # null-terminator
  //
  // <encoded-string> ::= <simple character>           # uninteresting character
  //                  ::= '?$' <hex digit> <hex digit> # these two nibbles
  //                                                   # encode the byte for the
  //                                                   # character
  //                  ::= '?' [a-z]                    # \xe1 - \xfa
  //                  ::= '?' [A-Z]                    # \xc1 - \xda
  //                  ::= '?' [0-9]                    # [,/\:. \n\t'-]
  //
  // <literal> ::= '??_C@_' <char-type> <literal-length> <encoded-crc>
  //               <encoded-string> '@'
  MicrosoftCXXNameMangler Mangler(*this, Out);
  Mangler.getStream() << "\01??_C@_";

  // <char-type>: The "kind" of string literal is encoded into the mangled name.
  if (SL->isWide())
    Mangler.getStream() << '1';
  else
    Mangler.getStream() << '0';

  // <literal-length>: The next part of the mangled name consists of the length
  // of the string.
  // The StringLiteral does not consider the NUL terminator byte(s) but the
  // mangling does.
  // N.B. The length is in terms of bytes, not characters.
  Mangler.mangleNumber(SL->getByteLength() + SL->getCharByteWidth());

  auto GetLittleEndianByte = [&Mangler, &SL](unsigned Index) {
    unsigned CharByteWidth = SL->getCharByteWidth();
    uint32_t CodeUnit = SL->getCodeUnit(Index / CharByteWidth);
    unsigned OffsetInCodeUnit = Index % CharByteWidth;
    return static_cast<char>((CodeUnit >> (8 * OffsetInCodeUnit)) & 0xff);
  };

  auto GetBigEndianByte = [&Mangler, &SL](unsigned Index) {
    unsigned CharByteWidth = SL->getCharByteWidth();
    uint32_t CodeUnit = SL->getCodeUnit(Index / CharByteWidth);
    unsigned OffsetInCodeUnit = (CharByteWidth - 1) - (Index % CharByteWidth);
    return static_cast<char>((CodeUnit >> (8 * OffsetInCodeUnit)) & 0xff);
  };

  // CRC all the bytes of the StringLiteral.
  llvm::JamCRC JC;
  for (unsigned I = 0, E = SL->getByteLength(); I != E; ++I)
    JC.update(GetLittleEndianByte(I));

  // The NUL terminator byte(s) were not present earlier,
  // we need to manually process those bytes into the CRC.
  for (unsigned NullTerminator = 0; NullTerminator < SL->getCharByteWidth();
       ++NullTerminator)
    JC.update('\x00');

  // <encoded-crc>: The CRC is encoded utilizing the standard number mangling
  // scheme.
  Mangler.mangleNumber(JC.getCRC());

  // <encoded-string>: The mangled name also contains the first 32 _characters_
  // (including null-terminator bytes) of the StringLiteral.
  // Each character is encoded by splitting them into bytes and then encoding
  // the constituent bytes.
  auto MangleByte = [&Mangler](char Byte) {
    // There are five different manglings for characters:
    // - [a-zA-Z0-9_$]: A one-to-one mapping.
    // - ?[a-z]: The range from \xe1 to \xfa.
    // - ?[A-Z]: The range from \xc1 to \xda.
    // - ?[0-9]: The set of [,/\:. \n\t'-].
    // - ?$XX: A fallback which maps nibbles.
    if (isIdentifierBody(Byte, /*AllowDollar=*/true)) {
      Mangler.getStream() << Byte;
    } else if (isLetter(Byte & 0x7f)) {
      Mangler.getStream() << '?' << static_cast<char>(Byte & 0x7f);
    } else {
      const char SpecialChars[] = {',', '/',  '\\', ':',  '.',
                                   ' ', '\n', '\t', '\'', '-'};
      const char *Pos =
          std::find(std::begin(SpecialChars), std::end(SpecialChars), Byte);
      if (Pos != std::end(SpecialChars)) {
        Mangler.getStream() << '?' << (Pos - std::begin(SpecialChars));
      } else {
        Mangler.getStream() << "?$";
        Mangler.getStream() << static_cast<char>('A' + ((Byte >> 4) & 0xf));
        Mangler.getStream() << static_cast<char>('A' + (Byte & 0xf));
      }
    }
  };

  // Enforce our 32 character max.
  unsigned NumCharsToMangle = std::min(32U, SL->getLength());
  for (unsigned I = 0, E = NumCharsToMangle * SL->getCharByteWidth(); I != E;
       ++I)
    if (SL->isWide())
      MangleByte(GetBigEndianByte(I));
    else
      MangleByte(GetLittleEndianByte(I));

  // Encode the NUL terminator if there is room.
  if (NumCharsToMangle < 32)
    for (unsigned NullTerminator = 0; NullTerminator < SL->getCharByteWidth();
         ++NullTerminator)
      MangleByte(0);

  Mangler.getStream() << '@';
}

MicrosoftMangleContext *
MicrosoftMangleContext::create(ASTContext &Context, DiagnosticsEngine &Diags) {
  return new MicrosoftMangleContextImpl(Context, Diags);
}
