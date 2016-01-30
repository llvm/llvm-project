//===--- SemaInit.cpp - Semantic Analysis for Initializers ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements semantic analysis for initializers.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Initialization.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Designator.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
using namespace clang;

//===----------------------------------------------------------------------===//
// Sema Initialization Checking
//===----------------------------------------------------------------------===//

/// \brief Check whether T is compatible with a wide character type (wchar_t,
/// char16_t or char32_t).
static bool IsWideCharCompatible(QualType T, ASTContext &Context) {
  if (Context.typesAreCompatible(Context.getWideCharType(), T))
    return true;
  if (Context.getLangOpts().CPlusPlus || Context.getLangOpts().C11) {
    return Context.typesAreCompatible(Context.Char16Ty, T) ||
           Context.typesAreCompatible(Context.Char32Ty, T);
  }
  return false;
}

enum StringInitFailureKind {
  SIF_None,
  SIF_NarrowStringIntoWideChar,
  SIF_WideStringIntoChar,
  SIF_IncompatWideStringIntoWideChar,
  SIF_Other
};

/// \brief Check whether the array of type AT can be initialized by the Init
/// expression by means of string initialization. Returns SIF_None if so,
/// otherwise returns a StringInitFailureKind that describes why the
/// initialization would not work.
static StringInitFailureKind IsStringInit(Expr *Init, const ArrayType *AT,
                                          ASTContext &Context) {
  if (!isa<ConstantArrayType>(AT) && !isa<IncompleteArrayType>(AT))
    return SIF_Other;

  // See if this is a string literal or @encode.
  Init = Init->IgnoreParens();

  // Handle @encode, which is a narrow string.
  if (isa<ObjCEncodeExpr>(Init) && AT->getElementType()->isCharType())
    return SIF_None;

  // Otherwise we can only handle string literals.
  StringLiteral *SL = dyn_cast<StringLiteral>(Init);
  if (!SL)
    return SIF_Other;

  const QualType ElemTy =
      Context.getCanonicalType(AT->getElementType()).getUnqualifiedType();

  switch (SL->getKind()) {
  case StringLiteral::Ascii:
  case StringLiteral::UTF8:
    // char array can be initialized with a narrow string.
    // Only allow char x[] = "foo";  not char x[] = L"foo";
    if (ElemTy->isCharType())
      return SIF_None;
    if (IsWideCharCompatible(ElemTy, Context))
      return SIF_NarrowStringIntoWideChar;
    return SIF_Other;
  // C99 6.7.8p15 (with correction from DR343), or C11 6.7.9p15:
  // "An array with element type compatible with a qualified or unqualified
  // version of wchar_t, char16_t, or char32_t may be initialized by a wide
  // string literal with the corresponding encoding prefix (L, u, or U,
  // respectively), optionally enclosed in braces.
  case StringLiteral::UTF16:
    if (Context.typesAreCompatible(Context.Char16Ty, ElemTy))
      return SIF_None;
    if (ElemTy->isCharType())
      return SIF_WideStringIntoChar;
    if (IsWideCharCompatible(ElemTy, Context))
      return SIF_IncompatWideStringIntoWideChar;
    return SIF_Other;
  case StringLiteral::UTF32:
    if (Context.typesAreCompatible(Context.Char32Ty, ElemTy))
      return SIF_None;
    if (ElemTy->isCharType())
      return SIF_WideStringIntoChar;
    if (IsWideCharCompatible(ElemTy, Context))
      return SIF_IncompatWideStringIntoWideChar;
    return SIF_Other;
  case StringLiteral::Wide:
    if (Context.typesAreCompatible(Context.getWideCharType(), ElemTy))
      return SIF_None;
    if (ElemTy->isCharType())
      return SIF_WideStringIntoChar;
    if (IsWideCharCompatible(ElemTy, Context))
      return SIF_IncompatWideStringIntoWideChar;
    return SIF_Other;
  }

  llvm_unreachable("missed a StringLiteral kind?");
}

static StringInitFailureKind IsStringInit(Expr *init, QualType declType,
                                          ASTContext &Context) {
  const ArrayType *arrayType = Context.getAsArrayType(declType);
  if (!arrayType)
    return SIF_Other;
  return IsStringInit(init, arrayType, Context);
}

/// Update the type of a string literal, including any surrounding parentheses,
/// to match the type of the object which it is initializing.
static void updateStringLiteralType(Expr *E, QualType Ty) {
  while (true) {
    E->setType(Ty);
    if (isa<StringLiteral>(E) || isa<ObjCEncodeExpr>(E))
      break;
    else if (ParenExpr *PE = dyn_cast<ParenExpr>(E))
      E = PE->getSubExpr();
    else if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E))
      E = UO->getSubExpr();
    else if (GenericSelectionExpr *GSE = dyn_cast<GenericSelectionExpr>(E))
      E = GSE->getResultExpr();
    else
      llvm_unreachable("unexpected expr in string literal init");
  }
}

static void CheckStringInit(Expr *Str, QualType &DeclT, const ArrayType *AT,
                            Sema &S) {
  // Get the length of the string as parsed.
  auto *ConstantArrayTy =
      cast<ConstantArrayType>(Str->getType()->getAsArrayTypeUnsafe());
  uint64_t StrLength = ConstantArrayTy->getSize().getZExtValue();

  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT)) {
    // C99 6.7.8p14. We have an array of character type with unknown size
    // being initialized to a string literal.
    llvm::APInt ConstVal(32, StrLength);
    // Return a new array type (C99 6.7.8p22).
    DeclT = S.Context.getConstantArrayType(IAT->getElementType(),
                                           ConstVal,
                                           ArrayType::Normal, 0);
    updateStringLiteralType(Str, DeclT);
    return;
  }

  const ConstantArrayType *CAT = cast<ConstantArrayType>(AT);

  // We have an array of character type with known size.  However,
  // the size may be smaller or larger than the string we are initializing.
  // FIXME: Avoid truncation for 64-bit length strings.
  if (S.getLangOpts().CPlusPlus) {
    if (StringLiteral *SL = dyn_cast<StringLiteral>(Str->IgnoreParens())) {
      // For Pascal strings it's OK to strip off the terminating null character,
      // so the example below is valid:
      //
      // unsigned char a[2] = "\pa";
      if (SL->isPascal())
        StrLength--;
    }
  
    // [dcl.init.string]p2
    if (StrLength > CAT->getSize().getZExtValue())
      S.Diag(Str->getLocStart(),
             diag::err_initializer_string_for_char_array_too_long)
        << Str->getSourceRange();
  } else {
    // C99 6.7.8p14.
    if (StrLength-1 > CAT->getSize().getZExtValue())
      S.Diag(Str->getLocStart(),
             diag::ext_initializer_string_for_char_array_too_long)
        << Str->getSourceRange();
  }

  // Set the type to the actual size that we are initializing.  If we have
  // something like:
  //   char x[1] = "foo";
  // then this will set the string literal's type to char[1].
  updateStringLiteralType(Str, DeclT);
}

//===----------------------------------------------------------------------===//
// Semantic checking for initializer lists.
//===----------------------------------------------------------------------===//

/// @brief Semantic checking for initializer lists.
///
/// The InitListChecker class contains a set of routines that each
/// handle the initialization of a certain kind of entity, e.g.,
/// arrays, vectors, struct/union types, scalars, etc. The
/// InitListChecker itself performs a recursive walk of the subobject
/// structure of the type to be initialized, while stepping through
/// the initializer list one element at a time. The IList and Index
/// parameters to each of the Check* routines contain the active
/// (syntactic) initializer list and the index into that initializer
/// list that represents the current initializer. Each routine is
/// responsible for moving that Index forward as it consumes elements.
///
/// Each Check* routine also has a StructuredList/StructuredIndex
/// arguments, which contains the current "structured" (semantic)
/// initializer list and the index into that initializer list where we
/// are copying initializers as we map them over to the semantic
/// list. Once we have completed our recursive walk of the subobject
/// structure, we will have constructed a full semantic initializer
/// list.
///
/// C99 designators cause changes in the initializer list traversal,
/// because they make the initialization "jump" into a specific
/// subobject and then continue the initialization from that
/// point. CheckDesignatedInitializer() recursively steps into the
/// designated subobject and manages backing out the recursion to
/// initialize the subobjects after the one designated.
namespace {
class InitListChecker {
  Sema &SemaRef;
  bool hadError;
  bool VerifyOnly; // no diagnostics, no structure building
  llvm::DenseMap<InitListExpr *, InitListExpr *> SyntacticToSemantic;
  InitListExpr *FullyStructuredList;

  void CheckImplicitInitList(const InitializedEntity &Entity,
                             InitListExpr *ParentIList, QualType T,
                             unsigned &Index, InitListExpr *StructuredList,
                             unsigned &StructuredIndex);
  void CheckExplicitInitList(const InitializedEntity &Entity,
                             InitListExpr *IList, QualType &T,
                             InitListExpr *StructuredList,
                             bool TopLevelObject = false);
  void CheckListElementTypes(const InitializedEntity &Entity,
                             InitListExpr *IList, QualType &DeclType,
                             bool SubobjectIsDesignatorContext,
                             unsigned &Index,
                             InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckSubElementType(const InitializedEntity &Entity,
                           InitListExpr *IList, QualType ElemType,
                           unsigned &Index,
                           InitListExpr *StructuredList,
                           unsigned &StructuredIndex);
  void CheckComplexType(const InitializedEntity &Entity,
                        InitListExpr *IList, QualType DeclType,
                        unsigned &Index,
                        InitListExpr *StructuredList,
                        unsigned &StructuredIndex);
  void CheckScalarType(const InitializedEntity &Entity,
                       InitListExpr *IList, QualType DeclType,
                       unsigned &Index,
                       InitListExpr *StructuredList,
                       unsigned &StructuredIndex);
  void CheckReferenceType(const InitializedEntity &Entity,
                          InitListExpr *IList, QualType DeclType,
                          unsigned &Index,
                          InitListExpr *StructuredList,
                          unsigned &StructuredIndex);
  void CheckVectorType(const InitializedEntity &Entity,
                       InitListExpr *IList, QualType DeclType, unsigned &Index,
                       InitListExpr *StructuredList,
                       unsigned &StructuredIndex);
  void CheckStructUnionTypes(const InitializedEntity &Entity,
                             InitListExpr *IList, QualType DeclType,
                             RecordDecl::field_iterator Field,
                             bool SubobjectIsDesignatorContext, unsigned &Index,
                             InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckArrayType(const InitializedEntity &Entity,
                      InitListExpr *IList, QualType &DeclType,
                      llvm::APSInt elementIndex,
                      bool SubobjectIsDesignatorContext, unsigned &Index,
                      InitListExpr *StructuredList,
                      unsigned &StructuredIndex);
  bool CheckDesignatedInitializer(const InitializedEntity &Entity,
                                  InitListExpr *IList, DesignatedInitExpr *DIE,
                                  unsigned DesigIdx,
                                  QualType &CurrentObjectType,
                                  RecordDecl::field_iterator *NextField,
                                  llvm::APSInt *NextElementIndex,
                                  unsigned &Index,
                                  InitListExpr *StructuredList,
                                  unsigned &StructuredIndex,
                                  bool FinishSubobjectInit,
                                  bool TopLevelObject);
  InitListExpr *getStructuredSubobjectInit(InitListExpr *IList, unsigned Index,
                                           QualType CurrentObjectType,
                                           InitListExpr *StructuredList,
                                           unsigned StructuredIndex,
                                           SourceRange InitRange,
                                           bool IsFullyOverwritten = false);
  void UpdateStructuredListElement(InitListExpr *StructuredList,
                                   unsigned &StructuredIndex,
                                   Expr *expr);
  int numArrayElements(QualType DeclType);
  int numStructUnionElements(QualType DeclType);

  static ExprResult PerformEmptyInit(Sema &SemaRef,
                                     SourceLocation Loc,
                                     const InitializedEntity &Entity,
                                     bool VerifyOnly);

  // Explanation on the "FillWithNoInit" mode:
  //
  // Assume we have the following definitions (Case#1):
  // struct P { char x[6][6]; } xp = { .x[1] = "bar" };
  // struct PP { struct P lp; } l = { .lp = xp, .lp.x[1][2] = 'f' };
  //
  // l.lp.x[1][0..1] should not be filled with implicit initializers because the
  // "base" initializer "xp" will provide values for them; l.lp.x[1] will be "baf".
  //
  // But if we have (Case#2):
  // struct PP l = { .lp = xp, .lp.x[1] = { [2] = 'f' } };
  //
  // l.lp.x[1][0..1] are implicitly initialized and do not use values from the
  // "base" initializer; l.lp.x[1] will be "\0\0f\0\0\0".
  //
  // To distinguish Case#1 from Case#2, and also to avoid leaving many "holes"
  // in the InitListExpr, the "holes" in Case#1 are filled not with empty
  // initializers but with special "NoInitExpr" place holders, which tells the
  // CodeGen not to generate any initializers for these parts.
  void FillInEmptyInitForField(unsigned Init, FieldDecl *Field,
                               const InitializedEntity &ParentEntity,
                               InitListExpr *ILE, bool &RequiresSecondPass,
                               bool FillWithNoInit = false);
  void FillInEmptyInitializations(const InitializedEntity &Entity,
                                  InitListExpr *ILE, bool &RequiresSecondPass,
                                  bool FillWithNoInit = false);
  bool CheckFlexibleArrayInit(const InitializedEntity &Entity,
                              Expr *InitExpr, FieldDecl *Field,
                              bool TopLevelObject);
  void CheckEmptyInitializable(const InitializedEntity &Entity,
                               SourceLocation Loc);

public:
  InitListChecker(Sema &S, const InitializedEntity &Entity,
                  InitListExpr *IL, QualType &T, bool VerifyOnly);
  bool HadError() { return hadError; }

  // @brief Retrieves the fully-structured initializer list used for
  // semantic analysis and code generation.
  InitListExpr *getFullyStructuredList() const { return FullyStructuredList; }
};
} // end anonymous namespace

ExprResult InitListChecker::PerformEmptyInit(Sema &SemaRef,
                                             SourceLocation Loc,
                                             const InitializedEntity &Entity,
                                             bool VerifyOnly) {
  InitializationKind Kind = InitializationKind::CreateValue(Loc, Loc, Loc,
                                                            true);
  MultiExprArg SubInit;
  Expr *InitExpr;
  InitListExpr DummyInitList(SemaRef.Context, Loc, None, Loc);

  // C++ [dcl.init.aggr]p7:
  //   If there are fewer initializer-clauses in the list than there are
  //   members in the aggregate, then each member not explicitly initialized
  //   ...
  bool EmptyInitList = SemaRef.getLangOpts().CPlusPlus11 &&
      Entity.getType()->getBaseElementTypeUnsafe()->isRecordType();
  if (EmptyInitList) {
    // C++1y / DR1070:
    //   shall be initialized [...] from an empty initializer list.
    //
    // We apply the resolution of this DR to C++11 but not C++98, since C++98
    // does not have useful semantics for initialization from an init list.
    // We treat this as copy-initialization, because aggregate initialization
    // always performs copy-initialization on its elements.
    //
    // Only do this if we're initializing a class type, to avoid filling in
    // the initializer list where possible.
    InitExpr = VerifyOnly ? &DummyInitList : new (SemaRef.Context)
                   InitListExpr(SemaRef.Context, Loc, None, Loc);
    InitExpr->setType(SemaRef.Context.VoidTy);
    SubInit = InitExpr;
    Kind = InitializationKind::CreateCopy(Loc, Loc);
  } else {
    // C++03:
    //   shall be value-initialized.
  }

  InitializationSequence InitSeq(SemaRef, Entity, Kind, SubInit);
  // libstdc++4.6 marks the vector default constructor as explicit in
  // _GLIBCXX_DEBUG mode, so recover using the C++03 logic in that case.
  // stlport does so too. Look for std::__debug for libstdc++, and for
  // std:: for stlport.  This is effectively a compiler-side implementation of
  // LWG2193.
  if (!InitSeq && EmptyInitList && InitSeq.getFailureKind() ==
          InitializationSequence::FK_ExplicitConstructor) {
    OverloadCandidateSet::iterator Best;
    OverloadingResult O =
        InitSeq.getFailedCandidateSet()
            .BestViableFunction(SemaRef, Kind.getLocation(), Best);
    (void)O;
    assert(O == OR_Success && "Inconsistent overload resolution");
    CXXConstructorDecl *CtorDecl = cast<CXXConstructorDecl>(Best->Function);
    CXXRecordDecl *R = CtorDecl->getParent();

    if (CtorDecl->getMinRequiredArguments() == 0 &&
        CtorDecl->isExplicit() && R->getDeclName() &&
        SemaRef.SourceMgr.isInSystemHeader(CtorDecl->getLocation())) {


      bool IsInStd = false;
      for (NamespaceDecl *ND = dyn_cast<NamespaceDecl>(R->getDeclContext());
           ND && !IsInStd; ND = dyn_cast<NamespaceDecl>(ND->getParent())) {
        if (SemaRef.getStdNamespace()->InEnclosingNamespaceSetOf(ND))
          IsInStd = true;
      }

      if (IsInStd && llvm::StringSwitch<bool>(R->getName()) 
              .Cases("basic_string", "deque", "forward_list", true)
              .Cases("list", "map", "multimap", "multiset", true)
              .Cases("priority_queue", "queue", "set", "stack", true)
              .Cases("unordered_map", "unordered_set", "vector", true)
              .Default(false)) {
        InitSeq.InitializeFrom(
            SemaRef, Entity,
            InitializationKind::CreateValue(Loc, Loc, Loc, true),
            MultiExprArg(), /*TopLevelOfInitList=*/false);
        // Emit a warning for this.  System header warnings aren't shown
        // by default, but people working on system headers should see it.
        if (!VerifyOnly) {
          SemaRef.Diag(CtorDecl->getLocation(),
                       diag::warn_invalid_initializer_from_system_header);
          if (Entity.getKind() == InitializedEntity::EK_Member)
            SemaRef.Diag(Entity.getDecl()->getLocation(),
                         diag::note_used_in_initialization_here);
          else if (Entity.getKind() == InitializedEntity::EK_ArrayElement)
            SemaRef.Diag(Loc, diag::note_used_in_initialization_here);
        }
      }
    }
  }
  if (!InitSeq) {
    if (!VerifyOnly) {
      InitSeq.Diagnose(SemaRef, Entity, Kind, SubInit);
      if (Entity.getKind() == InitializedEntity::EK_Member)
        SemaRef.Diag(Entity.getDecl()->getLocation(),
                     diag::note_in_omitted_aggregate_initializer)
          << /*field*/1 << Entity.getDecl();
      else if (Entity.getKind() == InitializedEntity::EK_ArrayElement)
        SemaRef.Diag(Loc, diag::note_in_omitted_aggregate_initializer)
          << /*array element*/0 << Entity.getElementIndex();
    }
    return ExprError();
  }

  return VerifyOnly ? ExprResult(static_cast<Expr *>(nullptr))
                    : InitSeq.Perform(SemaRef, Entity, Kind, SubInit);
}

void InitListChecker::CheckEmptyInitializable(const InitializedEntity &Entity,
                                              SourceLocation Loc) {
  assert(VerifyOnly &&
         "CheckEmptyInitializable is only inteded for verification mode.");
  if (PerformEmptyInit(SemaRef, Loc, Entity, /*VerifyOnly*/true).isInvalid())
    hadError = true;
}

void InitListChecker::FillInEmptyInitForField(unsigned Init, FieldDecl *Field,
                                        const InitializedEntity &ParentEntity,
                                              InitListExpr *ILE,
                                              bool &RequiresSecondPass,
                                              bool FillWithNoInit) {
  SourceLocation Loc = ILE->getLocEnd();
  unsigned NumInits = ILE->getNumInits();
  InitializedEntity MemberEntity
    = InitializedEntity::InitializeMember(Field, &ParentEntity);

  if (const RecordType *RType = ILE->getType()->getAs<RecordType>())
    if (!RType->getDecl()->isUnion())
      assert(Init < NumInits && "This ILE should have been expanded");

  if (Init >= NumInits || !ILE->getInit(Init)) {
    if (FillWithNoInit) {
      Expr *Filler = new (SemaRef.Context) NoInitExpr(Field->getType());
      if (Init < NumInits)
        ILE->setInit(Init, Filler);
      else
        ILE->updateInit(SemaRef.Context, Init, Filler);
      return;
    }
    // C++1y [dcl.init.aggr]p7:
    //   If there are fewer initializer-clauses in the list than there are
    //   members in the aggregate, then each member not explicitly initialized
    //   shall be initialized from its brace-or-equal-initializer [...]
    if (Field->hasInClassInitializer()) {
      ExprResult DIE = SemaRef.BuildCXXDefaultInitExpr(Loc, Field);
      if (DIE.isInvalid()) {
        hadError = true;
        return;
      }
      if (Init < NumInits)
        ILE->setInit(Init, DIE.get());
      else {
        ILE->updateInit(SemaRef.Context, Init, DIE.get());
        RequiresSecondPass = true;
      }
      return;
    }

    if (Field->getType()->isReferenceType()) {
      // C++ [dcl.init.aggr]p9:
      //   If an incomplete or empty initializer-list leaves a
      //   member of reference type uninitialized, the program is
      //   ill-formed.
      SemaRef.Diag(Loc, diag::err_init_reference_member_uninitialized)
        << Field->getType()
        << ILE->getSyntacticForm()->getSourceRange();
      SemaRef.Diag(Field->getLocation(),
                   diag::note_uninit_reference_member);
      hadError = true;
      return;
    }

    ExprResult MemberInit = PerformEmptyInit(SemaRef, Loc, MemberEntity,
                                             /*VerifyOnly*/false);
    if (MemberInit.isInvalid()) {
      hadError = true;
      return;
    }

    if (hadError) {
      // Do nothing
    } else if (Init < NumInits) {
      ILE->setInit(Init, MemberInit.getAs<Expr>());
    } else if (!isa<ImplicitValueInitExpr>(MemberInit.get())) {
      // Empty initialization requires a constructor call, so
      // extend the initializer list to include the constructor
      // call and make a note that we'll need to take another pass
      // through the initializer list.
      ILE->updateInit(SemaRef.Context, Init, MemberInit.getAs<Expr>());
      RequiresSecondPass = true;
    }
  } else if (InitListExpr *InnerILE
               = dyn_cast<InitListExpr>(ILE->getInit(Init)))
    FillInEmptyInitializations(MemberEntity, InnerILE,
                               RequiresSecondPass, FillWithNoInit);
  else if (DesignatedInitUpdateExpr *InnerDIUE
               = dyn_cast<DesignatedInitUpdateExpr>(ILE->getInit(Init)))
    FillInEmptyInitializations(MemberEntity, InnerDIUE->getUpdater(),
                               RequiresSecondPass, /*FillWithNoInit =*/ true);
}

/// Recursively replaces NULL values within the given initializer list
/// with expressions that perform value-initialization of the
/// appropriate type.
void
InitListChecker::FillInEmptyInitializations(const InitializedEntity &Entity,
                                            InitListExpr *ILE,
                                            bool &RequiresSecondPass,
                                            bool FillWithNoInit) {
  assert((ILE->getType() != SemaRef.Context.VoidTy) &&
         "Should not have void type");

  if (const RecordType *RType = ILE->getType()->getAs<RecordType>()) {
    const RecordDecl *RDecl = RType->getDecl();
    if (RDecl->isUnion() && ILE->getInitializedFieldInUnion())
      FillInEmptyInitForField(0, ILE->getInitializedFieldInUnion(),
                              Entity, ILE, RequiresSecondPass, FillWithNoInit);
    else if (RDecl->isUnion() && isa<CXXRecordDecl>(RDecl) &&
             cast<CXXRecordDecl>(RDecl)->hasInClassInitializer()) {
      for (auto *Field : RDecl->fields()) {
        if (Field->hasInClassInitializer()) {
          FillInEmptyInitForField(0, Field, Entity, ILE, RequiresSecondPass,
                                  FillWithNoInit);
          break;
        }
      }
    } else {
      // The fields beyond ILE->getNumInits() are default initialized, so in
      // order to leave them uninitialized, the ILE is expanded and the extra
      // fields are then filled with NoInitExpr.
      unsigned NumFields = 0;
      for (auto *Field : RDecl->fields())
        if (!Field->isUnnamedBitfield())
          ++NumFields;
      if (ILE->getNumInits() < NumFields)
        ILE->resizeInits(SemaRef.Context, NumFields);

      unsigned Init = 0;
      for (auto *Field : RDecl->fields()) {
        if (Field->isUnnamedBitfield())
          continue;

        if (hadError)
          return;

        FillInEmptyInitForField(Init, Field, Entity, ILE, RequiresSecondPass,
                                FillWithNoInit);
        if (hadError)
          return;

        ++Init;

        // Only look at the first initialization of a union.
        if (RDecl->isUnion())
          break;
      }
    }

    return;
  }

  QualType ElementType;

  InitializedEntity ElementEntity = Entity;
  unsigned NumInits = ILE->getNumInits();
  unsigned NumElements = NumInits;
  if (const ArrayType *AType = SemaRef.Context.getAsArrayType(ILE->getType())) {
    ElementType = AType->getElementType();
    if (const ConstantArrayType *CAType = dyn_cast<ConstantArrayType>(AType))
      NumElements = CAType->getSize().getZExtValue();
    ElementEntity = InitializedEntity::InitializeElement(SemaRef.Context,
                                                         0, Entity);
  } else if (const VectorType *VType = ILE->getType()->getAs<VectorType>()) {
    ElementType = VType->getElementType();
    NumElements = VType->getNumElements();
    ElementEntity = InitializedEntity::InitializeElement(SemaRef.Context,
                                                         0, Entity);
  } else
    ElementType = ILE->getType();

  for (unsigned Init = 0; Init != NumElements; ++Init) {
    if (hadError)
      return;

    if (ElementEntity.getKind() == InitializedEntity::EK_ArrayElement ||
        ElementEntity.getKind() == InitializedEntity::EK_VectorElement)
      ElementEntity.setElementIndex(Init);

    Expr *InitExpr = (Init < NumInits ? ILE->getInit(Init) : nullptr);
    if (!InitExpr && Init < NumInits && ILE->hasArrayFiller())
      ILE->setInit(Init, ILE->getArrayFiller());
    else if (!InitExpr && !ILE->hasArrayFiller()) {
      Expr *Filler = nullptr;

      if (FillWithNoInit)
        Filler = new (SemaRef.Context) NoInitExpr(ElementType);
      else {
        ExprResult ElementInit = PerformEmptyInit(SemaRef, ILE->getLocEnd(),
                                                  ElementEntity,
                                                  /*VerifyOnly*/false);
        if (ElementInit.isInvalid()) {
          hadError = true;
          return;
        }

        Filler = ElementInit.getAs<Expr>();
      }

      if (hadError) {
        // Do nothing
      } else if (Init < NumInits) {
        // For arrays, just set the expression used for value-initialization
        // of the "holes" in the array.
        if (ElementEntity.getKind() == InitializedEntity::EK_ArrayElement)
          ILE->setArrayFiller(Filler);
        else
          ILE->setInit(Init, Filler);
      } else {
        // For arrays, just set the expression used for value-initialization
        // of the rest of elements and exit.
        if (ElementEntity.getKind() == InitializedEntity::EK_ArrayElement) {
          ILE->setArrayFiller(Filler);
          return;
        }

        if (!isa<ImplicitValueInitExpr>(Filler) && !isa<NoInitExpr>(Filler)) {
          // Empty initialization requires a constructor call, so
          // extend the initializer list to include the constructor
          // call and make a note that we'll need to take another pass
          // through the initializer list.
          ILE->updateInit(SemaRef.Context, Init, Filler);
          RequiresSecondPass = true;
        }
      }
    } else if (InitListExpr *InnerILE
                 = dyn_cast_or_null<InitListExpr>(InitExpr))
      FillInEmptyInitializations(ElementEntity, InnerILE, RequiresSecondPass,
                                 FillWithNoInit);
    else if (DesignatedInitUpdateExpr *InnerDIUE
                 = dyn_cast_or_null<DesignatedInitUpdateExpr>(InitExpr))
      FillInEmptyInitializations(ElementEntity, InnerDIUE->getUpdater(),
                                 RequiresSecondPass, /*FillWithNoInit =*/ true);
  }
}


InitListChecker::InitListChecker(Sema &S, const InitializedEntity &Entity,
                                 InitListExpr *IL, QualType &T,
                                 bool VerifyOnly)
  : SemaRef(S), VerifyOnly(VerifyOnly) {
  // FIXME: Check that IL isn't already the semantic form of some other
  // InitListExpr. If it is, we'd create a broken AST.

  hadError = false;

  FullyStructuredList =
      getStructuredSubobjectInit(IL, 0, T, nullptr, 0, IL->getSourceRange());
  CheckExplicitInitList(Entity, IL, T, FullyStructuredList,
                        /*TopLevelObject=*/true);

  if (!hadError && !VerifyOnly) {
    bool RequiresSecondPass = false;
    FillInEmptyInitializations(Entity, FullyStructuredList, RequiresSecondPass);
    if (RequiresSecondPass && !hadError)
      FillInEmptyInitializations(Entity, FullyStructuredList,
                                 RequiresSecondPass);
  }
}

int InitListChecker::numArrayElements(QualType DeclType) {
  // FIXME: use a proper constant
  int maxElements = 0x7FFFFFFF;
  if (const ConstantArrayType *CAT =
        SemaRef.Context.getAsConstantArrayType(DeclType)) {
    maxElements = static_cast<int>(CAT->getSize().getZExtValue());
  }
  return maxElements;
}

int InitListChecker::numStructUnionElements(QualType DeclType) {
  RecordDecl *structDecl = DeclType->getAs<RecordType>()->getDecl();
  int InitializableMembers = 0;
  for (const auto *Field : structDecl->fields())
    if (!Field->isUnnamedBitfield())
      ++InitializableMembers;

  if (structDecl->isUnion())
    return std::min(InitializableMembers, 1);
  return InitializableMembers - structDecl->hasFlexibleArrayMember();
}

/// Check whether the range of the initializer \p ParentIList from element
/// \p Index onwards can be used to initialize an object of type \p T. Update
/// \p Index to indicate how many elements of the list were consumed.
///
/// This also fills in \p StructuredList, from element \p StructuredIndex
/// onwards, with the fully-braced, desugared form of the initialization.
void InitListChecker::CheckImplicitInitList(const InitializedEntity &Entity,
                                            InitListExpr *ParentIList,
                                            QualType T, unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex) {
  int maxElements = 0;

  if (T->isArrayType())
    maxElements = numArrayElements(T);
  else if (T->isRecordType())
    maxElements = numStructUnionElements(T);
  else if (T->isVectorType())
    maxElements = T->getAs<VectorType>()->getNumElements();
  else
    llvm_unreachable("CheckImplicitInitList(): Illegal type");

  if (maxElements == 0) {
    if (!VerifyOnly)
      SemaRef.Diag(ParentIList->getInit(Index)->getLocStart(),
                   diag::err_implicit_empty_initializer);
    ++Index;
    hadError = true;
    return;
  }

  // Build a structured initializer list corresponding to this subobject.
  InitListExpr *StructuredSubobjectInitList
    = getStructuredSubobjectInit(ParentIList, Index, T, StructuredList,
                                 StructuredIndex,
          SourceRange(ParentIList->getInit(Index)->getLocStart(),
                      ParentIList->getSourceRange().getEnd()));
  unsigned StructuredSubobjectInitIndex = 0;

  // Check the element types and build the structural subobject.
  unsigned StartIndex = Index;
  CheckListElementTypes(Entity, ParentIList, T,
                        /*SubobjectIsDesignatorContext=*/false, Index,
                        StructuredSubobjectInitList,
                        StructuredSubobjectInitIndex);

  if (!VerifyOnly) {
    StructuredSubobjectInitList->setType(T);

    unsigned EndIndex = (Index == StartIndex? StartIndex : Index - 1);
    // Update the structured sub-object initializer so that it's ending
    // range corresponds with the end of the last initializer it used.
    if (EndIndex < ParentIList->getNumInits() &&
        ParentIList->getInit(EndIndex)) {
      SourceLocation EndLoc
        = ParentIList->getInit(EndIndex)->getSourceRange().getEnd();
      StructuredSubobjectInitList->setRBraceLoc(EndLoc);
    }

    // Complain about missing braces.
    if (T->isArrayType() || T->isRecordType()) {
      SemaRef.Diag(StructuredSubobjectInitList->getLocStart(),
                   diag::warn_missing_braces)
          << StructuredSubobjectInitList->getSourceRange()
          << FixItHint::CreateInsertion(
                 StructuredSubobjectInitList->getLocStart(), "{")
          << FixItHint::CreateInsertion(
                 SemaRef.getLocForEndOfToken(
                     StructuredSubobjectInitList->getLocEnd()),
                 "}");
    }
  }
}

/// Warn that \p Entity was of scalar type and was initialized by a
/// single-element braced initializer list.
static void warnBracedScalarInit(Sema &S, const InitializedEntity &Entity,
                                 SourceRange Braces) {
  // Don't warn during template instantiation. If the initialization was
  // non-dependent, we warned during the initial parse; otherwise, the
  // type might not be scalar in some uses of the template.
  if (!S.ActiveTemplateInstantiations.empty())
    return;

  unsigned DiagID = 0;

  switch (Entity.getKind()) {
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_ComplexElement:
  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Parameter_CF_Audited:
  case InitializedEntity::EK_Result:
    // Extra braces here are suspicious.
    DiagID = diag::warn_braces_around_scalar_init;
    break;

  case InitializedEntity::EK_Member:
    // Warn on aggregate initialization but not on ctor init list or
    // default member initializer.
    if (Entity.getParent())
      DiagID = diag::warn_braces_around_scalar_init;
    break;

  case InitializedEntity::EK_Variable:
  case InitializedEntity::EK_LambdaCapture:
    // No warning, might be direct-list-initialization.
    // FIXME: Should we warn for copy-list-initialization in these cases?
    break;

  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_CompoundLiteralInit:
    // No warning, braces are part of the syntax of the underlying construct.
    break;

  case InitializedEntity::EK_RelatedResult:
    // No warning, we already warned when initializing the result.
    break;

  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_Delegating:
  case InitializedEntity::EK_BlockElement:
    llvm_unreachable("unexpected braced scalar init");
  }

  if (DiagID) {
    S.Diag(Braces.getBegin(), DiagID)
      << Braces
      << FixItHint::CreateRemoval(Braces.getBegin())
      << FixItHint::CreateRemoval(Braces.getEnd());
  }
}


/// Check whether the initializer \p IList (that was written with explicit
/// braces) can be used to initialize an object of type \p T.
///
/// This also fills in \p StructuredList with the fully-braced, desugared
/// form of the initialization.
void InitListChecker::CheckExplicitInitList(const InitializedEntity &Entity,
                                            InitListExpr *IList, QualType &T,
                                            InitListExpr *StructuredList,
                                            bool TopLevelObject) {
  if (!VerifyOnly) {
    SyntacticToSemantic[IList] = StructuredList;
    StructuredList->setSyntacticForm(IList);
  }

  unsigned Index = 0, StructuredIndex = 0;
  CheckListElementTypes(Entity, IList, T, /*SubobjectIsDesignatorContext=*/true,
                        Index, StructuredList, StructuredIndex, TopLevelObject);
  if (!VerifyOnly) {
    QualType ExprTy = T;
    if (!ExprTy->isArrayType())
      ExprTy = ExprTy.getNonLValueExprType(SemaRef.Context);
    IList->setType(ExprTy);
    StructuredList->setType(ExprTy);
  }
  if (hadError)
    return;

  if (Index < IList->getNumInits()) {
    // We have leftover initializers
    if (VerifyOnly) {
      if (SemaRef.getLangOpts().CPlusPlus ||
          (SemaRef.getLangOpts().OpenCL &&
           IList->getType()->isVectorType())) {
        hadError = true;
      }
      return;
    }

    if (StructuredIndex == 1 &&
        IsStringInit(StructuredList->getInit(0), T, SemaRef.Context) ==
            SIF_None) {
      unsigned DK = diag::ext_excess_initializers_in_char_array_initializer;
      if (SemaRef.getLangOpts().CPlusPlus) {
        DK = diag::err_excess_initializers_in_char_array_initializer;
        hadError = true;
      }
      // Special-case
      SemaRef.Diag(IList->getInit(Index)->getLocStart(), DK)
        << IList->getInit(Index)->getSourceRange();
    } else if (!T->isIncompleteType()) {
      // Don't complain for incomplete types, since we'll get an error
      // elsewhere
      QualType CurrentObjectType = StructuredList->getType();
      int initKind =
        CurrentObjectType->isArrayType()? 0 :
        CurrentObjectType->isVectorType()? 1 :
        CurrentObjectType->isScalarType()? 2 :
        CurrentObjectType->isUnionType()? 3 :
        4;

      unsigned DK = diag::ext_excess_initializers;
      if (SemaRef.getLangOpts().CPlusPlus) {
        DK = diag::err_excess_initializers;
        hadError = true;
      }
      if (SemaRef.getLangOpts().OpenCL && initKind == 1) {
        DK = diag::err_excess_initializers;
        hadError = true;
      }

      SemaRef.Diag(IList->getInit(Index)->getLocStart(), DK)
        << initKind << IList->getInit(Index)->getSourceRange();
    }
  }

  if (!VerifyOnly && T->isScalarType() &&
      IList->getNumInits() == 1 && !isa<InitListExpr>(IList->getInit(0)))
    warnBracedScalarInit(SemaRef, Entity, IList->getSourceRange());
}

void InitListChecker::CheckListElementTypes(const InitializedEntity &Entity,
                                            InitListExpr *IList,
                                            QualType &DeclType,
                                            bool SubobjectIsDesignatorContext,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  if (DeclType->isAnyComplexType() && SubobjectIsDesignatorContext) {
    // Explicitly braced initializer for complex type can be real+imaginary
    // parts.
    CheckComplexType(Entity, IList, DeclType, Index,
                     StructuredList, StructuredIndex);
  } else if (DeclType->isScalarType()) {
    CheckScalarType(Entity, IList, DeclType, Index,
                    StructuredList, StructuredIndex);
  } else if (DeclType->isVectorType()) {
    CheckVectorType(Entity, IList, DeclType, Index,
                    StructuredList, StructuredIndex);
  } else if (DeclType->isRecordType()) {
    assert(DeclType->isAggregateType() &&
           "non-aggregate records should be handed in CheckSubElementType");
    RecordDecl *RD = DeclType->getAs<RecordType>()->getDecl();
    CheckStructUnionTypes(Entity, IList, DeclType, RD->field_begin(),
                          SubobjectIsDesignatorContext, Index,
                          StructuredList, StructuredIndex,
                          TopLevelObject);
  } else if (DeclType->isArrayType()) {
    llvm::APSInt Zero(
                    SemaRef.Context.getTypeSize(SemaRef.Context.getSizeType()),
                    false);
    CheckArrayType(Entity, IList, DeclType, Zero,
                   SubobjectIsDesignatorContext, Index,
                   StructuredList, StructuredIndex);
  } else if (DeclType->isVoidType() || DeclType->isFunctionType()) {
    // This type is invalid, issue a diagnostic.
    ++Index;
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(), diag::err_illegal_initializer_type)
        << DeclType;
    hadError = true;
  } else if (DeclType->isReferenceType()) {
    CheckReferenceType(Entity, IList, DeclType, Index,
                       StructuredList, StructuredIndex);
  } else if (DeclType->isObjCObjectType()) {
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(), diag::err_init_objc_class)
        << DeclType;
    hadError = true;
  } else {
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(), diag::err_illegal_initializer_type)
        << DeclType;
    hadError = true;
  }
}

void InitListChecker::CheckSubElementType(const InitializedEntity &Entity,
                                          InitListExpr *IList,
                                          QualType ElemType,
                                          unsigned &Index,
                                          InitListExpr *StructuredList,
                                          unsigned &StructuredIndex) {
  Expr *expr = IList->getInit(Index);

  if (ElemType->isReferenceType())
    return CheckReferenceType(Entity, IList, ElemType, Index,
                              StructuredList, StructuredIndex);

  if (InitListExpr *SubInitList = dyn_cast<InitListExpr>(expr)) {
    if (SubInitList->getNumInits() == 1 &&
        IsStringInit(SubInitList->getInit(0), ElemType, SemaRef.Context) ==
        SIF_None) {
      expr = SubInitList->getInit(0);
    } else if (!SemaRef.getLangOpts().CPlusPlus) {
      InitListExpr *InnerStructuredList
        = getStructuredSubobjectInit(IList, Index, ElemType,
                                     StructuredList, StructuredIndex,
                                     SubInitList->getSourceRange(), true);
      CheckExplicitInitList(Entity, SubInitList, ElemType,
                            InnerStructuredList);

      if (!hadError && !VerifyOnly) {
        bool RequiresSecondPass = false;
        FillInEmptyInitializations(Entity, InnerStructuredList,
                                   RequiresSecondPass);
        if (RequiresSecondPass && !hadError)
          FillInEmptyInitializations(Entity, InnerStructuredList,
                                     RequiresSecondPass);
      }
      ++StructuredIndex;
      ++Index;
      return;
    }
    // C++ initialization is handled later.
  } else if (isa<ImplicitValueInitExpr>(expr)) {
    // This happens during template instantiation when we see an InitListExpr
    // that we've already checked once.
    assert(SemaRef.Context.hasSameType(expr->getType(), ElemType) &&
           "found implicit initialization for the wrong type");
    if (!VerifyOnly)
      UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
    ++Index;
    return;
  }

  if (SemaRef.getLangOpts().CPlusPlus) {
    // C++ [dcl.init.aggr]p2:
    //   Each member is copy-initialized from the corresponding
    //   initializer-clause.

    // FIXME: Better EqualLoc?
    InitializationKind Kind =
      InitializationKind::CreateCopy(expr->getLocStart(), SourceLocation());
    InitializationSequence Seq(SemaRef, Entity, Kind, expr,
                               /*TopLevelOfInitList*/ true);

    // C++14 [dcl.init.aggr]p13:
    //   If the assignment-expression can initialize a member, the member is
    //   initialized. Otherwise [...] brace elision is assumed
    //
    // Brace elision is never performed if the element is not an
    // assignment-expression.
    if (Seq || isa<InitListExpr>(expr)) {
      if (!VerifyOnly) {
        ExprResult Result =
          Seq.Perform(SemaRef, Entity, Kind, expr);
        if (Result.isInvalid())
          hadError = true;

        UpdateStructuredListElement(StructuredList, StructuredIndex,
                                    Result.getAs<Expr>());
      } else if (!Seq)
        hadError = true;
      ++Index;
      return;
    }

    // Fall through for subaggregate initialization
  } else if (ElemType->isScalarType() || ElemType->isAtomicType()) {
    // FIXME: Need to handle atomic aggregate types with implicit init lists.
    return CheckScalarType(Entity, IList, ElemType, Index,
                           StructuredList, StructuredIndex);
  } else if (const ArrayType *arrayType =
                 SemaRef.Context.getAsArrayType(ElemType)) {
    // arrayType can be incomplete if we're initializing a flexible
    // array member.  There's nothing we can do with the completed
    // type here, though.

    if (IsStringInit(expr, arrayType, SemaRef.Context) == SIF_None) {
      if (!VerifyOnly) {
        CheckStringInit(expr, ElemType, arrayType, SemaRef);
        UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
      }
      ++Index;
      return;
    }

    // Fall through for subaggregate initialization.

  } else {
    assert((ElemType->isRecordType() || ElemType->isVectorType()) &&
           "Unexpected type");

    // C99 6.7.8p13:
    //
    //   The initializer for a structure or union object that has
    //   automatic storage duration shall be either an initializer
    //   list as described below, or a single expression that has
    //   compatible structure or union type. In the latter case, the
    //   initial value of the object, including unnamed members, is
    //   that of the expression.
    ExprResult ExprRes = expr;
    if (SemaRef.CheckSingleAssignmentConstraints(
            ElemType, ExprRes, !VerifyOnly) != Sema::Incompatible) {
      if (ExprRes.isInvalid())
        hadError = true;
      else {
        ExprRes = SemaRef.DefaultFunctionArrayLvalueConversion(ExprRes.get());
          if (ExprRes.isInvalid())
            hadError = true;
      }
      UpdateStructuredListElement(StructuredList, StructuredIndex,
                                  ExprRes.getAs<Expr>());
      ++Index;
      return;
    }
    ExprRes.get();
    // Fall through for subaggregate initialization
  }

  // C++ [dcl.init.aggr]p12:
  //
  //   [...] Otherwise, if the member is itself a non-empty
  //   subaggregate, brace elision is assumed and the initializer is
  //   considered for the initialization of the first member of
  //   the subaggregate.
  if (!SemaRef.getLangOpts().OpenCL && 
      (ElemType->isAggregateType() || ElemType->isVectorType())) {
    CheckImplicitInitList(Entity, IList, ElemType, Index, StructuredList,
                          StructuredIndex);
    ++StructuredIndex;
  } else {
    if (!VerifyOnly) {
      // We cannot initialize this element, so let
      // PerformCopyInitialization produce the appropriate diagnostic.
      SemaRef.PerformCopyInitialization(Entity, SourceLocation(), expr,
                                        /*TopLevelOfInitList=*/true);
    }
    hadError = true;
    ++Index;
    ++StructuredIndex;
  }
}

void InitListChecker::CheckComplexType(const InitializedEntity &Entity,
                                       InitListExpr *IList, QualType DeclType,
                                       unsigned &Index,
                                       InitListExpr *StructuredList,
                                       unsigned &StructuredIndex) {
  assert(Index == 0 && "Index in explicit init list must be zero");

  // As an extension, clang supports complex initializers, which initialize
  // a complex number component-wise.  When an explicit initializer list for
  // a complex number contains two two initializers, this extension kicks in:
  // it exepcts the initializer list to contain two elements convertible to
  // the element type of the complex type. The first element initializes
  // the real part, and the second element intitializes the imaginary part.

  if (IList->getNumInits() != 2)
    return CheckScalarType(Entity, IList, DeclType, Index, StructuredList,
                           StructuredIndex);

  // This is an extension in C.  (The builtin _Complex type does not exist
  // in the C++ standard.)
  if (!SemaRef.getLangOpts().CPlusPlus && !VerifyOnly)
    SemaRef.Diag(IList->getLocStart(), diag::ext_complex_component_init)
      << IList->getSourceRange();

  // Initialize the complex number.
  QualType elementType = DeclType->getAs<ComplexType>()->getElementType();
  InitializedEntity ElementEntity =
    InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);

  for (unsigned i = 0; i < 2; ++i) {
    ElementEntity.setElementIndex(Index);
    CheckSubElementType(ElementEntity, IList, elementType, Index,
                        StructuredList, StructuredIndex);
  }
}


void InitListChecker::CheckScalarType(const InitializedEntity &Entity,
                                      InitListExpr *IList, QualType DeclType,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex) {
  if (Index >= IList->getNumInits()) {
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(),
                   SemaRef.getLangOpts().CPlusPlus11 ?
                     diag::warn_cxx98_compat_empty_scalar_initializer :
                     diag::err_empty_scalar_initializer)
        << IList->getSourceRange();
    hadError = !SemaRef.getLangOpts().CPlusPlus11;
    ++Index;
    ++StructuredIndex;
    return;
  }

  Expr *expr = IList->getInit(Index);
  if (InitListExpr *SubIList = dyn_cast<InitListExpr>(expr)) {
    // FIXME: This is invalid, and accepting it causes overload resolution
    // to pick the wrong overload in some corner cases.
    if (!VerifyOnly)
      SemaRef.Diag(SubIList->getLocStart(),
                   diag::ext_many_braces_around_scalar_init)
        << SubIList->getSourceRange();

    CheckScalarType(Entity, SubIList, DeclType, Index, StructuredList,
                    StructuredIndex);
    return;
  } else if (isa<DesignatedInitExpr>(expr)) {
    if (!VerifyOnly)
      SemaRef.Diag(expr->getLocStart(),
                   diag::err_designator_for_scalar_init)
        << DeclType << expr->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }

  if (VerifyOnly) {
    if (!SemaRef.CanPerformCopyInitialization(Entity,expr))
      hadError = true;
    ++Index;
    return;
  }

  ExprResult Result =
    SemaRef.PerformCopyInitialization(Entity, expr->getLocStart(), expr,
                                      /*TopLevelOfInitList=*/true);

  Expr *ResultExpr = nullptr;

  if (Result.isInvalid())
    hadError = true; // types weren't compatible.
  else {
    ResultExpr = Result.getAs<Expr>();

    if (ResultExpr != expr) {
      // The type was promoted, update initializer list.
      IList->setInit(Index, ResultExpr);
    }
  }
  if (hadError)
    ++StructuredIndex;
  else
    UpdateStructuredListElement(StructuredList, StructuredIndex, ResultExpr);
  ++Index;
}

void InitListChecker::CheckReferenceType(const InitializedEntity &Entity,
                                         InitListExpr *IList, QualType DeclType,
                                         unsigned &Index,
                                         InitListExpr *StructuredList,
                                         unsigned &StructuredIndex) {
  if (Index >= IList->getNumInits()) {
    // FIXME: It would be wonderful if we could point at the actual member. In
    // general, it would be useful to pass location information down the stack,
    // so that we know the location (or decl) of the "current object" being
    // initialized.
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(),
                    diag::err_init_reference_member_uninitialized)
        << DeclType
        << IList->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }

  Expr *expr = IList->getInit(Index);
  if (isa<InitListExpr>(expr) && !SemaRef.getLangOpts().CPlusPlus11) {
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(), diag::err_init_non_aggr_init_list)
        << DeclType << IList->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }

  if (VerifyOnly) {
    if (!SemaRef.CanPerformCopyInitialization(Entity,expr))
      hadError = true;
    ++Index;
    return;
  }

  ExprResult Result =
      SemaRef.PerformCopyInitialization(Entity, expr->getLocStart(), expr,
                                        /*TopLevelOfInitList=*/true);

  if (Result.isInvalid())
    hadError = true;

  expr = Result.getAs<Expr>();
  IList->setInit(Index, expr);

  if (hadError)
    ++StructuredIndex;
  else
    UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
  ++Index;
}

void InitListChecker::CheckVectorType(const InitializedEntity &Entity,
                                      InitListExpr *IList, QualType DeclType,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex) {
  const VectorType *VT = DeclType->getAs<VectorType>();
  unsigned maxElements = VT->getNumElements();
  unsigned numEltsInit = 0;
  QualType elementType = VT->getElementType();

  if (Index >= IList->getNumInits()) {
    // Make sure the element type can be value-initialized.
    if (VerifyOnly)
      CheckEmptyInitializable(
          InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity),
          IList->getLocEnd());
    return;
  }

  if (!SemaRef.getLangOpts().OpenCL) {
    // If the initializing element is a vector, try to copy-initialize
    // instead of breaking it apart (which is doomed to failure anyway).
    Expr *Init = IList->getInit(Index);
    if (!isa<InitListExpr>(Init) && Init->getType()->isVectorType()) {
      if (VerifyOnly) {
        if (!SemaRef.CanPerformCopyInitialization(Entity, Init))
          hadError = true;
        ++Index;
        return;
      }

  ExprResult Result =
      SemaRef.PerformCopyInitialization(Entity, Init->getLocStart(), Init,
                                        /*TopLevelOfInitList=*/true);

      Expr *ResultExpr = nullptr;
      if (Result.isInvalid())
        hadError = true; // types weren't compatible.
      else {
        ResultExpr = Result.getAs<Expr>();

        if (ResultExpr != Init) {
          // The type was promoted, update initializer list.
          IList->setInit(Index, ResultExpr);
        }
      }
      if (hadError)
        ++StructuredIndex;
      else
        UpdateStructuredListElement(StructuredList, StructuredIndex,
                                    ResultExpr);
      ++Index;
      return;
    }

    InitializedEntity ElementEntity =
      InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);

    for (unsigned i = 0; i < maxElements; ++i, ++numEltsInit) {
      // Don't attempt to go past the end of the init list
      if (Index >= IList->getNumInits()) {
        if (VerifyOnly)
          CheckEmptyInitializable(ElementEntity, IList->getLocEnd());
        break;
      }

      ElementEntity.setElementIndex(Index);
      CheckSubElementType(ElementEntity, IList, elementType, Index,
                          StructuredList, StructuredIndex);
    }

    if (VerifyOnly)
      return;

    bool isBigEndian = SemaRef.Context.getTargetInfo().isBigEndian();
    const VectorType *T = Entity.getType()->getAs<VectorType>();
    if (isBigEndian && (T->getVectorKind() == VectorType::NeonVector ||
                        T->getVectorKind() == VectorType::NeonPolyVector)) {
      // The ability to use vector initializer lists is a GNU vector extension
      // and is unrelated to the NEON intrinsics in arm_neon.h. On little
      // endian machines it works fine, however on big endian machines it 
      // exhibits surprising behaviour:
      //
      //   uint32x2_t x = {42, 64};
      //   return vget_lane_u32(x, 0); // Will return 64.
      //
      // Because of this, explicitly call out that it is non-portable.
      //
      SemaRef.Diag(IList->getLocStart(),
                   diag::warn_neon_vector_initializer_non_portable);

      const char *typeCode;
      unsigned typeSize = SemaRef.Context.getTypeSize(elementType);

      if (elementType->isFloatingType())
        typeCode = "f";
      else if (elementType->isSignedIntegerType())
        typeCode = "s";
      else if (elementType->isUnsignedIntegerType())
        typeCode = "u";
      else
        llvm_unreachable("Invalid element type!");

      SemaRef.Diag(IList->getLocStart(),
                   SemaRef.Context.getTypeSize(VT) > 64 ?
                   diag::note_neon_vector_initializer_non_portable_q :
                   diag::note_neon_vector_initializer_non_portable)
        << typeCode << typeSize;
    }

    return;
  }

  InitializedEntity ElementEntity =
    InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);

  // OpenCL initializers allows vectors to be constructed from vectors.
  for (unsigned i = 0; i < maxElements; ++i) {
    // Don't attempt to go past the end of the init list
    if (Index >= IList->getNumInits())
      break;

    ElementEntity.setElementIndex(Index);

    QualType IType = IList->getInit(Index)->getType();
    if (!IType->isVectorType()) {
      CheckSubElementType(ElementEntity, IList, elementType, Index,
                          StructuredList, StructuredIndex);
      ++numEltsInit;
    } else {
      QualType VecType;
      const VectorType *IVT = IType->getAs<VectorType>();
      unsigned numIElts = IVT->getNumElements();

      if (IType->isExtVectorType())
        VecType = SemaRef.Context.getExtVectorType(elementType, numIElts);
      else
        VecType = SemaRef.Context.getVectorType(elementType, numIElts,
                                                IVT->getVectorKind());
      CheckSubElementType(ElementEntity, IList, VecType, Index,
                          StructuredList, StructuredIndex);
      numEltsInit += numIElts;
    }
  }

  // OpenCL requires all elements to be initialized.
  if (numEltsInit != maxElements) {
    if (!VerifyOnly)
      SemaRef.Diag(IList->getLocStart(),
                   diag::err_vector_incorrect_num_initializers)
        << (numEltsInit < maxElements) << maxElements << numEltsInit;
    hadError = true;
  }
}

void InitListChecker::CheckArrayType(const InitializedEntity &Entity,
                                     InitListExpr *IList, QualType &DeclType,
                                     llvm::APSInt elementIndex,
                                     bool SubobjectIsDesignatorContext,
                                     unsigned &Index,
                                     InitListExpr *StructuredList,
                                     unsigned &StructuredIndex) {
  const ArrayType *arrayType = SemaRef.Context.getAsArrayType(DeclType);

  // Check for the special-case of initializing an array with a string.
  if (Index < IList->getNumInits()) {
    if (IsStringInit(IList->getInit(Index), arrayType, SemaRef.Context) ==
        SIF_None) {
      // We place the string literal directly into the resulting
      // initializer list. This is the only place where the structure
      // of the structured initializer list doesn't match exactly,
      // because doing so would involve allocating one character
      // constant for each string.
      if (!VerifyOnly) {
        CheckStringInit(IList->getInit(Index), DeclType, arrayType, SemaRef);
        UpdateStructuredListElement(StructuredList, StructuredIndex,
                                    IList->getInit(Index));
        StructuredList->resizeInits(SemaRef.Context, StructuredIndex);
      }
      ++Index;
      return;
    }
  }
  if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(arrayType)) {
    // Check for VLAs; in standard C it would be possible to check this
    // earlier, but I don't know where clang accepts VLAs (gcc accepts
    // them in all sorts of strange places).
    if (!VerifyOnly)
      SemaRef.Diag(VAT->getSizeExpr()->getLocStart(),
                    diag::err_variable_object_no_init)
        << VAT->getSizeExpr()->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }

  // We might know the maximum number of elements in advance.
  llvm::APSInt maxElements(elementIndex.getBitWidth(),
                           elementIndex.isUnsigned());
  bool maxElementsKnown = false;
  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(arrayType)) {
    maxElements = CAT->getSize();
    elementIndex = elementIndex.extOrTrunc(maxElements.getBitWidth());
    elementIndex.setIsUnsigned(maxElements.isUnsigned());
    maxElementsKnown = true;
  }

  QualType elementType = arrayType->getElementType();
  while (Index < IList->getNumInits()) {
    Expr *Init = IList->getInit(Index);
    if (DesignatedInitExpr *DIE = dyn_cast<DesignatedInitExpr>(Init)) {
      // If we're not the subobject that matches up with the '{' for
      // the designator, we shouldn't be handling the
      // designator. Return immediately.
      if (!SubobjectIsDesignatorContext)
        return;

      // Handle this designated initializer. elementIndex will be
      // updated to be the next array element we'll initialize.
      if (CheckDesignatedInitializer(Entity, IList, DIE, 0,
                                     DeclType, nullptr, &elementIndex, Index,
                                     StructuredList, StructuredIndex, true,
                                     false)) {
        hadError = true;
        continue;
      }

      if (elementIndex.getBitWidth() > maxElements.getBitWidth())
        maxElements = maxElements.extend(elementIndex.getBitWidth());
      else if (elementIndex.getBitWidth() < maxElements.getBitWidth())
        elementIndex = elementIndex.extend(maxElements.getBitWidth());
      elementIndex.setIsUnsigned(maxElements.isUnsigned());

      // If the array is of incomplete type, keep track of the number of
      // elements in the initializer.
      if (!maxElementsKnown && elementIndex > maxElements)
        maxElements = elementIndex;

      continue;
    }

    // If we know the maximum number of elements, and we've already
    // hit it, stop consuming elements in the initializer list.
    if (maxElementsKnown && elementIndex == maxElements)
      break;

    InitializedEntity ElementEntity =
      InitializedEntity::InitializeElement(SemaRef.Context, StructuredIndex,
                                           Entity);
    // Check this element.
    CheckSubElementType(ElementEntity, IList, elementType, Index,
                        StructuredList, StructuredIndex);
    ++elementIndex;

    // If the array is of incomplete type, keep track of the number of
    // elements in the initializer.
    if (!maxElementsKnown && elementIndex > maxElements)
      maxElements = elementIndex;
  }
  if (!hadError && DeclType->isIncompleteArrayType() && !VerifyOnly) {
    // If this is an incomplete array type, the actual type needs to
    // be calculated here.
    llvm::APSInt Zero(maxElements.getBitWidth(), maxElements.isUnsigned());
    if (maxElements == Zero) {
      // Sizing an array implicitly to zero is not allowed by ISO C,
      // but is supported by GNU.
      SemaRef.Diag(IList->getLocStart(),
                    diag::ext_typecheck_zero_array_size);
    }

    DeclType = SemaRef.Context.getConstantArrayType(elementType, maxElements,
                                                     ArrayType::Normal, 0);
  }
  if (!hadError && VerifyOnly) {
    // Check if there are any members of the array that get value-initialized.
    // If so, check if doing that is possible.
    // FIXME: This needs to detect holes left by designated initializers too.
    if (maxElementsKnown && elementIndex < maxElements)
      CheckEmptyInitializable(InitializedEntity::InitializeElement(
                                                  SemaRef.Context, 0, Entity),
                              IList->getLocEnd());
  }
}

bool InitListChecker::CheckFlexibleArrayInit(const InitializedEntity &Entity,
                                             Expr *InitExpr,
                                             FieldDecl *Field,
                                             bool TopLevelObject) {
  // Handle GNU flexible array initializers.
  unsigned FlexArrayDiag;
  if (isa<InitListExpr>(InitExpr) &&
      cast<InitListExpr>(InitExpr)->getNumInits() == 0) {
    // Empty flexible array init always allowed as an extension
    FlexArrayDiag = diag::ext_flexible_array_init;
  } else if (SemaRef.getLangOpts().CPlusPlus) {
    // Disallow flexible array init in C++; it is not required for gcc
    // compatibility, and it needs work to IRGen correctly in general.
    FlexArrayDiag = diag::err_flexible_array_init;
  } else if (!TopLevelObject) {
    // Disallow flexible array init on non-top-level object
    FlexArrayDiag = diag::err_flexible_array_init;
  } else if (Entity.getKind() != InitializedEntity::EK_Variable) {
    // Disallow flexible array init on anything which is not a variable.
    FlexArrayDiag = diag::err_flexible_array_init;
  } else if (cast<VarDecl>(Entity.getDecl())->hasLocalStorage()) {
    // Disallow flexible array init on local variables.
    FlexArrayDiag = diag::err_flexible_array_init;
  } else {
    // Allow other cases.
    FlexArrayDiag = diag::ext_flexible_array_init;
  }

  if (!VerifyOnly) {
    SemaRef.Diag(InitExpr->getLocStart(),
                 FlexArrayDiag)
      << InitExpr->getLocStart();
    SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
      << Field;
  }

  return FlexArrayDiag != diag::ext_flexible_array_init;
}

void InitListChecker::CheckStructUnionTypes(const InitializedEntity &Entity,
                                            InitListExpr *IList,
                                            QualType DeclType,
                                            RecordDecl::field_iterator Field,
                                            bool SubobjectIsDesignatorContext,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  RecordDecl* structDecl = DeclType->getAs<RecordType>()->getDecl();

  // If the record is invalid, some of it's members are invalid. To avoid
  // confusion, we forgo checking the intializer for the entire record.
  if (structDecl->isInvalidDecl()) {
    // Assume it was supposed to consume a single initializer.
    ++Index;
    hadError = true;
    return;
  }

  if (DeclType->isUnionType() && IList->getNumInits() == 0) {
    RecordDecl *RD = DeclType->getAs<RecordType>()->getDecl();

    // If there's a default initializer, use it.
    if (isa<CXXRecordDecl>(RD) && cast<CXXRecordDecl>(RD)->hasInClassInitializer()) {
      if (VerifyOnly)
        return;
      for (RecordDecl::field_iterator FieldEnd = RD->field_end();
           Field != FieldEnd; ++Field) {
        if (Field->hasInClassInitializer()) {
          StructuredList->setInitializedFieldInUnion(*Field);
          // FIXME: Actually build a CXXDefaultInitExpr?
          return;
        }
      }
    }

    // Value-initialize the first member of the union that isn't an unnamed
    // bitfield.
    for (RecordDecl::field_iterator FieldEnd = RD->field_end();
         Field != FieldEnd; ++Field) {
      if (!Field->isUnnamedBitfield()) {
        if (VerifyOnly)
          CheckEmptyInitializable(
              InitializedEntity::InitializeMember(*Field, &Entity),
              IList->getLocEnd());
        else
          StructuredList->setInitializedFieldInUnion(*Field);
        break;
      }
    }
    return;
  }

  // If structDecl is a forward declaration, this loop won't do
  // anything except look at designated initializers; That's okay,
  // because an error should get printed out elsewhere. It might be
  // worthwhile to skip over the rest of the initializer, though.
  RecordDecl *RD = DeclType->getAs<RecordType>()->getDecl();
  RecordDecl::field_iterator FieldEnd = RD->field_end();
  bool InitializedSomething = false;
  bool CheckForMissingFields = true;
  while (Index < IList->getNumInits()) {
    Expr *Init = IList->getInit(Index);

    if (DesignatedInitExpr *DIE = dyn_cast<DesignatedInitExpr>(Init)) {
      // If we're not the subobject that matches up with the '{' for
      // the designator, we shouldn't be handling the
      // designator. Return immediately.
      if (!SubobjectIsDesignatorContext)
        return;

      // Handle this designated initializer. Field will be updated to
      // the next field that we'll be initializing.
      if (CheckDesignatedInitializer(Entity, IList, DIE, 0,
                                     DeclType, &Field, nullptr, Index,
                                     StructuredList, StructuredIndex,
                                     true, TopLevelObject))
        hadError = true;

      InitializedSomething = true;

      // Disable check for missing fields when designators are used.
      // This matches gcc behaviour.
      CheckForMissingFields = false;
      continue;
    }

    if (Field == FieldEnd) {
      // We've run out of fields. We're done.
      break;
    }

    // We've already initialized a member of a union. We're done.
    if (InitializedSomething && DeclType->isUnionType())
      break;

    // If we've hit the flexible array member at the end, we're done.
    if (Field->getType()->isIncompleteArrayType())
      break;

    if (Field->isUnnamedBitfield()) {
      // Don't initialize unnamed bitfields, e.g. "int : 20;"
      ++Field;
      continue;
    }

    // Make sure we can use this declaration.
    bool InvalidUse;
    if (VerifyOnly)
      InvalidUse = !SemaRef.CanUseDecl(*Field);
    else
      InvalidUse = SemaRef.DiagnoseUseOfDecl(*Field,
                                          IList->getInit(Index)->getLocStart());
    if (InvalidUse) {
      ++Index;
      ++Field;
      hadError = true;
      continue;
    }

    InitializedEntity MemberEntity =
      InitializedEntity::InitializeMember(*Field, &Entity);
    CheckSubElementType(MemberEntity, IList, Field->getType(), Index,
                        StructuredList, StructuredIndex);
    InitializedSomething = true;

    if (DeclType->isUnionType() && !VerifyOnly) {
      // Initialize the first field within the union.
      StructuredList->setInitializedFieldInUnion(*Field);
    }

    ++Field;
  }

  // Emit warnings for missing struct field initializers.
  if (!VerifyOnly && InitializedSomething && CheckForMissingFields &&
      Field != FieldEnd && !Field->getType()->isIncompleteArrayType() &&
      !DeclType->isUnionType()) {
    // It is possible we have one or more unnamed bitfields remaining.
    // Find first (if any) named field and emit warning.
    for (RecordDecl::field_iterator it = Field, end = RD->field_end();
         it != end; ++it) {
      if (!it->isUnnamedBitfield() && !it->hasInClassInitializer()) {
        SemaRef.Diag(IList->getSourceRange().getEnd(),
                     diag::warn_missing_field_initializers) << *it;
        break;
      }
    }
  }

  // Check that any remaining fields can be value-initialized.
  if (VerifyOnly && Field != FieldEnd && !DeclType->isUnionType() &&
      !Field->getType()->isIncompleteArrayType()) {
    // FIXME: Should check for holes left by designated initializers too.
    for (; Field != FieldEnd && !hadError; ++Field) {
      if (!Field->isUnnamedBitfield() && !Field->hasInClassInitializer())
        CheckEmptyInitializable(
            InitializedEntity::InitializeMember(*Field, &Entity),
            IList->getLocEnd());
    }
  }

  if (Field == FieldEnd || !Field->getType()->isIncompleteArrayType() ||
      Index >= IList->getNumInits())
    return;

  if (CheckFlexibleArrayInit(Entity, IList->getInit(Index), *Field,
                             TopLevelObject)) {
    hadError = true;
    ++Index;
    return;
  }

  InitializedEntity MemberEntity =
    InitializedEntity::InitializeMember(*Field, &Entity);

  if (isa<InitListExpr>(IList->getInit(Index)))
    CheckSubElementType(MemberEntity, IList, Field->getType(), Index,
                        StructuredList, StructuredIndex);
  else
    CheckImplicitInitList(MemberEntity, IList, Field->getType(), Index,
                          StructuredList, StructuredIndex);
}

/// \brief Expand a field designator that refers to a member of an
/// anonymous struct or union into a series of field designators that
/// refers to the field within the appropriate subobject.
///
static void ExpandAnonymousFieldDesignator(Sema &SemaRef,
                                           DesignatedInitExpr *DIE,
                                           unsigned DesigIdx,
                                           IndirectFieldDecl *IndirectField) {
  typedef DesignatedInitExpr::Designator Designator;

  // Build the replacement designators.
  SmallVector<Designator, 4> Replacements;
  for (IndirectFieldDecl::chain_iterator PI = IndirectField->chain_begin(),
       PE = IndirectField->chain_end(); PI != PE; ++PI) {
    if (PI + 1 == PE)
      Replacements.push_back(Designator((IdentifierInfo *)nullptr,
                                    DIE->getDesignator(DesigIdx)->getDotLoc(),
                                DIE->getDesignator(DesigIdx)->getFieldLoc()));
    else
      Replacements.push_back(Designator((IdentifierInfo *)nullptr,
                                        SourceLocation(), SourceLocation()));
    assert(isa<FieldDecl>(*PI));
    Replacements.back().setField(cast<FieldDecl>(*PI));
  }

  // Expand the current designator into the set of replacement
  // designators, so we have a full subobject path down to where the
  // member of the anonymous struct/union is actually stored.
  DIE->ExpandDesignator(SemaRef.Context, DesigIdx, &Replacements[0],
                        &Replacements[0] + Replacements.size());
}

static DesignatedInitExpr *CloneDesignatedInitExpr(Sema &SemaRef,
                                                   DesignatedInitExpr *DIE) {
  unsigned NumIndexExprs = DIE->getNumSubExprs() - 1;
  SmallVector<Expr*, 4> IndexExprs(NumIndexExprs);
  for (unsigned I = 0; I < NumIndexExprs; ++I)
    IndexExprs[I] = DIE->getSubExpr(I + 1);
  return DesignatedInitExpr::Create(SemaRef.Context, DIE->designators_begin(),
                                    DIE->size(), IndexExprs,
                                    DIE->getEqualOrColonLoc(),
                                    DIE->usesGNUSyntax(), DIE->getInit());
}

namespace {

// Callback to only accept typo corrections that are for field members of
// the given struct or union.
class FieldInitializerValidatorCCC : public CorrectionCandidateCallback {
 public:
  explicit FieldInitializerValidatorCCC(RecordDecl *RD)
      : Record(RD) {}

  bool ValidateCandidate(const TypoCorrection &candidate) override {
    FieldDecl *FD = candidate.getCorrectionDeclAs<FieldDecl>();
    return FD && FD->getDeclContext()->getRedeclContext()->Equals(Record);
  }

 private:
  RecordDecl *Record;
};

}

/// @brief Check the well-formedness of a C99 designated initializer.
///
/// Determines whether the designated initializer @p DIE, which
/// resides at the given @p Index within the initializer list @p
/// IList, is well-formed for a current object of type @p DeclType
/// (C99 6.7.8). The actual subobject that this designator refers to
/// within the current subobject is returned in either
/// @p NextField or @p NextElementIndex (whichever is appropriate).
///
/// @param IList  The initializer list in which this designated
/// initializer occurs.
///
/// @param DIE The designated initializer expression.
///
/// @param DesigIdx  The index of the current designator.
///
/// @param CurrentObjectType The type of the "current object" (C99 6.7.8p17),
/// into which the designation in @p DIE should refer.
///
/// @param NextField  If non-NULL and the first designator in @p DIE is
/// a field, this will be set to the field declaration corresponding
/// to the field named by the designator.
///
/// @param NextElementIndex  If non-NULL and the first designator in @p
/// DIE is an array designator or GNU array-range designator, this
/// will be set to the last index initialized by this designator.
///
/// @param Index  Index into @p IList where the designated initializer
/// @p DIE occurs.
///
/// @param StructuredList  The initializer list expression that
/// describes all of the subobject initializers in the order they'll
/// actually be initialized.
///
/// @returns true if there was an error, false otherwise.
bool
InitListChecker::CheckDesignatedInitializer(const InitializedEntity &Entity,
                                            InitListExpr *IList,
                                            DesignatedInitExpr *DIE,
                                            unsigned DesigIdx,
                                            QualType &CurrentObjectType,
                                          RecordDecl::field_iterator *NextField,
                                            llvm::APSInt *NextElementIndex,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool FinishSubobjectInit,
                                            bool TopLevelObject) {
  if (DesigIdx == DIE->size()) {
    // Check the actual initialization for the designated object type.
    bool prevHadError = hadError;

    // Temporarily remove the designator expression from the
    // initializer list that the child calls see, so that we don't try
    // to re-process the designator.
    unsigned OldIndex = Index;
    IList->setInit(OldIndex, DIE->getInit());

    CheckSubElementType(Entity, IList, CurrentObjectType, Index,
                        StructuredList, StructuredIndex);

    // Restore the designated initializer expression in the syntactic
    // form of the initializer list.
    if (IList->getInit(OldIndex) != DIE->getInit())
      DIE->setInit(IList->getInit(OldIndex));
    IList->setInit(OldIndex, DIE);

    return hadError && !prevHadError;
  }

  DesignatedInitExpr::Designator *D = DIE->getDesignator(DesigIdx);
  bool IsFirstDesignator = (DesigIdx == 0);
  if (!VerifyOnly) {
    assert((IsFirstDesignator || StructuredList) &&
           "Need a non-designated initializer list to start from");

    // Determine the structural initializer list that corresponds to the
    // current subobject.
    if (IsFirstDesignator)
      StructuredList = SyntacticToSemantic.lookup(IList);
    else {
      Expr *ExistingInit = StructuredIndex < StructuredList->getNumInits() ?
          StructuredList->getInit(StructuredIndex) : nullptr;
      if (!ExistingInit && StructuredList->hasArrayFiller())
        ExistingInit = StructuredList->getArrayFiller();

      if (!ExistingInit)
        StructuredList =
          getStructuredSubobjectInit(IList, Index, CurrentObjectType,
                                     StructuredList, StructuredIndex,
                                     SourceRange(D->getLocStart(),
                                                 DIE->getLocEnd()));
      else if (InitListExpr *Result = dyn_cast<InitListExpr>(ExistingInit))
        StructuredList = Result;
      else {
        if (DesignatedInitUpdateExpr *E =
                dyn_cast<DesignatedInitUpdateExpr>(ExistingInit))
          StructuredList = E->getUpdater();
        else {
          DesignatedInitUpdateExpr *DIUE =
              new (SemaRef.Context) DesignatedInitUpdateExpr(SemaRef.Context,
                                        D->getLocStart(), ExistingInit,
                                        DIE->getLocEnd());
          StructuredList->updateInit(SemaRef.Context, StructuredIndex, DIUE);
          StructuredList = DIUE->getUpdater();
        }

        // We need to check on source range validity because the previous
        // initializer does not have to be an explicit initializer. e.g.,
        //
        // struct P { int a, b; };
        // struct PP { struct P p } l = { { .a = 2 }, .p.b = 3 };
        //
        // There is an overwrite taking place because the first braced initializer
        // list "{ .a = 2 }" already provides value for .p.b (which is zero).
        if (ExistingInit->getSourceRange().isValid()) {
          // We are creating an initializer list that initializes the
          // subobjects of the current object, but there was already an
          // initialization that completely initialized the current
          // subobject, e.g., by a compound literal:
          //
          // struct X { int a, b; };
          // struct X xs[] = { [0] = (struct X) { 1, 2 }, [0].b = 3 };
          //
          // Here, xs[0].a == 0 and xs[0].b == 3, since the second,
          // designated initializer re-initializes the whole
          // subobject [0], overwriting previous initializers.
          SemaRef.Diag(D->getLocStart(),
                       diag::warn_subobject_initializer_overrides)
            << SourceRange(D->getLocStart(), DIE->getLocEnd());
  
          SemaRef.Diag(ExistingInit->getLocStart(),
                       diag::note_previous_initializer)
            << /*FIXME:has side effects=*/0
            << ExistingInit->getSourceRange();
        }
      }
    }
    assert(StructuredList && "Expected a structured initializer list");
  }

  if (D->isFieldDesignator()) {
    // C99 6.7.8p7:
    //
    //   If a designator has the form
    //
    //      . identifier
    //
    //   then the current object (defined below) shall have
    //   structure or union type and the identifier shall be the
    //   name of a member of that type.
    const RecordType *RT = CurrentObjectType->getAs<RecordType>();
    if (!RT) {
      SourceLocation Loc = D->getDotLoc();
      if (Loc.isInvalid())
        Loc = D->getFieldLoc();
      if (!VerifyOnly)
        SemaRef.Diag(Loc, diag::err_field_designator_non_aggr)
          << SemaRef.getLangOpts().CPlusPlus << CurrentObjectType;
      ++Index;
      return true;
    }

    FieldDecl *KnownField = D->getField();
    if (!KnownField) {
      IdentifierInfo *FieldName = D->getFieldName();
      DeclContext::lookup_result Lookup = RT->getDecl()->lookup(FieldName);
      for (NamedDecl *ND : Lookup) {
        if (auto *FD = dyn_cast<FieldDecl>(ND)) {
          KnownField = FD;
          break;
        }
        if (auto *IFD = dyn_cast<IndirectFieldDecl>(ND)) {
          // In verify mode, don't modify the original.
          if (VerifyOnly)
            DIE = CloneDesignatedInitExpr(SemaRef, DIE);
          ExpandAnonymousFieldDesignator(SemaRef, DIE, DesigIdx, IFD);
          D = DIE->getDesignator(DesigIdx);
          KnownField = cast<FieldDecl>(*IFD->chain_begin());
          break;
        }
      }
      if (!KnownField) {
        if (VerifyOnly) {
          ++Index;
          return true;  // No typo correction when just trying this out.
        }

        // Name lookup found something, but it wasn't a field.
        if (!Lookup.empty()) {
          SemaRef.Diag(D->getFieldLoc(), diag::err_field_designator_nonfield)
            << FieldName;
          SemaRef.Diag(Lookup.front()->getLocation(),
                       diag::note_field_designator_found);
          ++Index;
          return true;
        }

        // Name lookup didn't find anything.
        // Determine whether this was a typo for another field name.
        if (TypoCorrection Corrected = SemaRef.CorrectTypo(
                DeclarationNameInfo(FieldName, D->getFieldLoc()),
                Sema::LookupMemberName, /*Scope=*/nullptr, /*SS=*/nullptr,
                llvm::make_unique<FieldInitializerValidatorCCC>(RT->getDecl()),
                Sema::CTK_ErrorRecovery, RT->getDecl())) {
          SemaRef.diagnoseTypo(
              Corrected,
              SemaRef.PDiag(diag::err_field_designator_unknown_suggest)
                << FieldName << CurrentObjectType);
          KnownField = Corrected.getCorrectionDeclAs<FieldDecl>();
          hadError = true;
        } else {
          // Typo correction didn't find anything.
          SemaRef.Diag(D->getFieldLoc(), diag::err_field_designator_unknown)
            << FieldName << CurrentObjectType;
          ++Index;
          return true;
        }
      }
    }

    unsigned FieldIndex = 0;
    for (auto *FI : RT->getDecl()->fields()) {
      if (FI->isUnnamedBitfield())
        continue;
      if (KnownField == FI)
        break;
      ++FieldIndex;
    }

    RecordDecl::field_iterator Field =
        RecordDecl::field_iterator(DeclContext::decl_iterator(KnownField));

    // All of the fields of a union are located at the same place in
    // the initializer list.
    if (RT->getDecl()->isUnion()) {
      FieldIndex = 0;
      if (!VerifyOnly) {
        FieldDecl *CurrentField = StructuredList->getInitializedFieldInUnion();
        if (CurrentField && CurrentField != *Field) {
          assert(StructuredList->getNumInits() == 1
                 && "A union should never have more than one initializer!");

          // we're about to throw away an initializer, emit warning
          SemaRef.Diag(D->getFieldLoc(),
                       diag::warn_initializer_overrides)
            << D->getSourceRange();
          Expr *ExistingInit = StructuredList->getInit(0);
          SemaRef.Diag(ExistingInit->getLocStart(),
                       diag::note_previous_initializer)
            << /*FIXME:has side effects=*/0
            << ExistingInit->getSourceRange();

          // remove existing initializer
          StructuredList->resizeInits(SemaRef.Context, 0);
          StructuredList->setInitializedFieldInUnion(nullptr);
        }

        StructuredList->setInitializedFieldInUnion(*Field);
      }
    }

    // Make sure we can use this declaration.
    bool InvalidUse;
    if (VerifyOnly)
      InvalidUse = !SemaRef.CanUseDecl(*Field);
    else
      InvalidUse = SemaRef.DiagnoseUseOfDecl(*Field, D->getFieldLoc());
    if (InvalidUse) {
      ++Index;
      return true;
    }

    if (!VerifyOnly) {
      // Update the designator with the field declaration.
      D->setField(*Field);

      // Make sure that our non-designated initializer list has space
      // for a subobject corresponding to this field.
      if (FieldIndex >= StructuredList->getNumInits())
        StructuredList->resizeInits(SemaRef.Context, FieldIndex + 1);
    }

    // This designator names a flexible array member.
    if (Field->getType()->isIncompleteArrayType()) {
      bool Invalid = false;
      if ((DesigIdx + 1) != DIE->size()) {
        // We can't designate an object within the flexible array
        // member (because GCC doesn't allow it).
        if (!VerifyOnly) {
          DesignatedInitExpr::Designator *NextD
            = DIE->getDesignator(DesigIdx + 1);
          SemaRef.Diag(NextD->getLocStart(),
                        diag::err_designator_into_flexible_array_member)
            << SourceRange(NextD->getLocStart(),
                           DIE->getLocEnd());
          SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
            << *Field;
        }
        Invalid = true;
      }

      if (!hadError && !isa<InitListExpr>(DIE->getInit()) &&
          !isa<StringLiteral>(DIE->getInit())) {
        // The initializer is not an initializer list.
        if (!VerifyOnly) {
          SemaRef.Diag(DIE->getInit()->getLocStart(),
                        diag::err_flexible_array_init_needs_braces)
            << DIE->getInit()->getSourceRange();
          SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
            << *Field;
        }
        Invalid = true;
      }

      // Check GNU flexible array initializer.
      if (!Invalid && CheckFlexibleArrayInit(Entity, DIE->getInit(), *Field,
                                             TopLevelObject))
        Invalid = true;

      if (Invalid) {
        ++Index;
        return true;
      }

      // Initialize the array.
      bool prevHadError = hadError;
      unsigned newStructuredIndex = FieldIndex;
      unsigned OldIndex = Index;
      IList->setInit(Index, DIE->getInit());

      InitializedEntity MemberEntity =
        InitializedEntity::InitializeMember(*Field, &Entity);
      CheckSubElementType(MemberEntity, IList, Field->getType(), Index,
                          StructuredList, newStructuredIndex);

      IList->setInit(OldIndex, DIE);
      if (hadError && !prevHadError) {
        ++Field;
        ++FieldIndex;
        if (NextField)
          *NextField = Field;
        StructuredIndex = FieldIndex;
        return true;
      }
    } else {
      // Recurse to check later designated subobjects.
      QualType FieldType = Field->getType();
      unsigned newStructuredIndex = FieldIndex;

      InitializedEntity MemberEntity =
        InitializedEntity::InitializeMember(*Field, &Entity);
      if (CheckDesignatedInitializer(MemberEntity, IList, DIE, DesigIdx + 1,
                                     FieldType, nullptr, nullptr, Index,
                                     StructuredList, newStructuredIndex,
                                     FinishSubobjectInit, false))
        return true;
    }

    // Find the position of the next field to be initialized in this
    // subobject.
    ++Field;
    ++FieldIndex;

    // If this the first designator, our caller will continue checking
    // the rest of this struct/class/union subobject.
    if (IsFirstDesignator) {
      if (NextField)
        *NextField = Field;
      StructuredIndex = FieldIndex;
      return false;
    }

    if (!FinishSubobjectInit)
      return false;

    // We've already initialized something in the union; we're done.
    if (RT->getDecl()->isUnion())
      return hadError;

    // Check the remaining fields within this class/struct/union subobject.
    bool prevHadError = hadError;

    CheckStructUnionTypes(Entity, IList, CurrentObjectType, Field, false, Index,
                          StructuredList, FieldIndex);
    return hadError && !prevHadError;
  }

  // C99 6.7.8p6:
  //
  //   If a designator has the form
  //
  //      [ constant-expression ]
  //
  //   then the current object (defined below) shall have array
  //   type and the expression shall be an integer constant
  //   expression. If the array is of unknown size, any
  //   nonnegative value is valid.
  //
  // Additionally, cope with the GNU extension that permits
  // designators of the form
  //
  //      [ constant-expression ... constant-expression ]
  const ArrayType *AT = SemaRef.Context.getAsArrayType(CurrentObjectType);
  if (!AT) {
    if (!VerifyOnly)
      SemaRef.Diag(D->getLBracketLoc(), diag::err_array_designator_non_array)
        << CurrentObjectType;
    ++Index;
    return true;
  }

  Expr *IndexExpr = nullptr;
  llvm::APSInt DesignatedStartIndex, DesignatedEndIndex;
  if (D->isArrayDesignator()) {
    IndexExpr = DIE->getArrayIndex(*D);
    DesignatedStartIndex = IndexExpr->EvaluateKnownConstInt(SemaRef.Context);
    DesignatedEndIndex = DesignatedStartIndex;
  } else {
    assert(D->isArrayRangeDesignator() && "Need array-range designator");

    DesignatedStartIndex =
      DIE->getArrayRangeStart(*D)->EvaluateKnownConstInt(SemaRef.Context);
    DesignatedEndIndex =
      DIE->getArrayRangeEnd(*D)->EvaluateKnownConstInt(SemaRef.Context);
    IndexExpr = DIE->getArrayRangeEnd(*D);

    // Codegen can't handle evaluating array range designators that have side
    // effects, because we replicate the AST value for each initialized element.
    // As such, set the sawArrayRangeDesignator() bit if we initialize multiple
    // elements with something that has a side effect, so codegen can emit an
    // "error unsupported" error instead of miscompiling the app.
    if (DesignatedStartIndex.getZExtValue()!=DesignatedEndIndex.getZExtValue()&&
        DIE->getInit()->HasSideEffects(SemaRef.Context) && !VerifyOnly)
      FullyStructuredList->sawArrayRangeDesignator();
  }

  if (isa<ConstantArrayType>(AT)) {
    llvm::APSInt MaxElements(cast<ConstantArrayType>(AT)->getSize(), false);
    DesignatedStartIndex
      = DesignatedStartIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedStartIndex.setIsUnsigned(MaxElements.isUnsigned());
    DesignatedEndIndex
      = DesignatedEndIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedEndIndex.setIsUnsigned(MaxElements.isUnsigned());
    if (DesignatedEndIndex >= MaxElements) {
      if (!VerifyOnly)
        SemaRef.Diag(IndexExpr->getLocStart(),
                      diag::err_array_designator_too_large)
          << DesignatedEndIndex.toString(10) << MaxElements.toString(10)
          << IndexExpr->getSourceRange();
      ++Index;
      return true;
    }
  } else {
    unsigned DesignatedIndexBitWidth =
      ConstantArrayType::getMaxSizeBits(SemaRef.Context);
    DesignatedStartIndex =
      DesignatedStartIndex.extOrTrunc(DesignatedIndexBitWidth);
    DesignatedEndIndex =
      DesignatedEndIndex.extOrTrunc(DesignatedIndexBitWidth);
    DesignatedStartIndex.setIsUnsigned(true);
    DesignatedEndIndex.setIsUnsigned(true);
  }

  if (!VerifyOnly && StructuredList->isStringLiteralInit()) {
    // We're modifying a string literal init; we have to decompose the string
    // so we can modify the individual characters.
    ASTContext &Context = SemaRef.Context;
    Expr *SubExpr = StructuredList->getInit(0)->IgnoreParens();

    // Compute the character type
    QualType CharTy = AT->getElementType();

    // Compute the type of the integer literals.
    QualType PromotedCharTy = CharTy;
    if (CharTy->isPromotableIntegerType())
      PromotedCharTy = Context.getPromotedIntegerType(CharTy);
    unsigned PromotedCharTyWidth = Context.getTypeSize(PromotedCharTy);

    if (StringLiteral *SL = dyn_cast<StringLiteral>(SubExpr)) {
      // Get the length of the string.
      uint64_t StrLen = SL->getLength();
      if (cast<ConstantArrayType>(AT)->getSize().ult(StrLen))
        StrLen = cast<ConstantArrayType>(AT)->getSize().getZExtValue();
      StructuredList->resizeInits(Context, StrLen);

      // Build a literal for each character in the string, and put them into
      // the init list.
      for (unsigned i = 0, e = StrLen; i != e; ++i) {
        llvm::APInt CodeUnit(PromotedCharTyWidth, SL->getCodeUnit(i));
        Expr *Init = new (Context) IntegerLiteral(
            Context, CodeUnit, PromotedCharTy, SubExpr->getExprLoc());
        if (CharTy != PromotedCharTy)
          Init = ImplicitCastExpr::Create(Context, CharTy, CK_IntegralCast,
                                          Init, nullptr, VK_RValue);
        StructuredList->updateInit(Context, i, Init);
      }
    } else {
      ObjCEncodeExpr *E = cast<ObjCEncodeExpr>(SubExpr);
      std::string Str;
      Context.getObjCEncodingForType(E->getEncodedType(), Str);

      // Get the length of the string.
      uint64_t StrLen = Str.size();
      if (cast<ConstantArrayType>(AT)->getSize().ult(StrLen))
        StrLen = cast<ConstantArrayType>(AT)->getSize().getZExtValue();
      StructuredList->resizeInits(Context, StrLen);

      // Build a literal for each character in the string, and put them into
      // the init list.
      for (unsigned i = 0, e = StrLen; i != e; ++i) {
        llvm::APInt CodeUnit(PromotedCharTyWidth, Str[i]);
        Expr *Init = new (Context) IntegerLiteral(
            Context, CodeUnit, PromotedCharTy, SubExpr->getExprLoc());
        if (CharTy != PromotedCharTy)
          Init = ImplicitCastExpr::Create(Context, CharTy, CK_IntegralCast,
                                          Init, nullptr, VK_RValue);
        StructuredList->updateInit(Context, i, Init);
      }
    }
  }

  // Make sure that our non-designated initializer list has space
  // for a subobject corresponding to this array element.
  if (!VerifyOnly &&
      DesignatedEndIndex.getZExtValue() >= StructuredList->getNumInits())
    StructuredList->resizeInits(SemaRef.Context,
                                DesignatedEndIndex.getZExtValue() + 1);

  // Repeatedly perform subobject initializations in the range
  // [DesignatedStartIndex, DesignatedEndIndex].

  // Move to the next designator
  unsigned ElementIndex = DesignatedStartIndex.getZExtValue();
  unsigned OldIndex = Index;

  InitializedEntity ElementEntity =
    InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);

  while (DesignatedStartIndex <= DesignatedEndIndex) {
    // Recurse to check later designated subobjects.
    QualType ElementType = AT->getElementType();
    Index = OldIndex;

    ElementEntity.setElementIndex(ElementIndex);
    if (CheckDesignatedInitializer(
            ElementEntity, IList, DIE, DesigIdx + 1, ElementType, nullptr,
            nullptr, Index, StructuredList, ElementIndex,
            FinishSubobjectInit && (DesignatedStartIndex == DesignatedEndIndex),
            false))
      return true;

    // Move to the next index in the array that we'll be initializing.
    ++DesignatedStartIndex;
    ElementIndex = DesignatedStartIndex.getZExtValue();
  }

  // If this the first designator, our caller will continue checking
  // the rest of this array subobject.
  if (IsFirstDesignator) {
    if (NextElementIndex)
      *NextElementIndex = DesignatedStartIndex;
    StructuredIndex = ElementIndex;
    return false;
  }

  if (!FinishSubobjectInit)
    return false;

  // Check the remaining elements within this array subobject.
  bool prevHadError = hadError;
  CheckArrayType(Entity, IList, CurrentObjectType, DesignatedStartIndex,
                 /*SubobjectIsDesignatorContext=*/false, Index,
                 StructuredList, ElementIndex);
  return hadError && !prevHadError;
}

// Get the structured initializer list for a subobject of type
// @p CurrentObjectType.
InitListExpr *
InitListChecker::getStructuredSubobjectInit(InitListExpr *IList, unsigned Index,
                                            QualType CurrentObjectType,
                                            InitListExpr *StructuredList,
                                            unsigned StructuredIndex,
                                            SourceRange InitRange,
                                            bool IsFullyOverwritten) {
  if (VerifyOnly)
    return nullptr; // No structured list in verification-only mode.
  Expr *ExistingInit = nullptr;
  if (!StructuredList)
    ExistingInit = SyntacticToSemantic.lookup(IList);
  else if (StructuredIndex < StructuredList->getNumInits())
    ExistingInit = StructuredList->getInit(StructuredIndex);

  if (InitListExpr *Result = dyn_cast_or_null<InitListExpr>(ExistingInit))
    // There might have already been initializers for subobjects of the current
    // object, but a subsequent initializer list will overwrite the entirety
    // of the current object. (See DR 253 and C99 6.7.8p21). e.g.,
    //
    // struct P { char x[6]; };
    // struct P l = { .x[2] = 'x', .x = { [0] = 'f' } };
    //
    // The first designated initializer is ignored, and l.x is just "f".
    if (!IsFullyOverwritten)
      return Result;

  if (ExistingInit) {
    // We are creating an initializer list that initializes the
    // subobjects of the current object, but there was already an
    // initialization that completely initialized the current
    // subobject, e.g., by a compound literal:
    //
    // struct X { int a, b; };
    // struct X xs[] = { [0] = (struct X) { 1, 2 }, [0].b = 3 };
    //
    // Here, xs[0].a == 0 and xs[0].b == 3, since the second,
    // designated initializer re-initializes the whole
    // subobject [0], overwriting previous initializers.
    SemaRef.Diag(InitRange.getBegin(),
                 diag::warn_subobject_initializer_overrides)
      << InitRange;
    SemaRef.Diag(ExistingInit->getLocStart(),
                  diag::note_previous_initializer)
      << /*FIXME:has side effects=*/0
      << ExistingInit->getSourceRange();
  }

  InitListExpr *Result
    = new (SemaRef.Context) InitListExpr(SemaRef.Context,
                                         InitRange.getBegin(), None,
                                         InitRange.getEnd());

  QualType ResultType = CurrentObjectType;
  if (!ResultType->isArrayType())
    ResultType = ResultType.getNonLValueExprType(SemaRef.Context);
  Result->setType(ResultType);

  // Pre-allocate storage for the structured initializer list.
  unsigned NumElements = 0;
  unsigned NumInits = 0;
  bool GotNumInits = false;
  if (!StructuredList) {
    NumInits = IList->getNumInits();
    GotNumInits = true;
  } else if (Index < IList->getNumInits()) {
    if (InitListExpr *SubList = dyn_cast<InitListExpr>(IList->getInit(Index))) {
      NumInits = SubList->getNumInits();
      GotNumInits = true;
    }
  }

  if (const ArrayType *AType
      = SemaRef.Context.getAsArrayType(CurrentObjectType)) {
    if (const ConstantArrayType *CAType = dyn_cast<ConstantArrayType>(AType)) {
      NumElements = CAType->getSize().getZExtValue();
      // Simple heuristic so that we don't allocate a very large
      // initializer with many empty entries at the end.
      if (GotNumInits && NumElements > NumInits)
        NumElements = 0;
    }
  } else if (const VectorType *VType = CurrentObjectType->getAs<VectorType>())
    NumElements = VType->getNumElements();
  else if (const RecordType *RType = CurrentObjectType->getAs<RecordType>()) {
    RecordDecl *RDecl = RType->getDecl();
    if (RDecl->isUnion())
      NumElements = 1;
    else
      NumElements = std::distance(RDecl->field_begin(), RDecl->field_end());
  }

  Result->reserveInits(SemaRef.Context, NumElements);

  // Link this new initializer list into the structured initializer
  // lists.
  if (StructuredList)
    StructuredList->updateInit(SemaRef.Context, StructuredIndex, Result);
  else {
    Result->setSyntacticForm(IList);
    SyntacticToSemantic[IList] = Result;
  }

  return Result;
}

/// Update the initializer at index @p StructuredIndex within the
/// structured initializer list to the value @p expr.
void InitListChecker::UpdateStructuredListElement(InitListExpr *StructuredList,
                                                  unsigned &StructuredIndex,
                                                  Expr *expr) {
  // No structured initializer list to update
  if (!StructuredList)
    return;

  if (Expr *PrevInit = StructuredList->updateInit(SemaRef.Context,
                                                  StructuredIndex, expr)) {
    // This initializer overwrites a previous initializer. Warn.
    // We need to check on source range validity because the previous
    // initializer does not have to be an explicit initializer.
    // struct P { int a, b; };
    // struct PP { struct P p } l = { { .a = 2 }, .p.b = 3 };
    // There is an overwrite taking place because the first braced initializer
    // list "{ .a = 2 }' already provides value for .p.b (which is zero).
    if (PrevInit->getSourceRange().isValid()) {
      SemaRef.Diag(expr->getLocStart(),
                   diag::warn_initializer_overrides)
        << expr->getSourceRange();

      SemaRef.Diag(PrevInit->getLocStart(),
                   diag::note_previous_initializer)
        << /*FIXME:has side effects=*/0
        << PrevInit->getSourceRange();
    }
  }

  ++StructuredIndex;
}

/// Check that the given Index expression is a valid array designator
/// value. This is essentially just a wrapper around
/// VerifyIntegerConstantExpression that also checks for negative values
/// and produces a reasonable diagnostic if there is a
/// failure. Returns the index expression, possibly with an implicit cast
/// added, on success.  If everything went okay, Value will receive the
/// value of the constant expression.
static ExprResult
CheckArrayDesignatorExpr(Sema &S, Expr *Index, llvm::APSInt &Value) {
  SourceLocation Loc = Index->getLocStart();

  // Make sure this is an integer constant expression.
  ExprResult Result = S.VerifyIntegerConstantExpression(Index, &Value);
  if (Result.isInvalid())
    return Result;

  if (Value.isSigned() && Value.isNegative())
    return S.Diag(Loc, diag::err_array_designator_negative)
      << Value.toString(10) << Index->getSourceRange();

  Value.setIsUnsigned(true);
  return Result;
}

ExprResult Sema::ActOnDesignatedInitializer(Designation &Desig,
                                            SourceLocation Loc,
                                            bool GNUSyntax,
                                            ExprResult Init) {
  typedef DesignatedInitExpr::Designator ASTDesignator;

  bool Invalid = false;
  SmallVector<ASTDesignator, 32> Designators;
  SmallVector<Expr *, 32> InitExpressions;

  // Build designators and check array designator expressions.
  for (unsigned Idx = 0; Idx < Desig.getNumDesignators(); ++Idx) {
    const Designator &D = Desig.getDesignator(Idx);
    switch (D.getKind()) {
    case Designator::FieldDesignator:
      Designators.push_back(ASTDesignator(D.getField(), D.getDotLoc(),
                                          D.getFieldLoc()));
      break;

    case Designator::ArrayDesignator: {
      Expr *Index = static_cast<Expr *>(D.getArrayIndex());
      llvm::APSInt IndexValue;
      if (!Index->isTypeDependent() && !Index->isValueDependent())
        Index = CheckArrayDesignatorExpr(*this, Index, IndexValue).get();
      if (!Index)
        Invalid = true;
      else {
        Designators.push_back(ASTDesignator(InitExpressions.size(),
                                            D.getLBracketLoc(),
                                            D.getRBracketLoc()));
        InitExpressions.push_back(Index);
      }
      break;
    }

    case Designator::ArrayRangeDesignator: {
      Expr *StartIndex = static_cast<Expr *>(D.getArrayRangeStart());
      Expr *EndIndex = static_cast<Expr *>(D.getArrayRangeEnd());
      llvm::APSInt StartValue;
      llvm::APSInt EndValue;
      bool StartDependent = StartIndex->isTypeDependent() ||
                            StartIndex->isValueDependent();
      bool EndDependent = EndIndex->isTypeDependent() ||
                          EndIndex->isValueDependent();
      if (!StartDependent)
        StartIndex =
            CheckArrayDesignatorExpr(*this, StartIndex, StartValue).get();
      if (!EndDependent)
        EndIndex = CheckArrayDesignatorExpr(*this, EndIndex, EndValue).get();

      if (!StartIndex || !EndIndex)
        Invalid = true;
      else {
        // Make sure we're comparing values with the same bit width.
        if (StartDependent || EndDependent) {
          // Nothing to compute.
        } else if (StartValue.getBitWidth() > EndValue.getBitWidth())
          EndValue = EndValue.extend(StartValue.getBitWidth());
        else if (StartValue.getBitWidth() < EndValue.getBitWidth())
          StartValue = StartValue.extend(EndValue.getBitWidth());

        if (!StartDependent && !EndDependent && EndValue < StartValue) {
          Diag(D.getEllipsisLoc(), diag::err_array_designator_empty_range)
            << StartValue.toString(10) << EndValue.toString(10)
            << StartIndex->getSourceRange() << EndIndex->getSourceRange();
          Invalid = true;
        } else {
          Designators.push_back(ASTDesignator(InitExpressions.size(),
                                              D.getLBracketLoc(),
                                              D.getEllipsisLoc(),
                                              D.getRBracketLoc()));
          InitExpressions.push_back(StartIndex);
          InitExpressions.push_back(EndIndex);
        }
      }
      break;
    }
    }
  }

  if (Invalid || Init.isInvalid())
    return ExprError();

  // Clear out the expressions within the designation.
  Desig.ClearExprs(*this);

  DesignatedInitExpr *DIE
    = DesignatedInitExpr::Create(Context,
                                 Designators.data(), Designators.size(),
                                 InitExpressions, Loc, GNUSyntax,
                                 Init.getAs<Expr>());

  if (!getLangOpts().C99)
    Diag(DIE->getLocStart(), diag::ext_designated_init)
      << DIE->getSourceRange();

  return DIE;
}

//===----------------------------------------------------------------------===//
// Initialization entity
//===----------------------------------------------------------------------===//

InitializedEntity::InitializedEntity(ASTContext &Context, unsigned Index,
                                     const InitializedEntity &Parent)
  : Parent(&Parent), Index(Index)
{
  if (const ArrayType *AT = Context.getAsArrayType(Parent.getType())) {
    Kind = EK_ArrayElement;
    Type = AT->getElementType();
  } else if (const VectorType *VT = Parent.getType()->getAs<VectorType>()) {
    Kind = EK_VectorElement;
    Type = VT->getElementType();
  } else {
    const ComplexType *CT = Parent.getType()->getAs<ComplexType>();
    assert(CT && "Unexpected type");
    Kind = EK_ComplexElement;
    Type = CT->getElementType();
  }
}

InitializedEntity
InitializedEntity::InitializeBase(ASTContext &Context,
                                  const CXXBaseSpecifier *Base,
                                  bool IsInheritedVirtualBase) {
  InitializedEntity Result;
  Result.Kind = EK_Base;
  Result.Parent = nullptr;
  Result.Base = reinterpret_cast<uintptr_t>(Base);
  if (IsInheritedVirtualBase)
    Result.Base |= 0x01;

  Result.Type = Base->getType();
  return Result;
}

DeclarationName InitializedEntity::getName() const {
  switch (getKind()) {
  case EK_Parameter:
  case EK_Parameter_CF_Audited: {
    ParmVarDecl *D = reinterpret_cast<ParmVarDecl*>(Parameter & ~0x1);
    return (D ? D->getDeclName() : DeclarationName());
  }

  case EK_Variable:
  case EK_Member:
    return VariableOrMember->getDeclName();

  case EK_LambdaCapture:
    return DeclarationName(Capture.VarID);
      
  case EK_Result:
  case EK_Exception:
  case EK_New:
  case EK_Temporary:
  case EK_Base:
  case EK_Delegating:
  case EK_ArrayElement:
  case EK_VectorElement:
  case EK_ComplexElement:
  case EK_BlockElement:
  case EK_CompoundLiteralInit:
  case EK_RelatedResult:
    return DeclarationName();
  }

  llvm_unreachable("Invalid EntityKind!");
}

DeclaratorDecl *InitializedEntity::getDecl() const {
  switch (getKind()) {
  case EK_Variable:
  case EK_Member:
    return VariableOrMember;

  case EK_Parameter:
  case EK_Parameter_CF_Audited:
    return reinterpret_cast<ParmVarDecl*>(Parameter & ~0x1);

  case EK_Result:
  case EK_Exception:
  case EK_New:
  case EK_Temporary:
  case EK_Base:
  case EK_Delegating:
  case EK_ArrayElement:
  case EK_VectorElement:
  case EK_ComplexElement:
  case EK_BlockElement:
  case EK_LambdaCapture:
  case EK_CompoundLiteralInit:
  case EK_RelatedResult:
    return nullptr;
  }

  llvm_unreachable("Invalid EntityKind!");
}

bool InitializedEntity::allowsNRVO() const {
  switch (getKind()) {
  case EK_Result:
  case EK_Exception:
    return LocAndNRVO.NRVO;

  case EK_Variable:
  case EK_Parameter:
  case EK_Parameter_CF_Audited:
  case EK_Member:
  case EK_New:
  case EK_Temporary:
  case EK_CompoundLiteralInit:
  case EK_Base:
  case EK_Delegating:
  case EK_ArrayElement:
  case EK_VectorElement:
  case EK_ComplexElement:
  case EK_BlockElement:
  case EK_LambdaCapture:
  case EK_RelatedResult:
    break;
  }

  return false;
}

unsigned InitializedEntity::dumpImpl(raw_ostream &OS) const {
  assert(getParent() != this);
  unsigned Depth = getParent() ? getParent()->dumpImpl(OS) : 0;
  for (unsigned I = 0; I != Depth; ++I)
    OS << "`-";

  switch (getKind()) {
  case EK_Variable: OS << "Variable"; break;
  case EK_Parameter: OS << "Parameter"; break;
  case EK_Parameter_CF_Audited: OS << "CF audited function Parameter";
    break;
  case EK_Result: OS << "Result"; break;
  case EK_Exception: OS << "Exception"; break;
  case EK_Member: OS << "Member"; break;
  case EK_New: OS << "New"; break;
  case EK_Temporary: OS << "Temporary"; break;
  case EK_CompoundLiteralInit: OS << "CompoundLiteral";break;
  case EK_RelatedResult: OS << "RelatedResult"; break;
  case EK_Base: OS << "Base"; break;
  case EK_Delegating: OS << "Delegating"; break;
  case EK_ArrayElement: OS << "ArrayElement " << Index; break;
  case EK_VectorElement: OS << "VectorElement " << Index; break;
  case EK_ComplexElement: OS << "ComplexElement " << Index; break;
  case EK_BlockElement: OS << "Block"; break;
  case EK_LambdaCapture:
    OS << "LambdaCapture ";
    OS << DeclarationName(Capture.VarID);
    break;
  }

  if (Decl *D = getDecl()) {
    OS << " ";
    cast<NamedDecl>(D)->printQualifiedName(OS);
  }

  OS << " '" << getType().getAsString() << "'\n";

  return Depth + 1;
}

LLVM_DUMP_METHOD void InitializedEntity::dump() const {
  dumpImpl(llvm::errs());
}

//===----------------------------------------------------------------------===//
// Initialization sequence
//===----------------------------------------------------------------------===//

void InitializationSequence::Step::Destroy() {
  switch (Kind) {
  case SK_ResolveAddressOfOverloadedFunction:
  case SK_CastDerivedToBaseRValue:
  case SK_CastDerivedToBaseXValue:
  case SK_CastDerivedToBaseLValue:
  case SK_BindReference:
  case SK_BindReferenceToTemporary:
  case SK_ExtraneousCopyToTemporary:
  case SK_UserConversion:
  case SK_QualificationConversionRValue:
  case SK_QualificationConversionXValue:
  case SK_QualificationConversionLValue:
  case SK_AtomicConversion:
  case SK_LValueToRValue:
  case SK_ListInitialization:
  case SK_UnwrapInitList:
  case SK_RewrapInitList:
  case SK_ConstructorInitialization:
  case SK_ConstructorInitializationFromList:
  case SK_ZeroInitialization:
  case SK_CAssignment:
  case SK_StringInit:
  case SK_ObjCObjectConversion:
  case SK_ArrayInit:
  case SK_ParenthesizedArrayInit:
  case SK_PassByIndirectCopyRestore:
  case SK_PassByIndirectRestore:
  case SK_ProduceObjCObject:
  case SK_StdInitializerList:
  case SK_StdInitializerListConstructorCall:
  case SK_OCLSamplerInit:
  case SK_OCLZeroEvent:
    break;

  case SK_ConversionSequence:
  case SK_ConversionSequenceNoNarrowing:
    delete ICS;
  }
}

bool InitializationSequence::isDirectReferenceBinding() const {
  return !Steps.empty() && Steps.back().Kind == SK_BindReference;
}

bool InitializationSequence::isAmbiguous() const {
  if (!Failed())
    return false;

  switch (getFailureKind()) {
  case FK_TooManyInitsForReference:
  case FK_ArrayNeedsInitList:
  case FK_ArrayNeedsInitListOrStringLiteral:
  case FK_ArrayNeedsInitListOrWideStringLiteral:
  case FK_NarrowStringIntoWideCharArray:
  case FK_WideStringIntoCharArray:
  case FK_IncompatWideStringIntoWideChar:
  case FK_AddressOfOverloadFailed: // FIXME: Could do better
  case FK_NonConstLValueReferenceBindingToTemporary:
  case FK_NonConstLValueReferenceBindingToUnrelated:
  case FK_RValueReferenceBindingToLValue:
  case FK_ReferenceInitDropsQualifiers:
  case FK_ReferenceInitFailed:
  case FK_ConversionFailed:
  case FK_ConversionFromPropertyFailed:
  case FK_TooManyInitsForScalar:
  case FK_ReferenceBindingToInitList:
  case FK_InitListBadDestinationType:
  case FK_DefaultInitOfConst:
  case FK_Incomplete:
  case FK_ArrayTypeMismatch:
  case FK_NonConstantArrayInit:
  case FK_ListInitializationFailed:
  case FK_VariableLengthArrayHasInitializer:
  case FK_PlaceholderType:
  case FK_ExplicitConstructor:
  case FK_AddressOfUnaddressableFunction:
    return false;

  case FK_ReferenceInitOverloadFailed:
  case FK_UserConversionOverloadFailed:
  case FK_ConstructorOverloadFailed:
  case FK_ListConstructorOverloadFailed:
    return FailedOverloadResult == OR_Ambiguous;
  }

  llvm_unreachable("Invalid EntityKind!");
}

bool InitializationSequence::isConstructorInitialization() const {
  return !Steps.empty() && Steps.back().Kind == SK_ConstructorInitialization;
}

void
InitializationSequence
::AddAddressOverloadResolutionStep(FunctionDecl *Function,
                                   DeclAccessPair Found,
                                   bool HadMultipleCandidates) {
  Step S;
  S.Kind = SK_ResolveAddressOfOverloadedFunction;
  S.Type = Function->getType();
  S.Function.HadMultipleCandidates = HadMultipleCandidates;
  S.Function.Function = Function;
  S.Function.FoundDecl = Found;
  Steps.push_back(S);
}

void InitializationSequence::AddDerivedToBaseCastStep(QualType BaseType,
                                                      ExprValueKind VK) {
  Step S;
  switch (VK) {
  case VK_RValue: S.Kind = SK_CastDerivedToBaseRValue; break;
  case VK_XValue: S.Kind = SK_CastDerivedToBaseXValue; break;
  case VK_LValue: S.Kind = SK_CastDerivedToBaseLValue; break;
  }
  S.Type = BaseType;
  Steps.push_back(S);
}

void InitializationSequence::AddReferenceBindingStep(QualType T,
                                                     bool BindingTemporary) {
  Step S;
  S.Kind = BindingTemporary? SK_BindReferenceToTemporary : SK_BindReference;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddExtraneousCopyToTemporary(QualType T) {
  Step S;
  S.Kind = SK_ExtraneousCopyToTemporary;
  S.Type = T;
  Steps.push_back(S);
}

void
InitializationSequence::AddUserConversionStep(FunctionDecl *Function,
                                              DeclAccessPair FoundDecl,
                                              QualType T,
                                              bool HadMultipleCandidates) {
  Step S;
  S.Kind = SK_UserConversion;
  S.Type = T;
  S.Function.HadMultipleCandidates = HadMultipleCandidates;
  S.Function.Function = Function;
  S.Function.FoundDecl = FoundDecl;
  Steps.push_back(S);
}

void InitializationSequence::AddQualificationConversionStep(QualType Ty,
                                                            ExprValueKind VK) {
  Step S;
  S.Kind = SK_QualificationConversionRValue; // work around a gcc warning
  switch (VK) {
  case VK_RValue:
    S.Kind = SK_QualificationConversionRValue;
    break;
  case VK_XValue:
    S.Kind = SK_QualificationConversionXValue;
    break;
  case VK_LValue:
    S.Kind = SK_QualificationConversionLValue;
    break;
  }
  S.Type = Ty;
  Steps.push_back(S);
}

void InitializationSequence::AddAtomicConversionStep(QualType Ty) {
  Step S;
  S.Kind = SK_AtomicConversion;
  S.Type = Ty;
  Steps.push_back(S);
}

void InitializationSequence::AddLValueToRValueStep(QualType Ty) {
  assert(!Ty.hasQualifiers() && "rvalues may not have qualifiers");

  Step S;
  S.Kind = SK_LValueToRValue;
  S.Type = Ty;
  Steps.push_back(S);
}

void InitializationSequence::AddConversionSequenceStep(
    const ImplicitConversionSequence &ICS, QualType T,
    bool TopLevelOfInitList) {
  Step S;
  S.Kind = TopLevelOfInitList ? SK_ConversionSequenceNoNarrowing
                              : SK_ConversionSequence;
  S.Type = T;
  S.ICS = new ImplicitConversionSequence(ICS);
  Steps.push_back(S);
}

void InitializationSequence::AddListInitializationStep(QualType T) {
  Step S;
  S.Kind = SK_ListInitialization;
  S.Type = T;
  Steps.push_back(S);
}

void
InitializationSequence
::AddConstructorInitializationStep(CXXConstructorDecl *Constructor,
                                   AccessSpecifier Access,
                                   QualType T,
                                   bool HadMultipleCandidates,
                                   bool FromInitList, bool AsInitList) {
  Step S;
  S.Kind = FromInitList ? AsInitList ? SK_StdInitializerListConstructorCall
                                     : SK_ConstructorInitializationFromList
                        : SK_ConstructorInitialization;
  S.Type = T;
  S.Function.HadMultipleCandidates = HadMultipleCandidates;
  S.Function.Function = Constructor;
  S.Function.FoundDecl = DeclAccessPair::make(Constructor, Access);
  Steps.push_back(S);
}

void InitializationSequence::AddZeroInitializationStep(QualType T) {
  Step S;
  S.Kind = SK_ZeroInitialization;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddCAssignmentStep(QualType T) {
  Step S;
  S.Kind = SK_CAssignment;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddStringInitStep(QualType T) {
  Step S;
  S.Kind = SK_StringInit;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddObjCObjectConversionStep(QualType T) {
  Step S;
  S.Kind = SK_ObjCObjectConversion;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddArrayInitStep(QualType T) {
  Step S;
  S.Kind = SK_ArrayInit;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddParenthesizedArrayInitStep(QualType T) {
  Step S;
  S.Kind = SK_ParenthesizedArrayInit;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddPassByIndirectCopyRestoreStep(QualType type,
                                                              bool shouldCopy) {
  Step s;
  s.Kind = (shouldCopy ? SK_PassByIndirectCopyRestore
                       : SK_PassByIndirectRestore);
  s.Type = type;
  Steps.push_back(s);
}

void InitializationSequence::AddProduceObjCObjectStep(QualType T) {
  Step S;
  S.Kind = SK_ProduceObjCObject;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddStdInitializerListConstructionStep(QualType T) {
  Step S;
  S.Kind = SK_StdInitializerList;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddOCLSamplerInitStep(QualType T) {
  Step S;
  S.Kind = SK_OCLSamplerInit;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddOCLZeroEventStep(QualType T) {
  Step S;
  S.Kind = SK_OCLZeroEvent;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::RewrapReferenceInitList(QualType T,
                                                     InitListExpr *Syntactic) {
  assert(Syntactic->getNumInits() == 1 &&
         "Can only rewrap trivial init lists.");
  Step S;
  S.Kind = SK_UnwrapInitList;
  S.Type = Syntactic->getInit(0)->getType();
  Steps.insert(Steps.begin(), S);

  S.Kind = SK_RewrapInitList;
  S.Type = T;
  S.WrappingSyntacticList = Syntactic;
  Steps.push_back(S);
}

void InitializationSequence::SetOverloadFailure(FailureKind Failure,
                                                OverloadingResult Result) {
  setSequenceKind(FailedSequence);
  this->Failure = Failure;
  this->FailedOverloadResult = Result;
}

//===----------------------------------------------------------------------===//
// Attempt initialization
//===----------------------------------------------------------------------===//

/// Tries to add a zero initializer. Returns true if that worked.
static bool
maybeRecoverWithZeroInitialization(Sema &S, InitializationSequence &Sequence,
                                   const InitializedEntity &Entity) {
  if (Entity.getKind() != InitializedEntity::EK_Variable)
    return false;

  VarDecl *VD = cast<VarDecl>(Entity.getDecl());
  if (VD->getInit() || VD->getLocEnd().isMacroID())
    return false;

  QualType VariableTy = VD->getType().getCanonicalType();
  SourceLocation Loc = S.getLocForEndOfToken(VD->getLocEnd());
  std::string Init = S.getFixItZeroInitializerForType(VariableTy, Loc);
  if (!Init.empty()) {
    Sequence.AddZeroInitializationStep(Entity.getType());
    Sequence.SetZeroInitializationFixit(Init, Loc);
    return true;
  }
  return false;
}

static void MaybeProduceObjCObject(Sema &S,
                                   InitializationSequence &Sequence,
                                   const InitializedEntity &Entity) {
  if (!S.getLangOpts().ObjCAutoRefCount) return;

  /// When initializing a parameter, produce the value if it's marked
  /// __attribute__((ns_consumed)).
  if (Entity.isParameterKind()) {
    if (!Entity.isParameterConsumed())
      return;

    assert(Entity.getType()->isObjCRetainableType() &&
           "consuming an object of unretainable type?");
    Sequence.AddProduceObjCObjectStep(Entity.getType());

  /// When initializing a return value, if the return type is a
  /// retainable type, then returns need to immediately retain the
  /// object.  If an autorelease is required, it will be done at the
  /// last instant.
  } else if (Entity.getKind() == InitializedEntity::EK_Result) {
    if (!Entity.getType()->isObjCRetainableType())
      return;

    Sequence.AddProduceObjCObjectStep(Entity.getType());
  }
}

static void TryListInitialization(Sema &S,
                                  const InitializedEntity &Entity,
                                  const InitializationKind &Kind,
                                  InitListExpr *InitList,
                                  InitializationSequence &Sequence);

/// \brief When initializing from init list via constructor, handle
/// initialization of an object of type std::initializer_list<T>.
///
/// \return true if we have handled initialization of an object of type
/// std::initializer_list<T>, false otherwise.
static bool TryInitializerListConstruction(Sema &S,
                                           InitListExpr *List,
                                           QualType DestType,
                                           InitializationSequence &Sequence) {
  QualType E;
  if (!S.isStdInitializerList(DestType, &E))
    return false;

  if (!S.isCompleteType(List->getExprLoc(), E)) {
    Sequence.setIncompleteTypeFailure(E);
    return true;
  }

  // Try initializing a temporary array from the init list.
  QualType ArrayType = S.Context.getConstantArrayType(
      E.withConst(), llvm::APInt(S.Context.getTypeSize(S.Context.getSizeType()),
                                 List->getNumInits()),
      clang::ArrayType::Normal, 0);
  InitializedEntity HiddenArray =
      InitializedEntity::InitializeTemporary(ArrayType);
  InitializationKind Kind =
      InitializationKind::CreateDirectList(List->getExprLoc());
  TryListInitialization(S, HiddenArray, Kind, List, Sequence);
  if (Sequence)
    Sequence.AddStdInitializerListConstructionStep(DestType);
  return true;
}

static OverloadingResult
ResolveConstructorOverload(Sema &S, SourceLocation DeclLoc,
                           MultiExprArg Args,
                           OverloadCandidateSet &CandidateSet,
                           DeclContext::lookup_result Ctors,
                           OverloadCandidateSet::iterator &Best,
                           bool CopyInitializing, bool AllowExplicit,
                           bool OnlyListConstructors, bool IsListInit) {
  CandidateSet.clear();

  for (NamedDecl *D : Ctors) {
    DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());
    bool SuppressUserConversions = false;

    // Find the constructor (which may be a template).
    CXXConstructorDecl *Constructor = nullptr;
    FunctionTemplateDecl *ConstructorTmpl = dyn_cast<FunctionTemplateDecl>(D);
    if (ConstructorTmpl)
      Constructor = cast<CXXConstructorDecl>(
                                           ConstructorTmpl->getTemplatedDecl());
    else {
      Constructor = cast<CXXConstructorDecl>(D);

      // C++11 [over.best.ics]p4:
      //   ... and the constructor or user-defined conversion function is a
      //   candidate by
      //   - 13.3.1.3, when the argument is the temporary in the second step
      //     of a class copy-initialization, or
      //   - 13.3.1.4, 13.3.1.5, or 13.3.1.6 (in all cases),
      //   user-defined conversion sequences are not considered.
      // FIXME: This breaks backward compatibility, e.g. PR12117. As a
      //        temporary fix, let's re-instate the third bullet above until
      //        there is a resolution in the standard, i.e.,
      //   - 13.3.1.7 when the initializer list has exactly one element that is
      //     itself an initializer list and a conversion to some class X or
      //     reference to (possibly cv-qualified) X is considered for the first
      //     parameter of a constructor of X.
      if ((CopyInitializing ||
           (IsListInit && Args.size() == 1 && isa<InitListExpr>(Args[0]))) &&
          Constructor->isCopyOrMoveConstructor())
        SuppressUserConversions = true;
    }

    if (!Constructor->isInvalidDecl() &&
        (AllowExplicit || !Constructor->isExplicit()) &&
        (!OnlyListConstructors || S.isInitListConstructor(Constructor))) {
      if (ConstructorTmpl)
        S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                       /*ExplicitArgs*/ nullptr, Args,
                                       CandidateSet, SuppressUserConversions);
      else {
        // C++ [over.match.copy]p1:
        //   - When initializing a temporary to be bound to the first parameter 
        //     of a constructor that takes a reference to possibly cv-qualified 
        //     T as its first argument, called with a single argument in the 
        //     context of direct-initialization, explicit conversion functions
        //     are also considered.
        bool AllowExplicitConv = AllowExplicit && !CopyInitializing && 
                                 Args.size() == 1 &&
                                 Constructor->isCopyOrMoveConstructor();
        S.AddOverloadCandidate(Constructor, FoundDecl, Args, CandidateSet,
                               SuppressUserConversions,
                               /*PartialOverloading=*/false,
                               /*AllowExplicit=*/AllowExplicitConv);
      }
    }
  }

  // Perform overload resolution and return the result.
  return CandidateSet.BestViableFunction(S, DeclLoc, Best);
}

/// \brief Attempt initialization by constructor (C++ [dcl.init]), which
/// enumerates the constructors of the initialized entity and performs overload
/// resolution to select the best.
/// \param IsListInit     Is this list-initialization?
/// \param IsInitListCopy Is this non-list-initialization resulting from a
///                       list-initialization from {x} where x is the same
///                       type as the entity?
static void TryConstructorInitialization(Sema &S,
                                         const InitializedEntity &Entity,
                                         const InitializationKind &Kind,
                                         MultiExprArg Args, QualType DestType,
                                         InitializationSequence &Sequence,
                                         bool IsListInit = false,
                                         bool IsInitListCopy = false) {
  assert((!IsListInit || (Args.size() == 1 && isa<InitListExpr>(Args[0]))) &&
         "IsListInit must come with a single initializer list argument.");

  // The type we're constructing needs to be complete.
  if (!S.isCompleteType(Kind.getLocation(), DestType)) {
    Sequence.setIncompleteTypeFailure(DestType);
    return;
  }

  const RecordType *DestRecordType = DestType->getAs<RecordType>();
  assert(DestRecordType && "Constructor initialization requires record type");
  CXXRecordDecl *DestRecordDecl
    = cast<CXXRecordDecl>(DestRecordType->getDecl());

  // Build the candidate set directly in the initialization sequence
  // structure, so that it will persist if we fail.
  OverloadCandidateSet &CandidateSet = Sequence.getFailedCandidateSet();

  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = Kind.AllowExplicit() || IsListInit;
  bool CopyInitialization = Kind.getKind() == InitializationKind::IK_Copy;

  //   - Otherwise, if T is a class type, constructors are considered. The
  //     applicable constructors are enumerated, and the best one is chosen
  //     through overload resolution.
  DeclContext::lookup_result Ctors = S.LookupConstructors(DestRecordDecl);

  OverloadingResult Result = OR_No_Viable_Function;
  OverloadCandidateSet::iterator Best;
  bool AsInitializerList = false;

  // C++11 [over.match.list]p1, per DR1467:
  //   When objects of non-aggregate type T are list-initialized, such that
  //   8.5.4 [dcl.init.list] specifies that overload resolution is performed
  //   according to the rules in this section, overload resolution selects
  //   the constructor in two phases:
  //
  //   - Initially, the candidate functions are the initializer-list
  //     constructors of the class T and the argument list consists of the
  //     initializer list as a single argument.
  if (IsListInit) {
    InitListExpr *ILE = cast<InitListExpr>(Args[0]);
    AsInitializerList = true;

    // If the initializer list has no elements and T has a default constructor,
    // the first phase is omitted.
    if (ILE->getNumInits() != 0 || !DestRecordDecl->hasDefaultConstructor())
      Result = ResolveConstructorOverload(S, Kind.getLocation(), Args,
                                          CandidateSet, Ctors, Best,
                                          CopyInitialization, AllowExplicit,
                                          /*OnlyListConstructor=*/true,
                                          IsListInit);

    // Time to unwrap the init list.
    Args = MultiExprArg(ILE->getInits(), ILE->getNumInits());
  }

  // C++11 [over.match.list]p1:
  //   - If no viable initializer-list constructor is found, overload resolution
  //     is performed again, where the candidate functions are all the
  //     constructors of the class T and the argument list consists of the
  //     elements of the initializer list.
  if (Result == OR_No_Viable_Function) {
    AsInitializerList = false;
    Result = ResolveConstructorOverload(S, Kind.getLocation(), Args,
                                        CandidateSet, Ctors, Best,
                                        CopyInitialization, AllowExplicit,
                                        /*OnlyListConstructors=*/false,
                                        IsListInit);
  }
  if (Result) {
    Sequence.SetOverloadFailure(IsListInit ?
                      InitializationSequence::FK_ListConstructorOverloadFailed :
                      InitializationSequence::FK_ConstructorOverloadFailed,
                                Result);
    return;
  }

  // C++11 [dcl.init]p6:
  //   If a program calls for the default initialization of an object
  //   of a const-qualified type T, T shall be a class type with a
  //   user-provided default constructor.
  if (Kind.getKind() == InitializationKind::IK_Default &&
      Entity.getType().isConstQualified() &&
      !cast<CXXConstructorDecl>(Best->Function)->isUserProvided()) {
    if (!maybeRecoverWithZeroInitialization(S, Sequence, Entity))
      Sequence.SetFailed(InitializationSequence::FK_DefaultInitOfConst);
    return;
  }

  // C++11 [over.match.list]p1:
  //   In copy-list-initialization, if an explicit constructor is chosen, the
  //   initializer is ill-formed.
  CXXConstructorDecl *CtorDecl = cast<CXXConstructorDecl>(Best->Function);
  if (IsListInit && !Kind.AllowExplicit() && CtorDecl->isExplicit()) {
    Sequence.SetFailed(InitializationSequence::FK_ExplicitConstructor);
    return;
  }

  // Add the constructor initialization step. Any cv-qualification conversion is
  // subsumed by the initialization.
  bool HadMultipleCandidates = (CandidateSet.size() > 1);
  Sequence.AddConstructorInitializationStep(
      CtorDecl, Best->FoundDecl.getAccess(), DestType, HadMultipleCandidates,
      IsListInit | IsInitListCopy, AsInitializerList);
}

static bool
ResolveOverloadedFunctionForReferenceBinding(Sema &S,
                                             Expr *Initializer,
                                             QualType &SourceType,
                                             QualType &UnqualifiedSourceType,
                                             QualType UnqualifiedTargetType,
                                             InitializationSequence &Sequence) {
  if (S.Context.getCanonicalType(UnqualifiedSourceType) ==
        S.Context.OverloadTy) {
    DeclAccessPair Found;
    bool HadMultipleCandidates = false;
    if (FunctionDecl *Fn
        = S.ResolveAddressOfOverloadedFunction(Initializer,
                                               UnqualifiedTargetType,
                                               false, Found,
                                               &HadMultipleCandidates)) {
      Sequence.AddAddressOverloadResolutionStep(Fn, Found,
                                                HadMultipleCandidates);
      SourceType = Fn->getType();
      UnqualifiedSourceType = SourceType.getUnqualifiedType();
    } else if (!UnqualifiedTargetType->isRecordType()) {
      Sequence.SetFailed(InitializationSequence::FK_AddressOfOverloadFailed);
      return true;
    }
  }
  return false;
}

static void TryReferenceInitializationCore(Sema &S,
                                           const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           Expr *Initializer,
                                           QualType cv1T1, QualType T1,
                                           Qualifiers T1Quals,
                                           QualType cv2T2, QualType T2,
                                           Qualifiers T2Quals,
                                           InitializationSequence &Sequence);

static void TryValueInitialization(Sema &S,
                                   const InitializedEntity &Entity,
                                   const InitializationKind &Kind,
                                   InitializationSequence &Sequence,
                                   InitListExpr *InitList = nullptr);

/// \brief Attempt list initialization of a reference.
static void TryReferenceListInitialization(Sema &S,
                                           const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           InitListExpr *InitList,
                                           InitializationSequence &Sequence) {
  // First, catch C++03 where this isn't possible.
  if (!S.getLangOpts().CPlusPlus11) {
    Sequence.SetFailed(InitializationSequence::FK_ReferenceBindingToInitList);
    return;
  }
  // Can't reference initialize a compound literal.
  if (Entity.getKind() == InitializedEntity::EK_CompoundLiteralInit) {
    Sequence.SetFailed(InitializationSequence::FK_ReferenceBindingToInitList);
    return;
  }

  QualType DestType = Entity.getType();
  QualType cv1T1 = DestType->getAs<ReferenceType>()->getPointeeType();
  Qualifiers T1Quals;
  QualType T1 = S.Context.getUnqualifiedArrayType(cv1T1, T1Quals);

  // Reference initialization via an initializer list works thus:
  // If the initializer list consists of a single element that is
  // reference-related to the referenced type, bind directly to that element
  // (possibly creating temporaries).
  // Otherwise, initialize a temporary with the initializer list and
  // bind to that.
  if (InitList->getNumInits() == 1) {
    Expr *Initializer = InitList->getInit(0);
    QualType cv2T2 = Initializer->getType();
    Qualifiers T2Quals;
    QualType T2 = S.Context.getUnqualifiedArrayType(cv2T2, T2Quals);

    // If this fails, creating a temporary wouldn't work either.
    if (ResolveOverloadedFunctionForReferenceBinding(S, Initializer, cv2T2, T2,
                                                     T1, Sequence))
      return;

    SourceLocation DeclLoc = Initializer->getLocStart();
    bool dummy1, dummy2, dummy3;
    Sema::ReferenceCompareResult RefRelationship
      = S.CompareReferenceRelationship(DeclLoc, cv1T1, cv2T2, dummy1,
                                       dummy2, dummy3);
    if (RefRelationship >= Sema::Ref_Related) {
      // Try to bind the reference here.
      TryReferenceInitializationCore(S, Entity, Kind, Initializer, cv1T1, T1,
                                     T1Quals, cv2T2, T2, T2Quals, Sequence);
      if (Sequence)
        Sequence.RewrapReferenceInitList(cv1T1, InitList);
      return;
    }

    // Update the initializer if we've resolved an overloaded function.
    if (Sequence.step_begin() != Sequence.step_end())
      Sequence.RewrapReferenceInitList(cv1T1, InitList);
  }

  // Not reference-related. Create a temporary and bind to that.
  InitializedEntity TempEntity = InitializedEntity::InitializeTemporary(cv1T1);

  TryListInitialization(S, TempEntity, Kind, InitList, Sequence);
  if (Sequence) {
    if (DestType->isRValueReferenceType() ||
        (T1Quals.hasConst() && !T1Quals.hasVolatile()))
      Sequence.AddReferenceBindingStep(cv1T1, /*bindingTemporary=*/true);
    else
      Sequence.SetFailed(
          InitializationSequence::FK_NonConstLValueReferenceBindingToTemporary);
  }
}

/// \brief Attempt list initialization (C++0x [dcl.init.list])
static void TryListInitialization(Sema &S,
                                  const InitializedEntity &Entity,
                                  const InitializationKind &Kind,
                                  InitListExpr *InitList,
                                  InitializationSequence &Sequence) {
  QualType DestType = Entity.getType();

  // C++ doesn't allow scalar initialization with more than one argument.
  // But C99 complex numbers are scalars and it makes sense there.
  if (S.getLangOpts().CPlusPlus && DestType->isScalarType() &&
      !DestType->isAnyComplexType() && InitList->getNumInits() > 1) {
    Sequence.SetFailed(InitializationSequence::FK_TooManyInitsForScalar);
    return;
  }
  if (DestType->isReferenceType()) {
    TryReferenceListInitialization(S, Entity, Kind, InitList, Sequence);
    return;
  }

  if (DestType->isRecordType() &&
      !S.isCompleteType(InitList->getLocStart(), DestType)) {
    Sequence.setIncompleteTypeFailure(DestType);
    return;
  }

  // C++11 [dcl.init.list]p3, per DR1467:
  // - If T is a class type and the initializer list has a single element of
  //   type cv U, where U is T or a class derived from T, the object is
  //   initialized from that element (by copy-initialization for
  //   copy-list-initialization, or by direct-initialization for
  //   direct-list-initialization).
  // - Otherwise, if T is a character array and the initializer list has a
  //   single element that is an appropriately-typed string literal
  //   (8.5.2 [dcl.init.string]), initialization is performed as described
  //   in that section.
  // - Otherwise, if T is an aggregate, [...] (continue below).
  if (S.getLangOpts().CPlusPlus11 && InitList->getNumInits() == 1) {
    if (DestType->isRecordType()) {
      QualType InitType = InitList->getInit(0)->getType();
      if (S.Context.hasSameUnqualifiedType(InitType, DestType) ||
          S.IsDerivedFrom(InitList->getLocStart(), InitType, DestType)) {
        Expr *InitAsExpr = InitList->getInit(0);
        TryConstructorInitialization(S, Entity, Kind, InitAsExpr, DestType,
                                     Sequence, /*InitListSyntax*/ false,
                                     /*IsInitListCopy*/ true);
        return;
      }
    }
    if (const ArrayType *DestAT = S.Context.getAsArrayType(DestType)) {
      Expr *SubInit[1] = {InitList->getInit(0)};
      if (!isa<VariableArrayType>(DestAT) &&
          IsStringInit(SubInit[0], DestAT, S.Context) == SIF_None) {
        InitializationKind SubKind =
            Kind.getKind() == InitializationKind::IK_DirectList
                ? InitializationKind::CreateDirect(Kind.getLocation(),
                                                   InitList->getLBraceLoc(),
                                                   InitList->getRBraceLoc())
                : Kind;
        Sequence.InitializeFrom(S, Entity, SubKind, SubInit,
                                /*TopLevelOfInitList*/ true);

        // TryStringLiteralInitialization() (in InitializeFrom()) will fail if
        // the element is not an appropriately-typed string literal, in which
        // case we should proceed as in C++11 (below).
        if (Sequence) {
          Sequence.RewrapReferenceInitList(Entity.getType(), InitList);
          return;
        }
      }
    }
  }

  // C++11 [dcl.init.list]p3:
  //   - If T is an aggregate, aggregate initialization is performed.
  if ((DestType->isRecordType() && !DestType->isAggregateType()) ||
      (S.getLangOpts().CPlusPlus11 &&
       S.isStdInitializerList(DestType, nullptr))) {
    if (S.getLangOpts().CPlusPlus11) {
      //   - Otherwise, if the initializer list has no elements and T is a
      //     class type with a default constructor, the object is
      //     value-initialized.
      if (InitList->getNumInits() == 0) {
        CXXRecordDecl *RD = DestType->getAsCXXRecordDecl();
        if (RD->hasDefaultConstructor()) {
          TryValueInitialization(S, Entity, Kind, Sequence, InitList);
          return;
        }
      }

      //   - Otherwise, if T is a specialization of std::initializer_list<E>,
      //     an initializer_list object constructed [...]
      if (TryInitializerListConstruction(S, InitList, DestType, Sequence))
        return;

      //   - Otherwise, if T is a class type, constructors are considered.
      Expr *InitListAsExpr = InitList;
      TryConstructorInitialization(S, Entity, Kind, InitListAsExpr, DestType,
                                   Sequence, /*InitListSyntax*/ true);
    } else
      Sequence.SetFailed(InitializationSequence::FK_InitListBadDestinationType);
    return;
  }

  if (S.getLangOpts().CPlusPlus && !DestType->isAggregateType() &&
      InitList->getNumInits() == 1 &&
      InitList->getInit(0)->getType()->isRecordType()) {
    //   - Otherwise, if the initializer list has a single element of type E
    //     [...references are handled above...], the object or reference is
    //     initialized from that element (by copy-initialization for
    //     copy-list-initialization, or by direct-initialization for
    //     direct-list-initialization); if a narrowing conversion is required
    //     to convert the element to T, the program is ill-formed.
    //
    // Per core-24034, this is direct-initialization if we were performing
    // direct-list-initialization and copy-initialization otherwise.
    // We can't use InitListChecker for this, because it always performs
    // copy-initialization. This only matters if we might use an 'explicit'
    // conversion operator, so we only need to handle the cases where the source
    // is of record type.
    InitializationKind SubKind =
        Kind.getKind() == InitializationKind::IK_DirectList
            ? InitializationKind::CreateDirect(Kind.getLocation(),
                                               InitList->getLBraceLoc(),
                                               InitList->getRBraceLoc())
            : Kind;
    Expr *SubInit[1] = { InitList->getInit(0) };
    Sequence.InitializeFrom(S, Entity, SubKind, SubInit,
                            /*TopLevelOfInitList*/true);
    if (Sequence)
      Sequence.RewrapReferenceInitList(Entity.getType(), InitList);
    return;
  }

  InitListChecker CheckInitList(S, Entity, InitList,
          DestType, /*VerifyOnly=*/true);
  if (CheckInitList.HadError()) {
    Sequence.SetFailed(InitializationSequence::FK_ListInitializationFailed);
    return;
  }

  // Add the list initialization step with the built init list.
  Sequence.AddListInitializationStep(DestType);
}

/// \brief Try a reference initialization that involves calling a conversion
/// function.
static OverloadingResult TryRefInitWithConversionFunction(Sema &S,
                                             const InitializedEntity &Entity,
                                             const InitializationKind &Kind,
                                             Expr *Initializer,
                                             bool AllowRValues,
                                             InitializationSequence &Sequence) {
  QualType DestType = Entity.getType();
  QualType cv1T1 = DestType->getAs<ReferenceType>()->getPointeeType();
  QualType T1 = cv1T1.getUnqualifiedType();
  QualType cv2T2 = Initializer->getType();
  QualType T2 = cv2T2.getUnqualifiedType();

  bool DerivedToBase;
  bool ObjCConversion;
  bool ObjCLifetimeConversion;
  assert(!S.CompareReferenceRelationship(Initializer->getLocStart(),
                                         T1, T2, DerivedToBase,
                                         ObjCConversion,
                                         ObjCLifetimeConversion) &&
         "Must have incompatible references when binding via conversion");
  (void)DerivedToBase;
  (void)ObjCConversion;
  (void)ObjCLifetimeConversion;
  
  // Build the candidate set directly in the initialization sequence
  // structure, so that it will persist if we fail.
  OverloadCandidateSet &CandidateSet = Sequence.getFailedCandidateSet();
  CandidateSet.clear();

  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = Kind.AllowExplicit();
  bool AllowExplicitConvs = Kind.allowExplicitConversionFunctionsInRefBinding();

  const RecordType *T1RecordType = nullptr;
  if (AllowRValues && (T1RecordType = T1->getAs<RecordType>()) &&
      S.isCompleteType(Kind.getLocation(), T1)) {
    // The type we're converting to is a class type. Enumerate its constructors
    // to see if there is a suitable conversion.
    CXXRecordDecl *T1RecordDecl = cast<CXXRecordDecl>(T1RecordType->getDecl());

    for (NamedDecl *D : S.LookupConstructors(T1RecordDecl)) {
      DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());

      // Find the constructor (which may be a template).
      CXXConstructorDecl *Constructor = nullptr;
      FunctionTemplateDecl *ConstructorTmpl = dyn_cast<FunctionTemplateDecl>(D);
      if (ConstructorTmpl)
        Constructor = cast<CXXConstructorDecl>(
                                         ConstructorTmpl->getTemplatedDecl());
      else
        Constructor = cast<CXXConstructorDecl>(D);

      if (!Constructor->isInvalidDecl() &&
          Constructor->isConvertingConstructor(AllowExplicit)) {
        if (ConstructorTmpl)
          S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                         /*ExplicitArgs*/ nullptr,
                                         Initializer, CandidateSet,
                                         /*SuppressUserConversions=*/true);
        else
          S.AddOverloadCandidate(Constructor, FoundDecl,
                                 Initializer, CandidateSet,
                                 /*SuppressUserConversions=*/true);
      }
    }
  }
  if (T1RecordType && T1RecordType->getDecl()->isInvalidDecl())
    return OR_No_Viable_Function;

  const RecordType *T2RecordType = nullptr;
  if ((T2RecordType = T2->getAs<RecordType>()) &&
      S.isCompleteType(Kind.getLocation(), T2)) {
    // The type we're converting from is a class type, enumerate its conversion
    // functions.
    CXXRecordDecl *T2RecordDecl = cast<CXXRecordDecl>(T2RecordType->getDecl());

    const auto &Conversions = T2RecordDecl->getVisibleConversionFunctions();
    for (auto I = Conversions.begin(), E = Conversions.end(); I != E; ++I) {
      NamedDecl *D = *I;
      CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(D->getDeclContext());
      if (isa<UsingShadowDecl>(D))
        D = cast<UsingShadowDecl>(D)->getTargetDecl();

      FunctionTemplateDecl *ConvTemplate = dyn_cast<FunctionTemplateDecl>(D);
      CXXConversionDecl *Conv;
      if (ConvTemplate)
        Conv = cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
      else
        Conv = cast<CXXConversionDecl>(D);

      // If the conversion function doesn't return a reference type,
      // it can't be considered for this conversion unless we're allowed to
      // consider rvalues.
      // FIXME: Do we need to make sure that we only consider conversion
      // candidates with reference-compatible results? That might be needed to
      // break recursion.
      if ((AllowExplicitConvs || !Conv->isExplicit()) &&
          (AllowRValues || Conv->getConversionType()->isLValueReferenceType())){
        if (ConvTemplate)
          S.AddTemplateConversionCandidate(ConvTemplate, I.getPair(),
                                           ActingDC, Initializer,
                                           DestType, CandidateSet,
                                           /*AllowObjCConversionOnExplicit=*/
                                             false);
        else
          S.AddConversionCandidate(Conv, I.getPair(), ActingDC,
                                   Initializer, DestType, CandidateSet,
                                   /*AllowObjCConversionOnExplicit=*/false);
      }
    }
  }
  if (T2RecordType && T2RecordType->getDecl()->isInvalidDecl())
    return OR_No_Viable_Function;

  SourceLocation DeclLoc = Initializer->getLocStart();

  // Perform overload resolution. If it fails, return the failed result.
  OverloadCandidateSet::iterator Best;
  if (OverloadingResult Result
        = CandidateSet.BestViableFunction(S, DeclLoc, Best, true))
    return Result;

  FunctionDecl *Function = Best->Function;
  // This is the overload that will be used for this initialization step if we
  // use this initialization. Mark it as referenced.
  Function->setReferenced();

  // Compute the returned type of the conversion.
  if (isa<CXXConversionDecl>(Function))
    T2 = Function->getReturnType();
  else
    T2 = cv1T1;

  // Add the user-defined conversion step.
  bool HadMultipleCandidates = (CandidateSet.size() > 1);
  Sequence.AddUserConversionStep(Function, Best->FoundDecl,
                                 T2.getNonLValueExprType(S.Context),
                                 HadMultipleCandidates);

  // Determine whether we need to perform derived-to-base or
  // cv-qualification adjustments.
  ExprValueKind VK = VK_RValue;
  if (T2->isLValueReferenceType())
    VK = VK_LValue;
  else if (const RValueReferenceType *RRef = T2->getAs<RValueReferenceType>())
    VK = RRef->getPointeeType()->isFunctionType() ? VK_LValue : VK_XValue;

  bool NewDerivedToBase = false;
  bool NewObjCConversion = false;
  bool NewObjCLifetimeConversion = false;
  Sema::ReferenceCompareResult NewRefRelationship
    = S.CompareReferenceRelationship(DeclLoc, T1,
                                     T2.getNonLValueExprType(S.Context),
                                     NewDerivedToBase, NewObjCConversion,
                                     NewObjCLifetimeConversion);
  if (NewRefRelationship == Sema::Ref_Incompatible) {
    // If the type we've converted to is not reference-related to the
    // type we're looking for, then there is another conversion step
    // we need to perform to produce a temporary of the right type
    // that we'll be binding to.
    ImplicitConversionSequence ICS;
    ICS.setStandard();
    ICS.Standard = Best->FinalConversion;
    T2 = ICS.Standard.getToType(2);
    Sequence.AddConversionSequenceStep(ICS, T2);
  } else if (NewDerivedToBase)
    Sequence.AddDerivedToBaseCastStep(
                                S.Context.getQualifiedType(T1,
                                  T2.getNonReferenceType().getQualifiers()),
                                      VK);
  else if (NewObjCConversion)
    Sequence.AddObjCObjectConversionStep(
                                S.Context.getQualifiedType(T1,
                                  T2.getNonReferenceType().getQualifiers()));

  if (cv1T1.getQualifiers() != T2.getNonReferenceType().getQualifiers())
    Sequence.AddQualificationConversionStep(cv1T1, VK);

  Sequence.AddReferenceBindingStep(cv1T1, !T2->isReferenceType());
  return OR_Success;
}

static void CheckCXX98CompatAccessibleCopy(Sema &S,
                                           const InitializedEntity &Entity,
                                           Expr *CurInitExpr);

/// \brief Attempt reference initialization (C++0x [dcl.init.ref])
static void TryReferenceInitialization(Sema &S,
                                       const InitializedEntity &Entity,
                                       const InitializationKind &Kind,
                                       Expr *Initializer,
                                       InitializationSequence &Sequence) {
  QualType DestType = Entity.getType();
  QualType cv1T1 = DestType->getAs<ReferenceType>()->getPointeeType();
  Qualifiers T1Quals;
  QualType T1 = S.Context.getUnqualifiedArrayType(cv1T1, T1Quals);
  QualType cv2T2 = Initializer->getType();
  Qualifiers T2Quals;
  QualType T2 = S.Context.getUnqualifiedArrayType(cv2T2, T2Quals);

  // If the initializer is the address of an overloaded function, try
  // to resolve the overloaded function. If all goes well, T2 is the
  // type of the resulting function.
  if (ResolveOverloadedFunctionForReferenceBinding(S, Initializer, cv2T2, T2,
                                                   T1, Sequence))
    return;

  // Delegate everything else to a subfunction.
  TryReferenceInitializationCore(S, Entity, Kind, Initializer, cv1T1, T1,
                                 T1Quals, cv2T2, T2, T2Quals, Sequence);
}

/// Converts the target of reference initialization so that it has the
/// appropriate qualifiers and value kind.
///
/// In this case, 'x' is an 'int' lvalue, but it needs to be 'const int'.
/// \code
///   int x;
///   const int &r = x;
/// \endcode
///
/// In this case the reference is binding to a bitfield lvalue, which isn't
/// valid. Perform a load to create a lifetime-extended temporary instead.
/// \code
///   const int &r = someStruct.bitfield;
/// \endcode
static ExprValueKind
convertQualifiersAndValueKindIfNecessary(Sema &S,
                                         InitializationSequence &Sequence,
                                         Expr *Initializer,
                                         QualType cv1T1,
                                         Qualifiers T1Quals,
                                         Qualifiers T2Quals,
                                         bool IsLValueRef) {
  bool IsNonAddressableType = Initializer->refersToBitField() ||
                              Initializer->refersToVectorElement();

  if (IsNonAddressableType) {
    // C++11 [dcl.init.ref]p5: [...] Otherwise, the reference shall be an
    // lvalue reference to a non-volatile const type, or the reference shall be
    // an rvalue reference.
    //
    // If not, we can't make a temporary and bind to that. Give up and allow the
    // error to be diagnosed later.
    if (IsLValueRef && (!T1Quals.hasConst() || T1Quals.hasVolatile())) {
      assert(Initializer->isGLValue());
      return Initializer->getValueKind();
    }

    // Force a load so we can materialize a temporary.
    Sequence.AddLValueToRValueStep(cv1T1.getUnqualifiedType());
    return VK_RValue;
  }

  if (T1Quals != T2Quals) {
    Sequence.AddQualificationConversionStep(cv1T1,
                                            Initializer->getValueKind());
  }

  return Initializer->getValueKind();
}


/// \brief Reference initialization without resolving overloaded functions.
static void TryReferenceInitializationCore(Sema &S,
                                           const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           Expr *Initializer,
                                           QualType cv1T1, QualType T1,
                                           Qualifiers T1Quals,
                                           QualType cv2T2, QualType T2,
                                           Qualifiers T2Quals,
                                           InitializationSequence &Sequence) {
  QualType DestType = Entity.getType();
  SourceLocation DeclLoc = Initializer->getLocStart();
  // Compute some basic properties of the types and the initializer.
  bool isLValueRef = DestType->isLValueReferenceType();
  bool isRValueRef = !isLValueRef;
  bool DerivedToBase = false;
  bool ObjCConversion = false;
  bool ObjCLifetimeConversion = false;
  Expr::Classification InitCategory = Initializer->Classify(S.Context);
  Sema::ReferenceCompareResult RefRelationship
    = S.CompareReferenceRelationship(DeclLoc, cv1T1, cv2T2, DerivedToBase,
                                     ObjCConversion, ObjCLifetimeConversion);

  // C++0x [dcl.init.ref]p5:
  //   A reference to type "cv1 T1" is initialized by an expression of type
  //   "cv2 T2" as follows:
  //
  //     - If the reference is an lvalue reference and the initializer
  //       expression
  // Note the analogous bullet points for rvalue refs to functions. Because
  // there are no function rvalues in C++, rvalue refs to functions are treated
  // like lvalue refs.
  OverloadingResult ConvOvlResult = OR_Success;
  bool T1Function = T1->isFunctionType();
  if (isLValueRef || T1Function) {
    if (InitCategory.isLValue() &&
        (RefRelationship >= Sema::Ref_Compatible_With_Added_Qualification ||
         (Kind.isCStyleOrFunctionalCast() &&
          RefRelationship == Sema::Ref_Related))) {
      //   - is an lvalue (but is not a bit-field), and "cv1 T1" is
      //     reference-compatible with "cv2 T2," or
      //
      // Per C++ [over.best.ics]p2, we don't diagnose whether the lvalue is a
      // bit-field when we're determining whether the reference initialization
      // can occur. However, we do pay attention to whether it is a bit-field
      // to decide whether we're actually binding to a temporary created from
      // the bit-field.
      if (DerivedToBase)
        Sequence.AddDerivedToBaseCastStep(
                         S.Context.getQualifiedType(T1, T2Quals),
                         VK_LValue);
      else if (ObjCConversion)
        Sequence.AddObjCObjectConversionStep(
                                     S.Context.getQualifiedType(T1, T2Quals));

      ExprValueKind ValueKind =
        convertQualifiersAndValueKindIfNecessary(S, Sequence, Initializer,
                                                 cv1T1, T1Quals, T2Quals,
                                                 isLValueRef);
      Sequence.AddReferenceBindingStep(cv1T1, ValueKind == VK_RValue);
      return;
    }

    //     - has a class type (i.e., T2 is a class type), where T1 is not
    //       reference-related to T2, and can be implicitly converted to an
    //       lvalue of type "cv3 T3," where "cv1 T1" is reference-compatible
    //       with "cv3 T3" (this conversion is selected by enumerating the
    //       applicable conversion functions (13.3.1.6) and choosing the best
    //       one through overload resolution (13.3)),
    // If we have an rvalue ref to function type here, the rhs must be
    // an rvalue. DR1287 removed the "implicitly" here.
    if (RefRelationship == Sema::Ref_Incompatible && T2->isRecordType() &&
        (isLValueRef || InitCategory.isRValue())) {
      ConvOvlResult = TryRefInitWithConversionFunction(
          S, Entity, Kind, Initializer, /*AllowRValues*/isRValueRef, Sequence);
      if (ConvOvlResult == OR_Success)
        return;
      if (ConvOvlResult != OR_No_Viable_Function)
        Sequence.SetOverloadFailure(
            InitializationSequence::FK_ReferenceInitOverloadFailed,
            ConvOvlResult);
    }
  }

  //     - Otherwise, the reference shall be an lvalue reference to a
  //       non-volatile const type (i.e., cv1 shall be const), or the reference
  //       shall be an rvalue reference.
  if (isLValueRef && !(T1Quals.hasConst() && !T1Quals.hasVolatile())) {
    if (S.Context.getCanonicalType(T2) == S.Context.OverloadTy)
      Sequence.SetFailed(InitializationSequence::FK_AddressOfOverloadFailed);
    else if (ConvOvlResult && !Sequence.getFailedCandidateSet().empty())
      Sequence.SetOverloadFailure(
                        InitializationSequence::FK_ReferenceInitOverloadFailed,
                                  ConvOvlResult);
    else
      Sequence.SetFailed(InitCategory.isLValue()
        ? (RefRelationship == Sema::Ref_Related
             ? InitializationSequence::FK_ReferenceInitDropsQualifiers
             : InitializationSequence::FK_NonConstLValueReferenceBindingToUnrelated)
        : InitializationSequence::FK_NonConstLValueReferenceBindingToTemporary);

    return;
  }

  //    - If the initializer expression
  //      - is an xvalue, class prvalue, array prvalue, or function lvalue and
  //        "cv1 T1" is reference-compatible with "cv2 T2"
  // Note: functions are handled below.
  if (!T1Function &&
      (RefRelationship >= Sema::Ref_Compatible_With_Added_Qualification ||
       (Kind.isCStyleOrFunctionalCast() &&
        RefRelationship == Sema::Ref_Related)) &&
      (InitCategory.isXValue() ||
       (InitCategory.isPRValue() && T2->isRecordType()) ||
       (InitCategory.isPRValue() && T2->isArrayType()))) {
    ExprValueKind ValueKind = InitCategory.isXValue()? VK_XValue : VK_RValue;
    if (InitCategory.isPRValue() && T2->isRecordType()) {
      // The corresponding bullet in C++03 [dcl.init.ref]p5 gives the
      // compiler the freedom to perform a copy here or bind to the
      // object, while C++0x requires that we bind directly to the
      // object. Hence, we always bind to the object without making an
      // extra copy. However, in C++03 requires that we check for the
      // presence of a suitable copy constructor:
      //
      //   The constructor that would be used to make the copy shall
      //   be callable whether or not the copy is actually done.
      if (!S.getLangOpts().CPlusPlus11 && !S.getLangOpts().MicrosoftExt)
        Sequence.AddExtraneousCopyToTemporary(cv2T2);
      else if (S.getLangOpts().CPlusPlus11)
        CheckCXX98CompatAccessibleCopy(S, Entity, Initializer);
    }

    if (DerivedToBase)
      Sequence.AddDerivedToBaseCastStep(S.Context.getQualifiedType(T1, T2Quals),
                                        ValueKind);
    else if (ObjCConversion)
      Sequence.AddObjCObjectConversionStep(
                                       S.Context.getQualifiedType(T1, T2Quals));

    ValueKind = convertQualifiersAndValueKindIfNecessary(S, Sequence,
                                                         Initializer, cv1T1,
                                                         T1Quals, T2Quals,
                                                         isLValueRef);

    Sequence.AddReferenceBindingStep(cv1T1, ValueKind == VK_RValue);
    return;
  }

  //       - has a class type (i.e., T2 is a class type), where T1 is not
  //         reference-related to T2, and can be implicitly converted to an
  //         xvalue, class prvalue, or function lvalue of type "cv3 T3",
  //         where "cv1 T1" is reference-compatible with "cv3 T3",
  //
  // DR1287 removes the "implicitly" here.
  if (T2->isRecordType()) {
    if (RefRelationship == Sema::Ref_Incompatible) {
      ConvOvlResult = TryRefInitWithConversionFunction(
          S, Entity, Kind, Initializer, /*AllowRValues*/true, Sequence);
      if (ConvOvlResult)
        Sequence.SetOverloadFailure(
            InitializationSequence::FK_ReferenceInitOverloadFailed,
            ConvOvlResult);

      return;
    }

    if ((RefRelationship == Sema::Ref_Compatible ||
         RefRelationship == Sema::Ref_Compatible_With_Added_Qualification) &&
        isRValueRef && InitCategory.isLValue()) {
      Sequence.SetFailed(
        InitializationSequence::FK_RValueReferenceBindingToLValue);
      return;
    }

    Sequence.SetFailed(InitializationSequence::FK_ReferenceInitDropsQualifiers);
    return;
  }

  //      - Otherwise, a temporary of type "cv1 T1" is created and initialized
  //        from the initializer expression using the rules for a non-reference
  //        copy-initialization (8.5). The reference is then bound to the
  //        temporary. [...]

  InitializedEntity TempEntity = InitializedEntity::InitializeTemporary(cv1T1);

  // FIXME: Why do we use an implicit conversion here rather than trying
  // copy-initialization?
  ImplicitConversionSequence ICS
    = S.TryImplicitConversion(Initializer, TempEntity.getType(),
                              /*SuppressUserConversions=*/false,
                              /*AllowExplicit=*/false,
                              /*FIXME:InOverloadResolution=*/false,
                              /*CStyle=*/Kind.isCStyleOrFunctionalCast(),
                              /*AllowObjCWritebackConversion=*/false);
  
  if (ICS.isBad()) {
    // FIXME: Use the conversion function set stored in ICS to turn
    // this into an overloading ambiguity diagnostic. However, we need
    // to keep that set as an OverloadCandidateSet rather than as some
    // other kind of set.
    if (ConvOvlResult && !Sequence.getFailedCandidateSet().empty())
      Sequence.SetOverloadFailure(
                        InitializationSequence::FK_ReferenceInitOverloadFailed,
                                  ConvOvlResult);
    else if (S.Context.getCanonicalType(T2) == S.Context.OverloadTy)
      Sequence.SetFailed(InitializationSequence::FK_AddressOfOverloadFailed);
    else
      Sequence.SetFailed(InitializationSequence::FK_ReferenceInitFailed);
    return;
  } else {
    Sequence.AddConversionSequenceStep(ICS, TempEntity.getType());
  }

  //        [...] If T1 is reference-related to T2, cv1 must be the
  //        same cv-qualification as, or greater cv-qualification
  //        than, cv2; otherwise, the program is ill-formed.
  unsigned T1CVRQuals = T1Quals.getCVRQualifiers();
  unsigned T2CVRQuals = T2Quals.getCVRQualifiers();
  if (RefRelationship == Sema::Ref_Related &&
      (T1CVRQuals | T2CVRQuals) != T1CVRQuals) {
    Sequence.SetFailed(InitializationSequence::FK_ReferenceInitDropsQualifiers);
    return;
  }

  //   [...] If T1 is reference-related to T2 and the reference is an rvalue
  //   reference, the initializer expression shall not be an lvalue.
  if (RefRelationship >= Sema::Ref_Related && !isLValueRef &&
      InitCategory.isLValue()) {
    Sequence.SetFailed(
                    InitializationSequence::FK_RValueReferenceBindingToLValue);
    return;
  }

  Sequence.AddReferenceBindingStep(cv1T1, /*bindingTemporary=*/true);
  return;
}

/// \brief Attempt character array initialization from a string literal
/// (C++ [dcl.init.string], C99 6.7.8).
static void TryStringLiteralInitialization(Sema &S,
                                           const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           Expr *Initializer,
                                       InitializationSequence &Sequence) {
  Sequence.AddStringInitStep(Entity.getType());
}

/// \brief Attempt value initialization (C++ [dcl.init]p7).
static void TryValueInitialization(Sema &S,
                                   const InitializedEntity &Entity,
                                   const InitializationKind &Kind,
                                   InitializationSequence &Sequence,
                                   InitListExpr *InitList) {
  assert((!InitList || InitList->getNumInits() == 0) &&
         "Shouldn't use value-init for non-empty init lists");

  // C++98 [dcl.init]p5, C++11 [dcl.init]p7:
  //
  //   To value-initialize an object of type T means:
  QualType T = Entity.getType();

  //     -- if T is an array type, then each element is value-initialized;
  T = S.Context.getBaseElementType(T);

  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      bool NeedZeroInitialization = true;
      if (!S.getLangOpts().CPlusPlus11) {
        // C++98:
        // -- if T is a class type (clause 9) with a user-declared constructor
        //    (12.1), then the default constructor for T is called (and the
        //    initialization is ill-formed if T has no accessible default
        //    constructor);
        if (ClassDecl->hasUserDeclaredConstructor())
          NeedZeroInitialization = false;
      } else {
        // C++11:
        // -- if T is a class type (clause 9) with either no default constructor
        //    (12.1 [class.ctor]) or a default constructor that is user-provided
        //    or deleted, then the object is default-initialized;
        CXXConstructorDecl *CD = S.LookupDefaultConstructor(ClassDecl);
        if (!CD || !CD->getCanonicalDecl()->isDefaulted() || CD->isDeleted())
          NeedZeroInitialization = false;
      }

      // -- if T is a (possibly cv-qualified) non-union class type without a
      //    user-provided or deleted default constructor, then the object is
      //    zero-initialized and, if T has a non-trivial default constructor,
      //    default-initialized;
      // The 'non-union' here was removed by DR1502. The 'non-trivial default
      // constructor' part was removed by DR1507.
      if (NeedZeroInitialization)
        Sequence.AddZeroInitializationStep(Entity.getType());

      // C++03:
      // -- if T is a non-union class type without a user-declared constructor,
      //    then every non-static data member and base class component of T is
      //    value-initialized;
      // [...] A program that calls for [...] value-initialization of an
      // entity of reference type is ill-formed.
      //
      // C++11 doesn't need this handling, because value-initialization does not
      // occur recursively there, and the implicit default constructor is
      // defined as deleted in the problematic cases.
      if (!S.getLangOpts().CPlusPlus11 &&
          ClassDecl->hasUninitializedReferenceMember()) {
        Sequence.SetFailed(InitializationSequence::FK_TooManyInitsForReference);
        return;
      }

      // If this is list-value-initialization, pass the empty init list on when
      // building the constructor call. This affects the semantics of a few
      // things (such as whether an explicit default constructor can be called).
      Expr *InitListAsExpr = InitList;
      MultiExprArg Args(&InitListAsExpr, InitList ? 1 : 0);
      bool InitListSyntax = InitList;

      return TryConstructorInitialization(S, Entity, Kind, Args, T, Sequence,
                                          InitListSyntax);
    }
  }

  Sequence.AddZeroInitializationStep(Entity.getType());
}

/// \brief Attempt default initialization (C++ [dcl.init]p6).
static void TryDefaultInitialization(Sema &S,
                                     const InitializedEntity &Entity,
                                     const InitializationKind &Kind,
                                     InitializationSequence &Sequence) {
  assert(Kind.getKind() == InitializationKind::IK_Default);

  // C++ [dcl.init]p6:
  //   To default-initialize an object of type T means:
  //     - if T is an array type, each element is default-initialized;
  QualType DestType = S.Context.getBaseElementType(Entity.getType());
         
  //     - if T is a (possibly cv-qualified) class type (Clause 9), the default
  //       constructor for T is called (and the initialization is ill-formed if
  //       T has no accessible default constructor);
  if (DestType->isRecordType() && S.getLangOpts().CPlusPlus) {
    TryConstructorInitialization(S, Entity, Kind, None, DestType, Sequence);
    return;
  }

  //     - otherwise, no initialization is performed.

  //   If a program calls for the default initialization of an object of
  //   a const-qualified type T, T shall be a class type with a user-provided
  //   default constructor.
  if (DestType.isConstQualified() && S.getLangOpts().CPlusPlus) {
    if (!maybeRecoverWithZeroInitialization(S, Sequence, Entity))
      Sequence.SetFailed(InitializationSequence::FK_DefaultInitOfConst);
    return;
  }

  // If the destination type has a lifetime property, zero-initialize it.
  if (DestType.getQualifiers().hasObjCLifetime()) {
    Sequence.AddZeroInitializationStep(Entity.getType());
    return;
  }
}

/// \brief Attempt a user-defined conversion between two types (C++ [dcl.init]),
/// which enumerates all conversion functions and performs overload resolution
/// to select the best.
static void TryUserDefinedConversion(Sema &S,
                                     QualType DestType,
                                     const InitializationKind &Kind,
                                     Expr *Initializer,
                                     InitializationSequence &Sequence,
                                     bool TopLevelOfInitList) {
  assert(!DestType->isReferenceType() && "References are handled elsewhere");
  QualType SourceType = Initializer->getType();
  assert((DestType->isRecordType() || SourceType->isRecordType()) &&
         "Must have a class type to perform a user-defined conversion");

  // Build the candidate set directly in the initialization sequence
  // structure, so that it will persist if we fail.
  OverloadCandidateSet &CandidateSet = Sequence.getFailedCandidateSet();
  CandidateSet.clear();

  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = Kind.AllowExplicit();

  if (const RecordType *DestRecordType = DestType->getAs<RecordType>()) {
    // The type we're converting to is a class type. Enumerate its constructors
    // to see if there is a suitable conversion.
    CXXRecordDecl *DestRecordDecl
      = cast<CXXRecordDecl>(DestRecordType->getDecl());

    // Try to complete the type we're converting to.
    if (S.isCompleteType(Kind.getLocation(), DestType)) {
      DeclContext::lookup_result R = S.LookupConstructors(DestRecordDecl);
      // The container holding the constructors can under certain conditions
      // be changed while iterating. To be safe we copy the lookup results
      // to a new container.
      SmallVector<NamedDecl*, 8> CopyOfCon(R.begin(), R.end());
      for (SmallVectorImpl<NamedDecl *>::iterator
             Con = CopyOfCon.begin(), ConEnd = CopyOfCon.end();
           Con != ConEnd; ++Con) {
        NamedDecl *D = *Con;
        DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());

        // Find the constructor (which may be a template).
        CXXConstructorDecl *Constructor = nullptr;
        FunctionTemplateDecl *ConstructorTmpl
          = dyn_cast<FunctionTemplateDecl>(D);
        if (ConstructorTmpl)
          Constructor = cast<CXXConstructorDecl>(
                                           ConstructorTmpl->getTemplatedDecl());
        else
          Constructor = cast<CXXConstructorDecl>(D);

        if (!Constructor->isInvalidDecl() &&
            Constructor->isConvertingConstructor(AllowExplicit)) {
          if (ConstructorTmpl)
            S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                           /*ExplicitArgs*/ nullptr,
                                           Initializer, CandidateSet,
                                           /*SuppressUserConversions=*/true);
          else
            S.AddOverloadCandidate(Constructor, FoundDecl,
                                   Initializer, CandidateSet,
                                   /*SuppressUserConversions=*/true);
        }
      }
    }
  }

  SourceLocation DeclLoc = Initializer->getLocStart();

  if (const RecordType *SourceRecordType = SourceType->getAs<RecordType>()) {
    // The type we're converting from is a class type, enumerate its conversion
    // functions.

    // We can only enumerate the conversion functions for a complete type; if
    // the type isn't complete, simply skip this step.
    if (S.isCompleteType(DeclLoc, SourceType)) {
      CXXRecordDecl *SourceRecordDecl
        = cast<CXXRecordDecl>(SourceRecordType->getDecl());

      const auto &Conversions =
          SourceRecordDecl->getVisibleConversionFunctions();
      for (auto I = Conversions.begin(), E = Conversions.end(); I != E; ++I) {
        NamedDecl *D = *I;
        CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(D->getDeclContext());
        if (isa<UsingShadowDecl>(D))
          D = cast<UsingShadowDecl>(D)->getTargetDecl();

        FunctionTemplateDecl *ConvTemplate = dyn_cast<FunctionTemplateDecl>(D);
        CXXConversionDecl *Conv;
        if (ConvTemplate)
          Conv = cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
        else
          Conv = cast<CXXConversionDecl>(D);

        if (AllowExplicit || !Conv->isExplicit()) {
          if (ConvTemplate)
            S.AddTemplateConversionCandidate(ConvTemplate, I.getPair(),
                                             ActingDC, Initializer, DestType,
                                             CandidateSet, AllowExplicit);
          else
            S.AddConversionCandidate(Conv, I.getPair(), ActingDC,
                                     Initializer, DestType, CandidateSet,
                                     AllowExplicit);
        }
      }
    }
  }

  // Perform overload resolution. If it fails, return the failed result.
  OverloadCandidateSet::iterator Best;
  if (OverloadingResult Result
        = CandidateSet.BestViableFunction(S, DeclLoc, Best, true)) {
    Sequence.SetOverloadFailure(
                        InitializationSequence::FK_UserConversionOverloadFailed,
                                Result);
    return;
  }

  FunctionDecl *Function = Best->Function;
  Function->setReferenced();
  bool HadMultipleCandidates = (CandidateSet.size() > 1);

  if (isa<CXXConstructorDecl>(Function)) {
    // Add the user-defined conversion step. Any cv-qualification conversion is
    // subsumed by the initialization. Per DR5, the created temporary is of the
    // cv-unqualified type of the destination.
    Sequence.AddUserConversionStep(Function, Best->FoundDecl,
                                   DestType.getUnqualifiedType(),
                                   HadMultipleCandidates);
    return;
  }

  // Add the user-defined conversion step that calls the conversion function.
  QualType ConvType = Function->getCallResultType();
  if (ConvType->getAs<RecordType>()) {
    // If we're converting to a class type, there may be an copy of
    // the resulting temporary object (possible to create an object of
    // a base class type). That copy is not a separate conversion, so
    // we just make a note of the actual destination type (possibly a
    // base class of the type returned by the conversion function) and
    // let the user-defined conversion step handle the conversion.
    Sequence.AddUserConversionStep(Function, Best->FoundDecl, DestType,
                                   HadMultipleCandidates);
    return;
  }

  Sequence.AddUserConversionStep(Function, Best->FoundDecl, ConvType,
                                 HadMultipleCandidates);

  // If the conversion following the call to the conversion function
  // is interesting, add it as a separate step.
  if (Best->FinalConversion.First || Best->FinalConversion.Second ||
      Best->FinalConversion.Third) {
    ImplicitConversionSequence ICS;
    ICS.setStandard();
    ICS.Standard = Best->FinalConversion;
    Sequence.AddConversionSequenceStep(ICS, DestType, TopLevelOfInitList);
  }
}

/// An egregious hack for compatibility with libstdc++-4.2: in <tr1/hashtable>,
/// a function with a pointer return type contains a 'return false;' statement.
/// In C++11, 'false' is not a null pointer, so this breaks the build of any
/// code using that header.
///
/// Work around this by treating 'return false;' as zero-initializing the result
/// if it's used in a pointer-returning function in a system header.
static bool isLibstdcxxPointerReturnFalseHack(Sema &S,
                                              const InitializedEntity &Entity,
                                              const Expr *Init) {
  return S.getLangOpts().CPlusPlus11 &&
         Entity.getKind() == InitializedEntity::EK_Result &&
         Entity.getType()->isPointerType() &&
         isa<CXXBoolLiteralExpr>(Init) &&
         !cast<CXXBoolLiteralExpr>(Init)->getValue() &&
         S.getSourceManager().isInSystemHeader(Init->getExprLoc());
}

/// The non-zero enum values here are indexes into diagnostic alternatives.
enum InvalidICRKind { IIK_okay, IIK_nonlocal, IIK_nonscalar };

/// Determines whether this expression is an acceptable ICR source.
static InvalidICRKind isInvalidICRSource(ASTContext &C, Expr *e,
                                         bool isAddressOf, bool &isWeakAccess) {
  // Skip parens.
  e = e->IgnoreParens();

  // Skip address-of nodes.
  if (UnaryOperator *op = dyn_cast<UnaryOperator>(e)) {
    if (op->getOpcode() == UO_AddrOf)
      return isInvalidICRSource(C, op->getSubExpr(), /*addressof*/ true,
                                isWeakAccess);

  // Skip certain casts.
  } else if (CastExpr *ce = dyn_cast<CastExpr>(e)) {
    switch (ce->getCastKind()) {
    case CK_Dependent:
    case CK_BitCast:
    case CK_LValueBitCast:
    case CK_NoOp:
      return isInvalidICRSource(C, ce->getSubExpr(), isAddressOf, isWeakAccess);

    case CK_ArrayToPointerDecay:
      return IIK_nonscalar;

    case CK_NullToPointer:
      return IIK_okay;

    default:
      break;
    }

  // If we have a declaration reference, it had better be a local variable.
  } else if (isa<DeclRefExpr>(e)) {
    // set isWeakAccess to true, to mean that there will be an implicit 
    // load which requires a cleanup.
    if (e->getType().getObjCLifetime() == Qualifiers::OCL_Weak)
      isWeakAccess = true;
    
    if (!isAddressOf) return IIK_nonlocal;

    VarDecl *var = dyn_cast<VarDecl>(cast<DeclRefExpr>(e)->getDecl());
    if (!var) return IIK_nonlocal;

    return (var->hasLocalStorage() ? IIK_okay : IIK_nonlocal);

  // If we have a conditional operator, check both sides.
  } else if (ConditionalOperator *cond = dyn_cast<ConditionalOperator>(e)) {
    if (InvalidICRKind iik = isInvalidICRSource(C, cond->getLHS(), isAddressOf,
                                                isWeakAccess))
      return iik;

    return isInvalidICRSource(C, cond->getRHS(), isAddressOf, isWeakAccess);

  // These are never scalar.
  } else if (isa<ArraySubscriptExpr>(e)) {
    return IIK_nonscalar;

  // Otherwise, it needs to be a null pointer constant.
  } else {
    return (e->isNullPointerConstant(C, Expr::NPC_ValueDependentIsNull)
            ? IIK_okay : IIK_nonlocal);
  }

  return IIK_nonlocal;
}

/// Check whether the given expression is a valid operand for an
/// indirect copy/restore.
static void checkIndirectCopyRestoreSource(Sema &S, Expr *src) {
  assert(src->isRValue());
  bool isWeakAccess = false;
  InvalidICRKind iik = isInvalidICRSource(S.Context, src, false, isWeakAccess);
  // If isWeakAccess to true, there will be an implicit 
  // load which requires a cleanup.
  if (S.getLangOpts().ObjCAutoRefCount && isWeakAccess)
    S.ExprNeedsCleanups = true;
  
  if (iik == IIK_okay) return;

  S.Diag(src->getExprLoc(), diag::err_arc_nonlocal_writeback)
    << ((unsigned) iik - 1)  // shift index into diagnostic explanations
    << src->getSourceRange();
}

/// \brief Determine whether we have compatible array types for the
/// purposes of GNU by-copy array initialization.
static bool hasCompatibleArrayTypes(ASTContext &Context, const ArrayType *Dest,
                                    const ArrayType *Source) {
  // If the source and destination array types are equivalent, we're
  // done.
  if (Context.hasSameType(QualType(Dest, 0), QualType(Source, 0)))
    return true;

  // Make sure that the element types are the same.
  if (!Context.hasSameType(Dest->getElementType(), Source->getElementType()))
    return false;

  // The only mismatch we allow is when the destination is an
  // incomplete array type and the source is a constant array type.
  return Source->isConstantArrayType() && Dest->isIncompleteArrayType();
}

static bool tryObjCWritebackConversion(Sema &S,
                                       InitializationSequence &Sequence,
                                       const InitializedEntity &Entity,
                                       Expr *Initializer) {
  bool ArrayDecay = false;
  QualType ArgType = Initializer->getType();
  QualType ArgPointee;
  if (const ArrayType *ArgArrayType = S.Context.getAsArrayType(ArgType)) {
    ArrayDecay = true;
    ArgPointee = ArgArrayType->getElementType();
    ArgType = S.Context.getPointerType(ArgPointee);
  }
      
  // Handle write-back conversion.
  QualType ConvertedArgType;
  if (!S.isObjCWritebackConversion(ArgType, Entity.getType(),
                                   ConvertedArgType))
    return false;

  // We should copy unless we're passing to an argument explicitly
  // marked 'out'.
  bool ShouldCopy = true;
  if (ParmVarDecl *param = cast_or_null<ParmVarDecl>(Entity.getDecl()))
    ShouldCopy = (param->getObjCDeclQualifier() != ParmVarDecl::OBJC_TQ_Out);

  // Do we need an lvalue conversion?
  if (ArrayDecay || Initializer->isGLValue()) {
    ImplicitConversionSequence ICS;
    ICS.setStandard();
    ICS.Standard.setAsIdentityConversion();

    QualType ResultType;
    if (ArrayDecay) {
      ICS.Standard.First = ICK_Array_To_Pointer;
      ResultType = S.Context.getPointerType(ArgPointee);
    } else {
      ICS.Standard.First = ICK_Lvalue_To_Rvalue;
      ResultType = Initializer->getType().getNonLValueExprType(S.Context);
    }
          
    Sequence.AddConversionSequenceStep(ICS, ResultType);
  }
        
  Sequence.AddPassByIndirectCopyRestoreStep(Entity.getType(), ShouldCopy);
  return true;
}

static bool TryOCLSamplerInitialization(Sema &S,
                                        InitializationSequence &Sequence,
                                        QualType DestType,
                                        Expr *Initializer) {
  if (!S.getLangOpts().OpenCL || !DestType->isSamplerT() ||
    !Initializer->isIntegerConstantExpr(S.getASTContext()))
    return false;

  Sequence.AddOCLSamplerInitStep(DestType);
  return true;
}

//
// OpenCL 1.2 spec, s6.12.10
//
// The event argument can also be used to associate the
// async_work_group_copy with a previous async copy allowing
// an event to be shared by multiple async copies; otherwise
// event should be zero.
//
static bool TryOCLZeroEventInitialization(Sema &S,
                                          InitializationSequence &Sequence,
                                          QualType DestType,
                                          Expr *Initializer) {
  if (!S.getLangOpts().OpenCL || !DestType->isEventT() ||
      !Initializer->isIntegerConstantExpr(S.getASTContext()) ||
      (Initializer->EvaluateKnownConstInt(S.getASTContext()) != 0))
    return false;

  Sequence.AddOCLZeroEventStep(DestType);
  return true;
}

InitializationSequence::InitializationSequence(Sema &S,
                                               const InitializedEntity &Entity,
                                               const InitializationKind &Kind,
                                               MultiExprArg Args,
                                               bool TopLevelOfInitList)
    : FailedCandidateSet(Kind.getLocation(), OverloadCandidateSet::CSK_Normal) {
  InitializeFrom(S, Entity, Kind, Args, TopLevelOfInitList);
}

/// Tries to get a FunctionDecl out of `E`. If it succeeds and we can take the
/// address of that function, this returns true. Otherwise, it returns false.
static bool isExprAnUnaddressableFunction(Sema &S, const Expr *E) {
  auto *DRE = dyn_cast<DeclRefExpr>(E);
  if (!DRE || !isa<FunctionDecl>(DRE->getDecl()))
    return false;

  return !S.checkAddressOfFunctionIsAvailable(
      cast<FunctionDecl>(DRE->getDecl()));
}

void InitializationSequence::InitializeFrom(Sema &S,
                                            const InitializedEntity &Entity,
                                            const InitializationKind &Kind,
                                            MultiExprArg Args,
                                            bool TopLevelOfInitList) {
  ASTContext &Context = S.Context;

  // Eliminate non-overload placeholder types in the arguments.  We
  // need to do this before checking whether types are dependent
  // because lowering a pseudo-object expression might well give us
  // something of dependent type.
  for (unsigned I = 0, E = Args.size(); I != E; ++I)
    if (Args[I]->getType()->isNonOverloadPlaceholderType()) {
      // FIXME: should we be doing this here?
      ExprResult result = S.CheckPlaceholderExpr(Args[I]);
      if (result.isInvalid()) {
        SetFailed(FK_PlaceholderType);
        return;
      }
      Args[I] = result.get();
    }

  // C++0x [dcl.init]p16:
  //   The semantics of initializers are as follows. The destination type is
  //   the type of the object or reference being initialized and the source
  //   type is the type of the initializer expression. The source type is not
  //   defined when the initializer is a braced-init-list or when it is a
  //   parenthesized list of expressions.
  QualType DestType = Entity.getType();

  if (DestType->isDependentType() ||
      Expr::hasAnyTypeDependentArguments(Args)) {
    SequenceKind = DependentSequence;
    return;
  }

  // Almost everything is a normal sequence.
  setSequenceKind(NormalSequence);

  QualType SourceType;
  Expr *Initializer = nullptr;
  if (Args.size() == 1) {
    Initializer = Args[0];
    if (S.getLangOpts().ObjC1) {
      if (S.CheckObjCBridgeRelatedConversions(Initializer->getLocStart(),
                                              DestType, Initializer->getType(),
                                              Initializer) ||
          S.ConversionToObjCStringLiteralCheck(DestType, Initializer))
        Args[0] = Initializer;
    }
    if (!isa<InitListExpr>(Initializer))
      SourceType = Initializer->getType();
  }

  //     - If the initializer is a (non-parenthesized) braced-init-list, the
  //       object is list-initialized (8.5.4).
  if (Kind.getKind() != InitializationKind::IK_Direct) {
    if (InitListExpr *InitList = dyn_cast_or_null<InitListExpr>(Initializer)) {
      TryListInitialization(S, Entity, Kind, InitList, *this);
      return;
    }
  }

  //     - If the destination type is a reference type, see 8.5.3.
  if (DestType->isReferenceType()) {
    // C++0x [dcl.init.ref]p1:
    //   A variable declared to be a T& or T&&, that is, "reference to type T"
    //   (8.3.2), shall be initialized by an object, or function, of type T or
    //   by an object that can be converted into a T.
    // (Therefore, multiple arguments are not permitted.)
    if (Args.size() != 1)
      SetFailed(FK_TooManyInitsForReference);
    else
      TryReferenceInitialization(S, Entity, Kind, Args[0], *this);
    return;
  }

  //     - If the initializer is (), the object is value-initialized.
  if (Kind.getKind() == InitializationKind::IK_Value ||
      (Kind.getKind() == InitializationKind::IK_Direct && Args.empty())) {
    TryValueInitialization(S, Entity, Kind, *this);
    return;
  }

  // Handle default initialization.
  if (Kind.getKind() == InitializationKind::IK_Default) {
    TryDefaultInitialization(S, Entity, Kind, *this);
    return;
  }

  //     - If the destination type is an array of characters, an array of
  //       char16_t, an array of char32_t, or an array of wchar_t, and the
  //       initializer is a string literal, see 8.5.2.
  //     - Otherwise, if the destination type is an array, the program is
  //       ill-formed.
  if (const ArrayType *DestAT = Context.getAsArrayType(DestType)) {
    if (Initializer && isa<VariableArrayType>(DestAT)) {
      SetFailed(FK_VariableLengthArrayHasInitializer);
      return;
    }

    if (Initializer) {
      switch (IsStringInit(Initializer, DestAT, Context)) {
      case SIF_None:
        TryStringLiteralInitialization(S, Entity, Kind, Initializer, *this);
        return;
      case SIF_NarrowStringIntoWideChar:
        SetFailed(FK_NarrowStringIntoWideCharArray);
        return;
      case SIF_WideStringIntoChar:
        SetFailed(FK_WideStringIntoCharArray);
        return;
      case SIF_IncompatWideStringIntoWideChar:
        SetFailed(FK_IncompatWideStringIntoWideChar);
        return;
      case SIF_Other:
        break;
      }
    }

    // Note: as an GNU C extension, we allow initialization of an
    // array from a compound literal that creates an array of the same
    // type, so long as the initializer has no side effects.
    if (!S.getLangOpts().CPlusPlus && Initializer &&
        isa<CompoundLiteralExpr>(Initializer->IgnoreParens()) &&
        Initializer->getType()->isArrayType()) {
      const ArrayType *SourceAT
        = Context.getAsArrayType(Initializer->getType());
      if (!hasCompatibleArrayTypes(S.Context, DestAT, SourceAT))
        SetFailed(FK_ArrayTypeMismatch);
      else if (Initializer->HasSideEffects(S.Context))
        SetFailed(FK_NonConstantArrayInit);
      else {
        AddArrayInitStep(DestType);
      }
    }
    // Note: as a GNU C++ extension, we allow list-initialization of a
    // class member of array type from a parenthesized initializer list.
    else if (S.getLangOpts().CPlusPlus &&
             Entity.getKind() == InitializedEntity::EK_Member &&
             Initializer && isa<InitListExpr>(Initializer)) {
      TryListInitialization(S, Entity, Kind, cast<InitListExpr>(Initializer),
                            *this);
      AddParenthesizedArrayInitStep(DestType);
    } else if (DestAT->getElementType()->isCharType())
      SetFailed(FK_ArrayNeedsInitListOrStringLiteral);
    else if (IsWideCharCompatible(DestAT->getElementType(), Context))
      SetFailed(FK_ArrayNeedsInitListOrWideStringLiteral);
    else
      SetFailed(FK_ArrayNeedsInitList);

    return;
  }

  // Determine whether we should consider writeback conversions for
  // Objective-C ARC.
  bool allowObjCWritebackConversion = S.getLangOpts().ObjCAutoRefCount &&
         Entity.isParameterKind();

  // We're at the end of the line for C: it's either a write-back conversion
  // or it's a C assignment. There's no need to check anything else.
  if (!S.getLangOpts().CPlusPlus) {
    // If allowed, check whether this is an Objective-C writeback conversion.
    if (allowObjCWritebackConversion &&
        tryObjCWritebackConversion(S, *this, Entity, Initializer)) {
      return;
    }

    if (TryOCLSamplerInitialization(S, *this, DestType, Initializer))
      return;

    if (TryOCLZeroEventInitialization(S, *this, DestType, Initializer))
      return;

    // Handle initialization in C
    AddCAssignmentStep(DestType);
    MaybeProduceObjCObject(S, *this, Entity);
    return;
  }

  assert(S.getLangOpts().CPlusPlus);

  //     - If the destination type is a (possibly cv-qualified) class type:
  if (DestType->isRecordType()) {
    //     - If the initialization is direct-initialization, or if it is
    //       copy-initialization where the cv-unqualified version of the
    //       source type is the same class as, or a derived class of, the
    //       class of the destination, constructors are considered. [...]
    if (Kind.getKind() == InitializationKind::IK_Direct ||
        (Kind.getKind() == InitializationKind::IK_Copy &&
         (Context.hasSameUnqualifiedType(SourceType, DestType) ||
          S.IsDerivedFrom(Initializer->getLocStart(), SourceType, DestType))))
      TryConstructorInitialization(S, Entity, Kind, Args,
                                   DestType, *this);
    //     - Otherwise (i.e., for the remaining copy-initialization cases),
    //       user-defined conversion sequences that can convert from the source
    //       type to the destination type or (when a conversion function is
    //       used) to a derived class thereof are enumerated as described in
    //       13.3.1.4, and the best one is chosen through overload resolution
    //       (13.3).
    else
      TryUserDefinedConversion(S, DestType, Kind, Initializer, *this,
                               TopLevelOfInitList);
    return;
  }

  if (Args.size() > 1) {
    SetFailed(FK_TooManyInitsForScalar);
    return;
  }
  assert(Args.size() == 1 && "Zero-argument case handled above");

  //    - Otherwise, if the source type is a (possibly cv-qualified) class
  //      type, conversion functions are considered.
  if (!SourceType.isNull() && SourceType->isRecordType()) {
    // For a conversion to _Atomic(T) from either T or a class type derived
    // from T, initialize the T object then convert to _Atomic type.
    bool NeedAtomicConversion = false;
    if (const AtomicType *Atomic = DestType->getAs<AtomicType>()) {
      if (Context.hasSameUnqualifiedType(SourceType, Atomic->getValueType()) ||
          S.IsDerivedFrom(Initializer->getLocStart(), SourceType,
                          Atomic->getValueType())) {
        DestType = Atomic->getValueType();
        NeedAtomicConversion = true;
      }
    }

    TryUserDefinedConversion(S, DestType, Kind, Initializer, *this,
                             TopLevelOfInitList);
    MaybeProduceObjCObject(S, *this, Entity);
    if (!Failed() && NeedAtomicConversion)
      AddAtomicConversionStep(Entity.getType());
    return;
  }

  //    - Otherwise, the initial value of the object being initialized is the
  //      (possibly converted) value of the initializer expression. Standard
  //      conversions (Clause 4) will be used, if necessary, to convert the
  //      initializer expression to the cv-unqualified version of the
  //      destination type; no user-defined conversions are considered.

  ImplicitConversionSequence ICS
    = S.TryImplicitConversion(Initializer, DestType,
                              /*SuppressUserConversions*/true,
                              /*AllowExplicitConversions*/ false,
                              /*InOverloadResolution*/ false,
                              /*CStyle=*/Kind.isCStyleOrFunctionalCast(),
                              allowObjCWritebackConversion);

  if (ICS.isStandard() &&
      ICS.Standard.Second == ICK_Writeback_Conversion) {
    // Objective-C ARC writeback conversion.
    
    // We should copy unless we're passing to an argument explicitly
    // marked 'out'.
    bool ShouldCopy = true;
    if (ParmVarDecl *Param = cast_or_null<ParmVarDecl>(Entity.getDecl()))
      ShouldCopy = (Param->getObjCDeclQualifier() != ParmVarDecl::OBJC_TQ_Out);
    
    // If there was an lvalue adjustment, add it as a separate conversion.
    if (ICS.Standard.First == ICK_Array_To_Pointer ||
        ICS.Standard.First == ICK_Lvalue_To_Rvalue) {
      ImplicitConversionSequence LvalueICS;
      LvalueICS.setStandard();
      LvalueICS.Standard.setAsIdentityConversion();
      LvalueICS.Standard.setAllToTypes(ICS.Standard.getToType(0));
      LvalueICS.Standard.First = ICS.Standard.First;
      AddConversionSequenceStep(LvalueICS, ICS.Standard.getToType(0));
    }
    
    AddPassByIndirectCopyRestoreStep(DestType, ShouldCopy);
  } else if (ICS.isBad()) {
    DeclAccessPair dap;
    if (isLibstdcxxPointerReturnFalseHack(S, Entity, Initializer)) {
      AddZeroInitializationStep(Entity.getType());
    } else if (Initializer->getType() == Context.OverloadTy &&
               !S.ResolveAddressOfOverloadedFunction(Initializer, DestType,
                                                     false, dap))
      SetFailed(InitializationSequence::FK_AddressOfOverloadFailed);
    else if (Initializer->getType()->isFunctionType() &&
             isExprAnUnaddressableFunction(S, Initializer))
      SetFailed(InitializationSequence::FK_AddressOfUnaddressableFunction);
    else
      SetFailed(InitializationSequence::FK_ConversionFailed);
  } else {
    AddConversionSequenceStep(ICS, DestType, TopLevelOfInitList);

    MaybeProduceObjCObject(S, *this, Entity);
  }
}

InitializationSequence::~InitializationSequence() {
  for (auto &S : Steps)
    S.Destroy();
}

//===----------------------------------------------------------------------===//
// Perform initialization
//===----------------------------------------------------------------------===//
static Sema::AssignmentAction
getAssignmentAction(const InitializedEntity &Entity, bool Diagnose = false) {
  switch(Entity.getKind()) {
  case InitializedEntity::EK_Variable:
  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_Delegating:
    return Sema::AA_Initializing;

  case InitializedEntity::EK_Parameter:
    if (Entity.getDecl() &&
        isa<ObjCMethodDecl>(Entity.getDecl()->getDeclContext()))
      return Sema::AA_Sending;

    return Sema::AA_Passing;

  case InitializedEntity::EK_Parameter_CF_Audited:
    if (Entity.getDecl() &&
      isa<ObjCMethodDecl>(Entity.getDecl()->getDeclContext()))
      return Sema::AA_Sending;
      
    return !Diagnose ? Sema::AA_Passing : Sema::AA_Passing_CFAudited;
      
  case InitializedEntity::EK_Result:
    return Sema::AA_Returning;

  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_RelatedResult:
    // FIXME: Can we tell apart casting vs. converting?
    return Sema::AA_Casting;

  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_ComplexElement:
  case InitializedEntity::EK_BlockElement:
  case InitializedEntity::EK_LambdaCapture:
  case InitializedEntity::EK_CompoundLiteralInit:
    return Sema::AA_Initializing;
  }

  llvm_unreachable("Invalid EntityKind!");
}

/// \brief Whether we should bind a created object as a temporary when
/// initializing the given entity.
static bool shouldBindAsTemporary(const InitializedEntity &Entity) {
  switch (Entity.getKind()) {
  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_Result:
  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Variable:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_Delegating:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_ComplexElement:
  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_BlockElement:
  case InitializedEntity::EK_LambdaCapture:
  case InitializedEntity::EK_CompoundLiteralInit:
    return false;

  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Parameter_CF_Audited:
  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_RelatedResult:
    return true;
  }

  llvm_unreachable("missed an InitializedEntity kind?");
}

/// \brief Whether the given entity, when initialized with an object
/// created for that initialization, requires destruction.
static bool shouldDestroyTemporary(const InitializedEntity &Entity) {
  switch (Entity.getKind()) {
    case InitializedEntity::EK_Result:
    case InitializedEntity::EK_New:
    case InitializedEntity::EK_Base:
    case InitializedEntity::EK_Delegating:
    case InitializedEntity::EK_VectorElement:
    case InitializedEntity::EK_ComplexElement:
    case InitializedEntity::EK_BlockElement:
    case InitializedEntity::EK_LambdaCapture:
      return false;

    case InitializedEntity::EK_Member:
    case InitializedEntity::EK_Variable:
    case InitializedEntity::EK_Parameter:
    case InitializedEntity::EK_Parameter_CF_Audited:
    case InitializedEntity::EK_Temporary:
    case InitializedEntity::EK_ArrayElement:
    case InitializedEntity::EK_Exception:
    case InitializedEntity::EK_CompoundLiteralInit:
    case InitializedEntity::EK_RelatedResult:
      return true;
  }

  llvm_unreachable("missed an InitializedEntity kind?");
}

/// \brief Look for copy and move constructors and constructor templates, for
/// copying an object via direct-initialization (per C++11 [dcl.init]p16).
static void LookupCopyAndMoveConstructors(Sema &S,
                                          OverloadCandidateSet &CandidateSet,
                                          CXXRecordDecl *Class,
                                          Expr *CurInitExpr) {
  DeclContext::lookup_result R = S.LookupConstructors(Class);
  // The container holding the constructors can under certain conditions
  // be changed while iterating (e.g. because of deserialization).
  // To be safe we copy the lookup results to a new container.
  SmallVector<NamedDecl*, 16> Ctors(R.begin(), R.end());
  for (SmallVectorImpl<NamedDecl *>::iterator
         CI = Ctors.begin(), CE = Ctors.end(); CI != CE; ++CI) {
    NamedDecl *D = *CI;
    CXXConstructorDecl *Constructor = nullptr;

    if ((Constructor = dyn_cast<CXXConstructorDecl>(D))) {
      // Handle copy/moveconstructors, only.
      if (!Constructor || Constructor->isInvalidDecl() ||
          !Constructor->isCopyOrMoveConstructor() ||
          !Constructor->isConvertingConstructor(/*AllowExplicit=*/true))
        continue;

      DeclAccessPair FoundDecl
        = DeclAccessPair::make(Constructor, Constructor->getAccess());
      S.AddOverloadCandidate(Constructor, FoundDecl,
                             CurInitExpr, CandidateSet);
      continue;
    }

    // Handle constructor templates.
    FunctionTemplateDecl *ConstructorTmpl = cast<FunctionTemplateDecl>(D);
    if (ConstructorTmpl->isInvalidDecl())
      continue;

    Constructor = cast<CXXConstructorDecl>(
                                         ConstructorTmpl->getTemplatedDecl());
    if (!Constructor->isConvertingConstructor(/*AllowExplicit=*/true))
      continue;

    // FIXME: Do we need to limit this to copy-constructor-like
    // candidates?
    DeclAccessPair FoundDecl
      = DeclAccessPair::make(ConstructorTmpl, ConstructorTmpl->getAccess());
    S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl, nullptr,
                                   CurInitExpr, CandidateSet, true);
  }
}

/// \brief Get the location at which initialization diagnostics should appear.
static SourceLocation getInitializationLoc(const InitializedEntity &Entity,
                                           Expr *Initializer) {
  switch (Entity.getKind()) {
  case InitializedEntity::EK_Result:
    return Entity.getReturnLoc();

  case InitializedEntity::EK_Exception:
    return Entity.getThrowLoc();

  case InitializedEntity::EK_Variable:
    return Entity.getDecl()->getLocation();

  case InitializedEntity::EK_LambdaCapture:
    return Entity.getCaptureLoc();
      
  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Parameter_CF_Audited:
  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_Delegating:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_ComplexElement:
  case InitializedEntity::EK_BlockElement:
  case InitializedEntity::EK_CompoundLiteralInit:
  case InitializedEntity::EK_RelatedResult:
    return Initializer->getLocStart();
  }
  llvm_unreachable("missed an InitializedEntity kind?");
}

/// \brief Make a (potentially elidable) temporary copy of the object
/// provided by the given initializer by calling the appropriate copy
/// constructor.
///
/// \param S The Sema object used for type-checking.
///
/// \param T The type of the temporary object, which must either be
/// the type of the initializer expression or a superclass thereof.
///
/// \param Entity The entity being initialized.
///
/// \param CurInit The initializer expression.
///
/// \param IsExtraneousCopy Whether this is an "extraneous" copy that
/// is permitted in C++03 (but not C++0x) when binding a reference to
/// an rvalue.
///
/// \returns An expression that copies the initializer expression into
/// a temporary object, or an error expression if a copy could not be
/// created.
static ExprResult CopyObject(Sema &S,
                             QualType T,
                             const InitializedEntity &Entity,
                             ExprResult CurInit,
                             bool IsExtraneousCopy) {
  if (CurInit.isInvalid())
    return CurInit;
  // Determine which class type we're copying to.
  Expr *CurInitExpr = (Expr *)CurInit.get();
  CXXRecordDecl *Class = nullptr;
  if (const RecordType *Record = T->getAs<RecordType>())
    Class = cast<CXXRecordDecl>(Record->getDecl());
  if (!Class)
    return CurInit;

  // C++0x [class.copy]p32:
  //   When certain criteria are met, an implementation is allowed to
  //   omit the copy/move construction of a class object, even if the
  //   copy/move constructor and/or destructor for the object have
  //   side effects. [...]
  //     - when a temporary class object that has not been bound to a
  //       reference (12.2) would be copied/moved to a class object
  //       with the same cv-unqualified type, the copy/move operation
  //       can be omitted by constructing the temporary object
  //       directly into the target of the omitted copy/move
  //
  // Note that the other three bullets are handled elsewhere. Copy
  // elision for return statements and throw expressions are handled as part
  // of constructor initialization, while copy elision for exception handlers
  // is handled by the run-time.
  bool Elidable = CurInitExpr->isTemporaryObject(S.Context, Class);
  SourceLocation Loc = getInitializationLoc(Entity, CurInit.get());

  // Make sure that the type we are copying is complete.
  if (S.RequireCompleteType(Loc, T, diag::err_temp_copy_incomplete))
    return CurInit;

  // Perform overload resolution using the class's copy/move constructors.
  // Only consider constructors and constructor templates. Per
  // C++0x [dcl.init]p16, second bullet to class types, this initialization
  // is direct-initialization.
  OverloadCandidateSet CandidateSet(Loc, OverloadCandidateSet::CSK_Normal);
  LookupCopyAndMoveConstructors(S, CandidateSet, Class, CurInitExpr);

  bool HadMultipleCandidates = (CandidateSet.size() > 1);

  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(S, Loc, Best)) {
  case OR_Success:
    break;

  case OR_No_Viable_Function:
    S.Diag(Loc, IsExtraneousCopy && !S.isSFINAEContext()
           ? diag::ext_rvalue_to_reference_temp_copy_no_viable
           : diag::err_temp_copy_no_viable)
      << (int)Entity.getKind() << CurInitExpr->getType()
      << CurInitExpr->getSourceRange();
    CandidateSet.NoteCandidates(S, OCD_AllCandidates, CurInitExpr);
    if (!IsExtraneousCopy || S.isSFINAEContext())
      return ExprError();
    return CurInit;

  case OR_Ambiguous:
    S.Diag(Loc, diag::err_temp_copy_ambiguous)
      << (int)Entity.getKind() << CurInitExpr->getType()
      << CurInitExpr->getSourceRange();
    CandidateSet.NoteCandidates(S, OCD_ViableCandidates, CurInitExpr);
    return ExprError();

  case OR_Deleted:
    S.Diag(Loc, diag::err_temp_copy_deleted)
      << (int)Entity.getKind() << CurInitExpr->getType()
      << CurInitExpr->getSourceRange();
    S.NoteDeletedFunction(Best->Function);
    return ExprError();
  }

  CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(Best->Function);
  SmallVector<Expr*, 8> ConstructorArgs;
  CurInit.get(); // Ownership transferred into MultiExprArg, below.

  S.CheckConstructorAccess(Loc, Constructor, Entity,
                           Best->FoundDecl.getAccess(), IsExtraneousCopy);

  if (IsExtraneousCopy) {
    // If this is a totally extraneous copy for C++03 reference
    // binding purposes, just return the original initialization
    // expression. We don't generate an (elided) copy operation here
    // because doing so would require us to pass down a flag to avoid
    // infinite recursion, where each step adds another extraneous,
    // elidable copy.

    // Instantiate the default arguments of any extra parameters in
    // the selected copy constructor, as if we were going to create a
    // proper call to the copy constructor.
    for (unsigned I = 1, N = Constructor->getNumParams(); I != N; ++I) {
      ParmVarDecl *Parm = Constructor->getParamDecl(I);
      if (S.RequireCompleteType(Loc, Parm->getType(),
                                diag::err_call_incomplete_argument))
        break;

      // Build the default argument expression; we don't actually care
      // if this succeeds or not, because this routine will complain
      // if there was a problem.
      S.BuildCXXDefaultArgExpr(Loc, Constructor, Parm);
    }

    return CurInitExpr;
  }

  // Determine the arguments required to actually perform the
  // constructor call (we might have derived-to-base conversions, or
  // the copy constructor may have default arguments).
  if (S.CompleteConstructorCall(Constructor, CurInitExpr, Loc, ConstructorArgs))
    return ExprError();

  // Actually perform the constructor call.
  CurInit = S.BuildCXXConstructExpr(Loc, T, Constructor, Elidable,
                                    ConstructorArgs,
                                    HadMultipleCandidates,
                                    /*ListInit*/ false,
                                    /*StdInitListInit*/ false,
                                    /*ZeroInit*/ false,
                                    CXXConstructExpr::CK_Complete,
                                    SourceRange());

  // If we're supposed to bind temporaries, do so.
  if (!CurInit.isInvalid() && shouldBindAsTemporary(Entity))
    CurInit = S.MaybeBindToTemporary(CurInit.getAs<Expr>());
  return CurInit;
}

/// \brief Check whether elidable copy construction for binding a reference to
/// a temporary would have succeeded if we were building in C++98 mode, for
/// -Wc++98-compat.
static void CheckCXX98CompatAccessibleCopy(Sema &S,
                                           const InitializedEntity &Entity,
                                           Expr *CurInitExpr) {
  assert(S.getLangOpts().CPlusPlus11);

  const RecordType *Record = CurInitExpr->getType()->getAs<RecordType>();
  if (!Record)
    return;

  SourceLocation Loc = getInitializationLoc(Entity, CurInitExpr);
  if (S.Diags.isIgnored(diag::warn_cxx98_compat_temp_copy, Loc))
    return;

  // Find constructors which would have been considered.
  OverloadCandidateSet CandidateSet(Loc, OverloadCandidateSet::CSK_Normal);
  LookupCopyAndMoveConstructors(
      S, CandidateSet, cast<CXXRecordDecl>(Record->getDecl()), CurInitExpr);

  // Perform overload resolution.
  OverloadCandidateSet::iterator Best;
  OverloadingResult OR = CandidateSet.BestViableFunction(S, Loc, Best);

  PartialDiagnostic Diag = S.PDiag(diag::warn_cxx98_compat_temp_copy)
    << OR << (int)Entity.getKind() << CurInitExpr->getType()
    << CurInitExpr->getSourceRange();

  switch (OR) {
  case OR_Success:
    S.CheckConstructorAccess(Loc, cast<CXXConstructorDecl>(Best->Function),
                             Entity, Best->FoundDecl.getAccess(), Diag);
    // FIXME: Check default arguments as far as that's possible.
    break;

  case OR_No_Viable_Function:
    S.Diag(Loc, Diag);
    CandidateSet.NoteCandidates(S, OCD_AllCandidates, CurInitExpr);
    break;

  case OR_Ambiguous:
    S.Diag(Loc, Diag);
    CandidateSet.NoteCandidates(S, OCD_ViableCandidates, CurInitExpr);
    break;

  case OR_Deleted:
    S.Diag(Loc, Diag);
    S.NoteDeletedFunction(Best->Function);
    break;
  }
}

void InitializationSequence::PrintInitLocationNote(Sema &S,
                                              const InitializedEntity &Entity) {
  if (Entity.isParameterKind() && Entity.getDecl()) {
    if (Entity.getDecl()->getLocation().isInvalid())
      return;

    if (Entity.getDecl()->getDeclName())
      S.Diag(Entity.getDecl()->getLocation(), diag::note_parameter_named_here)
        << Entity.getDecl()->getDeclName();
    else
      S.Diag(Entity.getDecl()->getLocation(), diag::note_parameter_here);
  }
  else if (Entity.getKind() == InitializedEntity::EK_RelatedResult &&
           Entity.getMethodDecl())
    S.Diag(Entity.getMethodDecl()->getLocation(),
           diag::note_method_return_type_change)
      << Entity.getMethodDecl()->getDeclName();
}

static bool isReferenceBinding(const InitializationSequence::Step &s) {
  return s.Kind == InitializationSequence::SK_BindReference ||
         s.Kind == InitializationSequence::SK_BindReferenceToTemporary;
}

/// Returns true if the parameters describe a constructor initialization of
/// an explicit temporary object, e.g. "Point(x, y)".
static bool isExplicitTemporary(const InitializedEntity &Entity,
                                const InitializationKind &Kind,
                                unsigned NumArgs) {
  switch (Entity.getKind()) {
  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_CompoundLiteralInit:
  case InitializedEntity::EK_RelatedResult:
    break;
  default:
    return false;
  }

  switch (Kind.getKind()) {
  case InitializationKind::IK_DirectList:
    return true;
  // FIXME: Hack to work around cast weirdness.
  case InitializationKind::IK_Direct:
  case InitializationKind::IK_Value:
    return NumArgs != 1;
  default:
    return false;
  }
}

static ExprResult
PerformConstructorInitialization(Sema &S,
                                 const InitializedEntity &Entity,
                                 const InitializationKind &Kind,
                                 MultiExprArg Args,
                                 const InitializationSequence::Step& Step,
                                 bool &ConstructorInitRequiresZeroInit,
                                 bool IsListInitialization,
                                 bool IsStdInitListInitialization,
                                 SourceLocation LBraceLoc,
                                 SourceLocation RBraceLoc) {
  unsigned NumArgs = Args.size();
  CXXConstructorDecl *Constructor
    = cast<CXXConstructorDecl>(Step.Function.Function);
  bool HadMultipleCandidates = Step.Function.HadMultipleCandidates;

  // Build a call to the selected constructor.
  SmallVector<Expr*, 8> ConstructorArgs;
  SourceLocation Loc = (Kind.isCopyInit() && Kind.getEqualLoc().isValid())
                         ? Kind.getEqualLoc()
                         : Kind.getLocation();

  if (Kind.getKind() == InitializationKind::IK_Default) {
    // Force even a trivial, implicit default constructor to be
    // semantically checked. We do this explicitly because we don't build
    // the definition for completely trivial constructors.
    assert(Constructor->getParent() && "No parent class for constructor.");
    if (Constructor->isDefaulted() && Constructor->isDefaultConstructor() &&
        Constructor->isTrivial() && !Constructor->isUsed(false))
      S.DefineImplicitDefaultConstructor(Loc, Constructor);
  }

  ExprResult CurInit((Expr *)nullptr);

  // C++ [over.match.copy]p1:
  //   - When initializing a temporary to be bound to the first parameter 
  //     of a constructor that takes a reference to possibly cv-qualified 
  //     T as its first argument, called with a single argument in the 
  //     context of direct-initialization, explicit conversion functions
  //     are also considered.
  bool AllowExplicitConv = Kind.AllowExplicit() && !Kind.isCopyInit() &&
                           Args.size() == 1 && 
                           Constructor->isCopyOrMoveConstructor();

  // Determine the arguments required to actually perform the constructor
  // call.
  if (S.CompleteConstructorCall(Constructor, Args,
                                Loc, ConstructorArgs,
                                AllowExplicitConv,
                                IsListInitialization))
    return ExprError();


  if (isExplicitTemporary(Entity, Kind, NumArgs)) {
    // An explicitly-constructed temporary, e.g., X(1, 2).
    S.MarkFunctionReferenced(Loc, Constructor);
    if (S.DiagnoseUseOfDecl(Constructor, Loc))
      return ExprError();

    TypeSourceInfo *TSInfo = Entity.getTypeSourceInfo();
    if (!TSInfo)
      TSInfo = S.Context.getTrivialTypeSourceInfo(Entity.getType(), Loc);
    SourceRange ParenOrBraceRange =
      (Kind.getKind() == InitializationKind::IK_DirectList)
      ? SourceRange(LBraceLoc, RBraceLoc)
      : Kind.getParenRange();

    CurInit = new (S.Context) CXXTemporaryObjectExpr(
        S.Context, Constructor, TSInfo, ConstructorArgs, ParenOrBraceRange,
        HadMultipleCandidates, IsListInitialization,
        IsStdInitListInitialization, ConstructorInitRequiresZeroInit);
  } else {
    CXXConstructExpr::ConstructionKind ConstructKind =
      CXXConstructExpr::CK_Complete;

    if (Entity.getKind() == InitializedEntity::EK_Base) {
      ConstructKind = Entity.getBaseSpecifier()->isVirtual() ?
        CXXConstructExpr::CK_VirtualBase :
        CXXConstructExpr::CK_NonVirtualBase;
    } else if (Entity.getKind() == InitializedEntity::EK_Delegating) {
      ConstructKind = CXXConstructExpr::CK_Delegating;
    }

    // Only get the parenthesis or brace range if it is a list initialization or
    // direct construction.
    SourceRange ParenOrBraceRange;
    if (IsListInitialization)
      ParenOrBraceRange = SourceRange(LBraceLoc, RBraceLoc);
    else if (Kind.getKind() == InitializationKind::IK_Direct)
      ParenOrBraceRange = Kind.getParenRange();

    // If the entity allows NRVO, mark the construction as elidable
    // unconditionally.
    if (Entity.allowsNRVO())
      CurInit = S.BuildCXXConstructExpr(Loc, Entity.getType(),
                                        Constructor, /*Elidable=*/true,
                                        ConstructorArgs,
                                        HadMultipleCandidates,
                                        IsListInitialization,
                                        IsStdInitListInitialization,
                                        ConstructorInitRequiresZeroInit,
                                        ConstructKind,
                                        ParenOrBraceRange);
    else
      CurInit = S.BuildCXXConstructExpr(Loc, Entity.getType(),
                                        Constructor,
                                        ConstructorArgs,
                                        HadMultipleCandidates,
                                        IsListInitialization,
                                        IsStdInitListInitialization,
                                        ConstructorInitRequiresZeroInit,
                                        ConstructKind,
                                        ParenOrBraceRange);
  }
  if (CurInit.isInvalid())
    return ExprError();

  // Only check access if all of that succeeded.
  S.CheckConstructorAccess(Loc, Constructor, Entity,
                           Step.Function.FoundDecl.getAccess());
  if (S.DiagnoseUseOfDecl(Step.Function.FoundDecl, Loc))
    return ExprError();

  if (shouldBindAsTemporary(Entity))
    CurInit = S.MaybeBindToTemporary(CurInit.get());

  return CurInit;
}

/// Determine whether the specified InitializedEntity definitely has a lifetime
/// longer than the current full-expression. Conservatively returns false if
/// it's unclear.
static bool
InitializedEntityOutlivesFullExpression(const InitializedEntity &Entity) {
  const InitializedEntity *Top = &Entity;
  while (Top->getParent())
    Top = Top->getParent();

  switch (Top->getKind()) {
  case InitializedEntity::EK_Variable:
  case InitializedEntity::EK_Result:
  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_Delegating:
    return true;

  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_BlockElement:
  case InitializedEntity::EK_ComplexElement:
    // Could not determine what the full initialization is. Assume it might not
    // outlive the full-expression.
    return false;

  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Parameter_CF_Audited:
  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_LambdaCapture:
  case InitializedEntity::EK_CompoundLiteralInit:
  case InitializedEntity::EK_RelatedResult:
    // The entity being initialized might not outlive the full-expression.
    return false;
  }

  llvm_unreachable("unknown entity kind");
}

/// Determine the declaration which an initialized entity ultimately refers to,
/// for the purpose of lifetime-extending a temporary bound to a reference in
/// the initialization of \p Entity.
static const InitializedEntity *getEntityForTemporaryLifetimeExtension(
    const InitializedEntity *Entity,
    const InitializedEntity *FallbackDecl = nullptr) {
  // C++11 [class.temporary]p5:
  switch (Entity->getKind()) {
  case InitializedEntity::EK_Variable:
    //   The temporary [...] persists for the lifetime of the reference
    return Entity;

  case InitializedEntity::EK_Member:
    // For subobjects, we look at the complete object.
    if (Entity->getParent())
      return getEntityForTemporaryLifetimeExtension(Entity->getParent(),
                                                    Entity);

    //   except:
    //   -- A temporary bound to a reference member in a constructor's
    //      ctor-initializer persists until the constructor exits.
    return Entity;

  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Parameter_CF_Audited:
    //   -- A temporary bound to a reference parameter in a function call
    //      persists until the completion of the full-expression containing
    //      the call.
  case InitializedEntity::EK_Result:
    //   -- The lifetime of a temporary bound to the returned value in a
    //      function return statement is not extended; the temporary is
    //      destroyed at the end of the full-expression in the return statement.
  case InitializedEntity::EK_New:
    //   -- A temporary bound to a reference in a new-initializer persists
    //      until the completion of the full-expression containing the
    //      new-initializer.
    return nullptr;

  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_CompoundLiteralInit:
  case InitializedEntity::EK_RelatedResult:
    // We don't yet know the storage duration of the surrounding temporary.
    // Assume it's got full-expression duration for now, it will patch up our
    // storage duration if that's not correct.
    return nullptr;

  case InitializedEntity::EK_ArrayElement:
    // For subobjects, we look at the complete object.
    return getEntityForTemporaryLifetimeExtension(Entity->getParent(),
                                                  FallbackDecl);

  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_Delegating:
    // We can reach this case for aggregate initialization in a constructor:
    //   struct A { int &&r; };
    //   struct B : A { B() : A{0} {} };
    // In this case, use the innermost field decl as the context.
    return FallbackDecl;

  case InitializedEntity::EK_BlockElement:
  case InitializedEntity::EK_LambdaCapture:
  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_ComplexElement:
    return nullptr;
  }
  llvm_unreachable("unknown entity kind");
}

static void performLifetimeExtension(Expr *Init,
                                     const InitializedEntity *ExtendingEntity);

/// Update a glvalue expression that is used as the initializer of a reference
/// to note that its lifetime is extended.
/// \return \c true if any temporary had its lifetime extended.
static bool
performReferenceExtension(Expr *Init,
                          const InitializedEntity *ExtendingEntity) {
  // Walk past any constructs which we can lifetime-extend across.
  Expr *Old;
  do {
    Old = Init;

    if (InitListExpr *ILE = dyn_cast<InitListExpr>(Init)) {
      if (ILE->getNumInits() == 1 && ILE->isGLValue()) {
        // This is just redundant braces around an initializer. Step over it.
        Init = ILE->getInit(0);
      }
    }

    // Step over any subobject adjustments; we may have a materialized
    // temporary inside them.
    SmallVector<const Expr *, 2> CommaLHSs;
    SmallVector<SubobjectAdjustment, 2> Adjustments;
    Init = const_cast<Expr *>(
        Init->skipRValueSubobjectAdjustments(CommaLHSs, Adjustments));

    // Per current approach for DR1376, look through casts to reference type
    // when performing lifetime extension.
    if (CastExpr *CE = dyn_cast<CastExpr>(Init))
      if (CE->getSubExpr()->isGLValue())
        Init = CE->getSubExpr();

    // FIXME: Per DR1213, subscripting on an array temporary produces an xvalue.
    // It's unclear if binding a reference to that xvalue extends the array
    // temporary.
  } while (Init != Old);

  if (MaterializeTemporaryExpr *ME = dyn_cast<MaterializeTemporaryExpr>(Init)) {
    // Update the storage duration of the materialized temporary.
    // FIXME: Rebuild the expression instead of mutating it.
    ME->setExtendingDecl(ExtendingEntity->getDecl(),
                         ExtendingEntity->allocateManglingNumber());
    performLifetimeExtension(ME->GetTemporaryExpr(), ExtendingEntity);
    return true;
  }

  return false;
}

/// Update a prvalue expression that is going to be materialized as a
/// lifetime-extended temporary.
static void performLifetimeExtension(Expr *Init,
                                     const InitializedEntity *ExtendingEntity) {
  // Dig out the expression which constructs the extended temporary.
  SmallVector<const Expr *, 2> CommaLHSs;
  SmallVector<SubobjectAdjustment, 2> Adjustments;
  Init = const_cast<Expr *>(
      Init->skipRValueSubobjectAdjustments(CommaLHSs, Adjustments));

  if (CXXBindTemporaryExpr *BTE = dyn_cast<CXXBindTemporaryExpr>(Init))
    Init = BTE->getSubExpr();

  if (CXXStdInitializerListExpr *ILE =
          dyn_cast<CXXStdInitializerListExpr>(Init)) {
    performReferenceExtension(ILE->getSubExpr(), ExtendingEntity);
    return;
  }

  if (InitListExpr *ILE = dyn_cast<InitListExpr>(Init)) {
    if (ILE->getType()->isArrayType()) {
      for (unsigned I = 0, N = ILE->getNumInits(); I != N; ++I)
        performLifetimeExtension(ILE->getInit(I), ExtendingEntity);
      return;
    }

    if (CXXRecordDecl *RD = ILE->getType()->getAsCXXRecordDecl()) {
      assert(RD->isAggregate() && "aggregate init on non-aggregate");

      // If we lifetime-extend a braced initializer which is initializing an
      // aggregate, and that aggregate contains reference members which are
      // bound to temporaries, those temporaries are also lifetime-extended.
      if (RD->isUnion() && ILE->getInitializedFieldInUnion() &&
          ILE->getInitializedFieldInUnion()->getType()->isReferenceType())
        performReferenceExtension(ILE->getInit(0), ExtendingEntity);
      else {
        unsigned Index = 0;
        for (const auto *I : RD->fields()) {
          if (Index >= ILE->getNumInits())
            break;
          if (I->isUnnamedBitfield())
            continue;
          Expr *SubInit = ILE->getInit(Index);
          if (I->getType()->isReferenceType())
            performReferenceExtension(SubInit, ExtendingEntity);
          else if (isa<InitListExpr>(SubInit) ||
                   isa<CXXStdInitializerListExpr>(SubInit))
            // This may be either aggregate-initialization of a member or
            // initialization of a std::initializer_list object. Either way,
            // we should recursively lifetime-extend that initializer.
            performLifetimeExtension(SubInit, ExtendingEntity);
          ++Index;
        }
      }
    }
  }
}

static void warnOnLifetimeExtension(Sema &S, const InitializedEntity &Entity,
                                    const Expr *Init, bool IsInitializerList,
                                    const ValueDecl *ExtendingDecl) {
  // Warn if a field lifetime-extends a temporary.
  if (isa<FieldDecl>(ExtendingDecl)) {
    if (IsInitializerList) {
      S.Diag(Init->getExprLoc(), diag::warn_dangling_std_initializer_list)
        << /*at end of constructor*/true;
      return;
    }

    bool IsSubobjectMember = false;
    for (const InitializedEntity *Ent = Entity.getParent(); Ent;
         Ent = Ent->getParent()) {
      if (Ent->getKind() != InitializedEntity::EK_Base) {
        IsSubobjectMember = true;
        break;
      }
    }
    S.Diag(Init->getExprLoc(),
           diag::warn_bind_ref_member_to_temporary)
      << ExtendingDecl << Init->getSourceRange()
      << IsSubobjectMember << IsInitializerList;
    if (IsSubobjectMember)
      S.Diag(ExtendingDecl->getLocation(),
             diag::note_ref_subobject_of_member_declared_here);
    else
      S.Diag(ExtendingDecl->getLocation(),
             diag::note_ref_or_ptr_member_declared_here)
        << /*is pointer*/false;
  }
}

static void DiagnoseNarrowingInInitList(Sema &S,
                                        const ImplicitConversionSequence &ICS,
                                        QualType PreNarrowingType,
                                        QualType EntityType,
                                        const Expr *PostInit);

/// Provide warnings when std::move is used on construction.
static void CheckMoveOnConstruction(Sema &S, const Expr *InitExpr,
                                    bool IsReturnStmt) {
  if (!InitExpr)
    return;

  if (!S.ActiveTemplateInstantiations.empty())
    return;

  QualType DestType = InitExpr->getType();
  if (!DestType->isRecordType())
    return;

  unsigned DiagID = 0;
  if (IsReturnStmt) {
    const CXXConstructExpr *CCE =
        dyn_cast<CXXConstructExpr>(InitExpr->IgnoreParens());
    if (!CCE || CCE->getNumArgs() != 1)
      return;

    if (!CCE->getConstructor()->isCopyOrMoveConstructor())
      return;

    InitExpr = CCE->getArg(0)->IgnoreImpCasts();
  }

  // Find the std::move call and get the argument.
  const CallExpr *CE = dyn_cast<CallExpr>(InitExpr->IgnoreParens());
  if (!CE || CE->getNumArgs() != 1)
    return;

  const FunctionDecl *MoveFunction = CE->getDirectCallee();
  if (!MoveFunction || !MoveFunction->isInStdNamespace() ||
      !MoveFunction->getIdentifier() ||
      !MoveFunction->getIdentifier()->isStr("move"))
    return;

  const Expr *Arg = CE->getArg(0)->IgnoreImplicit();

  if (IsReturnStmt) {
    const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreParenImpCasts());
    if (!DRE || DRE->refersToEnclosingVariableOrCapture())
      return;

    const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VD || !VD->hasLocalStorage())
      return;

    QualType SourceType = VD->getType();
    if (!SourceType->isRecordType())
      return;

    if (!S.Context.hasSameUnqualifiedType(DestType, SourceType)) {
      return;
    }

    // If we're returning a function parameter, copy elision
    // is not possible.
    if (isa<ParmVarDecl>(VD))
      DiagID = diag::warn_redundant_move_on_return;
    else
      DiagID = diag::warn_pessimizing_move_on_return;
  } else {
    DiagID = diag::warn_pessimizing_move_on_initialization;
    const Expr *ArgStripped = Arg->IgnoreImplicit()->IgnoreParens();
    if (!ArgStripped->isRValue() || !ArgStripped->getType()->isRecordType())
      return;
  }

  S.Diag(CE->getLocStart(), DiagID);

  // Get all the locations for a fix-it.  Don't emit the fix-it if any location
  // is within a macro.
  SourceLocation CallBegin = CE->getCallee()->getLocStart();
  if (CallBegin.isMacroID())
    return;
  SourceLocation RParen = CE->getRParenLoc();
  if (RParen.isMacroID())
    return;
  SourceLocation LParen;
  SourceLocation ArgLoc = Arg->getLocStart();

  // Special testing for the argument location.  Since the fix-it needs the
  // location right before the argument, the argument location can be in a
  // macro only if it is at the beginning of the macro.
  while (ArgLoc.isMacroID() &&
         S.getSourceManager().isAtStartOfImmediateMacroExpansion(ArgLoc)) {
    ArgLoc = S.getSourceManager().getImmediateExpansionRange(ArgLoc).first;
  }

  if (LParen.isMacroID())
    return;

  LParen = ArgLoc.getLocWithOffset(-1);

  S.Diag(CE->getLocStart(), diag::note_remove_move)
      << FixItHint::CreateRemoval(SourceRange(CallBegin, LParen))
      << FixItHint::CreateRemoval(SourceRange(RParen, RParen));
}

ExprResult
InitializationSequence::Perform(Sema &S,
                                const InitializedEntity &Entity,
                                const InitializationKind &Kind,
                                MultiExprArg Args,
                                QualType *ResultType) {
  if (Failed()) {
    Diagnose(S, Entity, Kind, Args);
    return ExprError();
  }
  if (!ZeroInitializationFixit.empty()) {
    unsigned DiagID = diag::err_default_init_const;
    if (Decl *D = Entity.getDecl())
      if (S.getLangOpts().MSVCCompat && D->hasAttr<SelectAnyAttr>())
        DiagID = diag::ext_default_init_const;

    // The initialization would have succeeded with this fixit. Since the fixit
    // is on the error, we need to build a valid AST in this case, so this isn't
    // handled in the Failed() branch above.
    QualType DestType = Entity.getType();
    S.Diag(Kind.getLocation(), DiagID)
        << DestType << (bool)DestType->getAs<RecordType>()
        << FixItHint::CreateInsertion(ZeroInitializationFixitLoc,
                                      ZeroInitializationFixit);
  }

  if (getKind() == DependentSequence) {
    // If the declaration is a non-dependent, incomplete array type
    // that has an initializer, then its type will be completed once
    // the initializer is instantiated.
    if (ResultType && !Entity.getType()->isDependentType() &&
        Args.size() == 1) {
      QualType DeclType = Entity.getType();
      if (const IncompleteArrayType *ArrayT
                           = S.Context.getAsIncompleteArrayType(DeclType)) {
        // FIXME: We don't currently have the ability to accurately
        // compute the length of an initializer list without
        // performing full type-checking of the initializer list
        // (since we have to determine where braces are implicitly
        // introduced and such).  So, we fall back to making the array
        // type a dependently-sized array type with no specified
        // bound.
        if (isa<InitListExpr>((Expr *)Args[0])) {
          SourceRange Brackets;

          // Scavange the location of the brackets from the entity, if we can.
          if (DeclaratorDecl *DD = Entity.getDecl()) {
            if (TypeSourceInfo *TInfo = DD->getTypeSourceInfo()) {
              TypeLoc TL = TInfo->getTypeLoc();
              if (IncompleteArrayTypeLoc ArrayLoc =
                      TL.getAs<IncompleteArrayTypeLoc>())
                Brackets = ArrayLoc.getBracketsRange();
            }
          }

          *ResultType
            = S.Context.getDependentSizedArrayType(ArrayT->getElementType(),
                                                   /*NumElts=*/nullptr,
                                                   ArrayT->getSizeModifier(),
                                       ArrayT->getIndexTypeCVRQualifiers(),
                                                   Brackets);
        }

      }
    }
    if (Kind.getKind() == InitializationKind::IK_Direct &&
        !Kind.isExplicitCast()) {
      // Rebuild the ParenListExpr.
      SourceRange ParenRange = Kind.getParenRange();
      return S.ActOnParenListExpr(ParenRange.getBegin(), ParenRange.getEnd(),
                                  Args);
    }
    assert(Kind.getKind() == InitializationKind::IK_Copy ||
           Kind.isExplicitCast() || 
           Kind.getKind() == InitializationKind::IK_DirectList);
    return ExprResult(Args[0]);
  }

  // No steps means no initialization.
  if (Steps.empty())
    return ExprResult((Expr *)nullptr);

  if (S.getLangOpts().CPlusPlus11 && Entity.getType()->isReferenceType() &&
      Args.size() == 1 && isa<InitListExpr>(Args[0]) &&
      !Entity.isParameterKind()) {
    // Produce a C++98 compatibility warning if we are initializing a reference
    // from an initializer list. For parameters, we produce a better warning
    // elsewhere.
    Expr *Init = Args[0];
    S.Diag(Init->getLocStart(), diag::warn_cxx98_compat_reference_list_init)
      << Init->getSourceRange();
  }

  // Diagnose cases where we initialize a pointer to an array temporary, and the
  // pointer obviously outlives the temporary.
  if (Args.size() == 1 && Args[0]->getType()->isArrayType() &&
      Entity.getType()->isPointerType() &&
      InitializedEntityOutlivesFullExpression(Entity)) {
    Expr *Init = Args[0];
    Expr::LValueClassification Kind = Init->ClassifyLValue(S.Context);
    if (Kind == Expr::LV_ClassTemporary || Kind == Expr::LV_ArrayTemporary)
      S.Diag(Init->getLocStart(), diag::warn_temporary_array_to_pointer_decay)
        << Init->getSourceRange();
  }

  QualType DestType = Entity.getType().getNonReferenceType();
  // FIXME: Ugly hack around the fact that Entity.getType() is not
  // the same as Entity.getDecl()->getType() in cases involving type merging,
  //  and we want latter when it makes sense.
  if (ResultType)
    *ResultType = Entity.getDecl() ? Entity.getDecl()->getType() :
                                     Entity.getType();

  ExprResult CurInit((Expr *)nullptr);

  // For initialization steps that start with a single initializer,
  // grab the only argument out the Args and place it into the "current"
  // initializer.
  switch (Steps.front().Kind) {
  case SK_ResolveAddressOfOverloadedFunction:
  case SK_CastDerivedToBaseRValue:
  case SK_CastDerivedToBaseXValue:
  case SK_CastDerivedToBaseLValue:
  case SK_BindReference:
  case SK_BindReferenceToTemporary:
  case SK_ExtraneousCopyToTemporary:
  case SK_UserConversion:
  case SK_QualificationConversionLValue:
  case SK_QualificationConversionXValue:
  case SK_QualificationConversionRValue:
  case SK_AtomicConversion:
  case SK_LValueToRValue:
  case SK_ConversionSequence:
  case SK_ConversionSequenceNoNarrowing:
  case SK_ListInitialization:
  case SK_UnwrapInitList:
  case SK_RewrapInitList:
  case SK_CAssignment:
  case SK_StringInit:
  case SK_ObjCObjectConversion:
  case SK_ArrayInit:
  case SK_ParenthesizedArrayInit:
  case SK_PassByIndirectCopyRestore:
  case SK_PassByIndirectRestore:
  case SK_ProduceObjCObject:
  case SK_StdInitializerList:
  case SK_OCLSamplerInit:
  case SK_OCLZeroEvent: {
    assert(Args.size() == 1);
    CurInit = Args[0];
    if (!CurInit.get()) return ExprError();
    break;
  }

  case SK_ConstructorInitialization:
  case SK_ConstructorInitializationFromList:
  case SK_StdInitializerListConstructorCall:
  case SK_ZeroInitialization:
    break;
  }

  // Walk through the computed steps for the initialization sequence,
  // performing the specified conversions along the way.
  bool ConstructorInitRequiresZeroInit = false;
  for (step_iterator Step = step_begin(), StepEnd = step_end();
       Step != StepEnd; ++Step) {
    if (CurInit.isInvalid())
      return ExprError();

    QualType SourceType = CurInit.get() ? CurInit.get()->getType() : QualType();

    switch (Step->Kind) {
    case SK_ResolveAddressOfOverloadedFunction:
      // Overload resolution determined which function invoke; update the
      // initializer to reflect that choice.
      S.CheckAddressOfMemberAccess(CurInit.get(), Step->Function.FoundDecl);
      if (S.DiagnoseUseOfDecl(Step->Function.FoundDecl, Kind.getLocation()))
        return ExprError();
      CurInit = S.FixOverloadedFunctionReference(CurInit,
                                                 Step->Function.FoundDecl,
                                                 Step->Function.Function);
      break;

    case SK_CastDerivedToBaseRValue:
    case SK_CastDerivedToBaseXValue:
    case SK_CastDerivedToBaseLValue: {
      // We have a derived-to-base cast that produces either an rvalue or an
      // lvalue. Perform that cast.

      CXXCastPath BasePath;

      // Casts to inaccessible base classes are allowed with C-style casts.
      bool IgnoreBaseAccess = Kind.isCStyleOrFunctionalCast();
      if (S.CheckDerivedToBaseConversion(SourceType, Step->Type,
                                         CurInit.get()->getLocStart(),
                                         CurInit.get()->getSourceRange(),
                                         &BasePath, IgnoreBaseAccess))
        return ExprError();

      ExprValueKind VK =
          Step->Kind == SK_CastDerivedToBaseLValue ?
              VK_LValue :
              (Step->Kind == SK_CastDerivedToBaseXValue ?
                   VK_XValue :
                   VK_RValue);
      CurInit =
          ImplicitCastExpr::Create(S.Context, Step->Type, CK_DerivedToBase,
                                   CurInit.get(), &BasePath, VK);
      break;
    }

    case SK_BindReference:
      // References cannot bind to bit-fields (C++ [dcl.init.ref]p5).
      if (CurInit.get()->refersToBitField()) {
        // We don't necessarily have an unambiguous source bit-field.
        FieldDecl *BitField = CurInit.get()->getSourceBitField();
        S.Diag(Kind.getLocation(), diag::err_reference_bind_to_bitfield)
          << Entity.getType().isVolatileQualified()
          << (BitField ? BitField->getDeclName() : DeclarationName())
          << (BitField != nullptr)
          << CurInit.get()->getSourceRange();
        if (BitField)
          S.Diag(BitField->getLocation(), diag::note_bitfield_decl);

        return ExprError();
      }

      if (CurInit.get()->refersToVectorElement()) {
        // References cannot bind to vector elements.
        S.Diag(Kind.getLocation(), diag::err_reference_bind_to_vector_element)
          << Entity.getType().isVolatileQualified()
          << CurInit.get()->getSourceRange();
        PrintInitLocationNote(S, Entity);
        return ExprError();
      }

      // Reference binding does not have any corresponding ASTs.

      // Check exception specifications
      if (S.CheckExceptionSpecCompatibility(CurInit.get(), DestType))
        return ExprError();

      // Even though we didn't materialize a temporary, the binding may still
      // extend the lifetime of a temporary. This happens if we bind a reference
      // to the result of a cast to reference type.
      if (const InitializedEntity *ExtendingEntity =
              getEntityForTemporaryLifetimeExtension(&Entity))
        if (performReferenceExtension(CurInit.get(), ExtendingEntity))
          warnOnLifetimeExtension(S, Entity, CurInit.get(),
                                  /*IsInitializerList=*/false,
                                  ExtendingEntity->getDecl());

      break;

    case SK_BindReferenceToTemporary: {
      // Make sure the "temporary" is actually an rvalue.
      assert(CurInit.get()->isRValue() && "not a temporary");

      // Check exception specifications
      if (S.CheckExceptionSpecCompatibility(CurInit.get(), DestType))
        return ExprError();

      // Materialize the temporary into memory.
      MaterializeTemporaryExpr *MTE = new (S.Context) MaterializeTemporaryExpr(
          Entity.getType().getNonReferenceType(), CurInit.get(),
          Entity.getType()->isLValueReferenceType());

      // Maybe lifetime-extend the temporary's subobjects to match the
      // entity's lifetime.
      if (const InitializedEntity *ExtendingEntity =
              getEntityForTemporaryLifetimeExtension(&Entity))
        if (performReferenceExtension(MTE, ExtendingEntity))
          warnOnLifetimeExtension(S, Entity, CurInit.get(), /*IsInitializerList=*/false,
                                  ExtendingEntity->getDecl());

      // If we're binding to an Objective-C object that has lifetime, we
      // need cleanups. Likewise if we're extending this temporary to automatic
      // storage duration -- we need to register its cleanup during the
      // full-expression's cleanups.
      if ((S.getLangOpts().ObjCAutoRefCount &&
           MTE->getType()->isObjCLifetimeType()) ||
          (MTE->getStorageDuration() == SD_Automatic &&
           MTE->getType().isDestructedType()))
        S.ExprNeedsCleanups = true;

      CurInit = MTE;
      break;
    }

    case SK_ExtraneousCopyToTemporary:
      CurInit = CopyObject(S, Step->Type, Entity, CurInit,
                           /*IsExtraneousCopy=*/true);
      break;

    case SK_UserConversion: {
      // We have a user-defined conversion that invokes either a constructor
      // or a conversion function.
      CastKind CastKind;
      bool IsCopy = false;
      FunctionDecl *Fn = Step->Function.Function;
      DeclAccessPair FoundFn = Step->Function.FoundDecl;
      bool HadMultipleCandidates = Step->Function.HadMultipleCandidates;
      bool CreatedObject = false;
      if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Fn)) {
        // Build a call to the selected constructor.
        SmallVector<Expr*, 8> ConstructorArgs;
        SourceLocation Loc = CurInit.get()->getLocStart();
        CurInit.get(); // Ownership transferred into MultiExprArg, below.

        // Determine the arguments required to actually perform the constructor
        // call.
        Expr *Arg = CurInit.get();
        if (S.CompleteConstructorCall(Constructor,
                                      MultiExprArg(&Arg, 1),
                                      Loc, ConstructorArgs))
          return ExprError();

        // Build an expression that constructs a temporary.
        CurInit = S.BuildCXXConstructExpr(Loc, Step->Type, Constructor,
                                          ConstructorArgs,
                                          HadMultipleCandidates,
                                          /*ListInit*/ false,
                                          /*StdInitListInit*/ false,
                                          /*ZeroInit*/ false,
                                          CXXConstructExpr::CK_Complete,
                                          SourceRange());
        if (CurInit.isInvalid())
          return ExprError();

        S.CheckConstructorAccess(Kind.getLocation(), Constructor, Entity,
                                 FoundFn.getAccess());
        if (S.DiagnoseUseOfDecl(FoundFn, Kind.getLocation()))
          return ExprError();

        CastKind = CK_ConstructorConversion;
        QualType Class = S.Context.getTypeDeclType(Constructor->getParent());
        if (S.Context.hasSameUnqualifiedType(SourceType, Class) ||
            S.IsDerivedFrom(Loc, SourceType, Class))
          IsCopy = true;

        CreatedObject = true;
      } else {
        // Build a call to the conversion function.
        CXXConversionDecl *Conversion = cast<CXXConversionDecl>(Fn);
        S.CheckMemberOperatorAccess(Kind.getLocation(), CurInit.get(), nullptr,
                                    FoundFn);
        if (S.DiagnoseUseOfDecl(FoundFn, Kind.getLocation()))
          return ExprError();

        // FIXME: Should we move this initialization into a separate
        // derived-to-base conversion? I believe the answer is "no", because
        // we don't want to turn off access control here for c-style casts.
        ExprResult CurInitExprRes =
          S.PerformObjectArgumentInitialization(CurInit.get(),
                                                /*Qualifier=*/nullptr,
                                                FoundFn, Conversion);
        if(CurInitExprRes.isInvalid())
          return ExprError();
        CurInit = CurInitExprRes;

        // Build the actual call to the conversion function.
        CurInit = S.BuildCXXMemberCallExpr(CurInit.get(), FoundFn, Conversion,
                                           HadMultipleCandidates);
        if (CurInit.isInvalid() || !CurInit.get())
          return ExprError();

        CastKind = CK_UserDefinedConversion;

        CreatedObject = Conversion->getReturnType()->isRecordType();
      }

      bool RequiresCopy = !IsCopy && !isReferenceBinding(Steps.back());
      bool MaybeBindToTemp = RequiresCopy || shouldBindAsTemporary(Entity);

      if (!MaybeBindToTemp && CreatedObject && shouldDestroyTemporary(Entity)) {
        QualType T = CurInit.get()->getType();
        if (const RecordType *Record = T->getAs<RecordType>()) {
          CXXDestructorDecl *Destructor
            = S.LookupDestructor(cast<CXXRecordDecl>(Record->getDecl()));
          S.CheckDestructorAccess(CurInit.get()->getLocStart(), Destructor,
                                  S.PDiag(diag::err_access_dtor_temp) << T);
          S.MarkFunctionReferenced(CurInit.get()->getLocStart(), Destructor);
          if (S.DiagnoseUseOfDecl(Destructor, CurInit.get()->getLocStart()))
            return ExprError();
        }
      }

      CurInit = ImplicitCastExpr::Create(S.Context, CurInit.get()->getType(),
                                         CastKind, CurInit.get(), nullptr,
                                         CurInit.get()->getValueKind());
      if (MaybeBindToTemp)
        CurInit = S.MaybeBindToTemporary(CurInit.getAs<Expr>());
      if (RequiresCopy)
        CurInit = CopyObject(S, Entity.getType().getNonReferenceType(), Entity,
                             CurInit, /*IsExtraneousCopy=*/false);
      break;
    }

    case SK_QualificationConversionLValue:
    case SK_QualificationConversionXValue:
    case SK_QualificationConversionRValue: {
      // Perform a qualification conversion; these can never go wrong.
      ExprValueKind VK =
          Step->Kind == SK_QualificationConversionLValue ?
              VK_LValue :
              (Step->Kind == SK_QualificationConversionXValue ?
                   VK_XValue :
                   VK_RValue);
      CurInit = S.ImpCastExprToType(CurInit.get(), Step->Type, CK_NoOp, VK);
      break;
    }

    case SK_AtomicConversion: {
      assert(CurInit.get()->isRValue() && "cannot convert glvalue to atomic");
      CurInit = S.ImpCastExprToType(CurInit.get(), Step->Type,
                                    CK_NonAtomicToAtomic, VK_RValue);
      break;
    }

    case SK_LValueToRValue: {
      assert(CurInit.get()->isGLValue() && "cannot load from a prvalue");
      CurInit = ImplicitCastExpr::Create(S.Context, Step->Type,
                                         CK_LValueToRValue, CurInit.get(),
                                         /*BasePath=*/nullptr, VK_RValue);
      break;
    }

    case SK_ConversionSequence:
    case SK_ConversionSequenceNoNarrowing: {
      Sema::CheckedConversionKind CCK
        = Kind.isCStyleCast()? Sema::CCK_CStyleCast
        : Kind.isFunctionalCast()? Sema::CCK_FunctionalCast
        : Kind.isExplicitCast()? Sema::CCK_OtherCast
        : Sema::CCK_ImplicitConversion;
      ExprResult CurInitExprRes =
        S.PerformImplicitConversion(CurInit.get(), Step->Type, *Step->ICS,
                                    getAssignmentAction(Entity), CCK);
      if (CurInitExprRes.isInvalid())
        return ExprError();
      CurInit = CurInitExprRes;

      if (Step->Kind == SK_ConversionSequenceNoNarrowing &&
          S.getLangOpts().CPlusPlus && !CurInit.get()->isValueDependent())
        DiagnoseNarrowingInInitList(S, *Step->ICS, SourceType, Entity.getType(),
                                    CurInit.get());
      break;
    }

    case SK_ListInitialization: {
      InitListExpr *InitList = cast<InitListExpr>(CurInit.get());
      // If we're not initializing the top-level entity, we need to create an
      // InitializeTemporary entity for our target type.
      QualType Ty = Step->Type;
      bool IsTemporary = !S.Context.hasSameType(Entity.getType(), Ty);
      InitializedEntity TempEntity = InitializedEntity::InitializeTemporary(Ty);
      InitializedEntity InitEntity = IsTemporary ? TempEntity : Entity;
      InitListChecker PerformInitList(S, InitEntity,
          InitList, Ty, /*VerifyOnly=*/false);
      if (PerformInitList.HadError())
        return ExprError();

      // Hack: We must update *ResultType if available in order to set the
      // bounds of arrays, e.g. in 'int ar[] = {1, 2, 3};'.
      // Worst case: 'const int (&arref)[] = {1, 2, 3};'.
      if (ResultType &&
          ResultType->getNonReferenceType()->isIncompleteArrayType()) {
        if ((*ResultType)->isRValueReferenceType())
          Ty = S.Context.getRValueReferenceType(Ty);
        else if ((*ResultType)->isLValueReferenceType())
          Ty = S.Context.getLValueReferenceType(Ty,
            (*ResultType)->getAs<LValueReferenceType>()->isSpelledAsLValue());
        *ResultType = Ty;
      }

      InitListExpr *StructuredInitList =
          PerformInitList.getFullyStructuredList();
      CurInit.get();
      CurInit = shouldBindAsTemporary(InitEntity)
          ? S.MaybeBindToTemporary(StructuredInitList)
          : StructuredInitList;
      break;
    }

    case SK_ConstructorInitializationFromList: {
      // When an initializer list is passed for a parameter of type "reference
      // to object", we don't get an EK_Temporary entity, but instead an
      // EK_Parameter entity with reference type.
      // FIXME: This is a hack. What we really should do is create a user
      // conversion step for this case, but this makes it considerably more
      // complicated. For now, this will do.
      InitializedEntity TempEntity = InitializedEntity::InitializeTemporary(
                                        Entity.getType().getNonReferenceType());
      bool UseTemporary = Entity.getType()->isReferenceType();
      assert(Args.size() == 1 && "expected a single argument for list init");
      InitListExpr *InitList = cast<InitListExpr>(Args[0]);
      S.Diag(InitList->getExprLoc(), diag::warn_cxx98_compat_ctor_list_init)
        << InitList->getSourceRange();
      MultiExprArg Arg(InitList->getInits(), InitList->getNumInits());
      CurInit = PerformConstructorInitialization(S, UseTemporary ? TempEntity :
                                                                   Entity,
                                                 Kind, Arg, *Step,
                                               ConstructorInitRequiresZeroInit,
                                               /*IsListInitialization*/true,
                                               /*IsStdInitListInit*/false,
                                               InitList->getLBraceLoc(),
                                               InitList->getRBraceLoc());
      break;
    }

    case SK_UnwrapInitList:
      CurInit = cast<InitListExpr>(CurInit.get())->getInit(0);
      break;

    case SK_RewrapInitList: {
      Expr *E = CurInit.get();
      InitListExpr *Syntactic = Step->WrappingSyntacticList;
      InitListExpr *ILE = new (S.Context) InitListExpr(S.Context,
          Syntactic->getLBraceLoc(), E, Syntactic->getRBraceLoc());
      ILE->setSyntacticForm(Syntactic);
      ILE->setType(E->getType());
      ILE->setValueKind(E->getValueKind());
      CurInit = ILE;
      break;
    }

    case SK_ConstructorInitialization:
    case SK_StdInitializerListConstructorCall: {
      // When an initializer list is passed for a parameter of type "reference
      // to object", we don't get an EK_Temporary entity, but instead an
      // EK_Parameter entity with reference type.
      // FIXME: This is a hack. What we really should do is create a user
      // conversion step for this case, but this makes it considerably more
      // complicated. For now, this will do.
      InitializedEntity TempEntity = InitializedEntity::InitializeTemporary(
                                        Entity.getType().getNonReferenceType());
      bool UseTemporary = Entity.getType()->isReferenceType();
      bool IsStdInitListInit =
          Step->Kind == SK_StdInitializerListConstructorCall;
      CurInit = PerformConstructorInitialization(
          S, UseTemporary ? TempEntity : Entity, Kind, Args, *Step,
          ConstructorInitRequiresZeroInit,
          /*IsListInitialization*/IsStdInitListInit,
          /*IsStdInitListInitialization*/IsStdInitListInit,
          /*LBraceLoc*/SourceLocation(),
          /*RBraceLoc*/SourceLocation());
      break;
    }

    case SK_ZeroInitialization: {
      step_iterator NextStep = Step;
      ++NextStep;
      if (NextStep != StepEnd &&
          (NextStep->Kind == SK_ConstructorInitialization ||
           NextStep->Kind == SK_ConstructorInitializationFromList)) {
        // The need for zero-initialization is recorded directly into
        // the call to the object's constructor within the next step.
        ConstructorInitRequiresZeroInit = true;
      } else if (Kind.getKind() == InitializationKind::IK_Value &&
                 S.getLangOpts().CPlusPlus &&
                 !Kind.isImplicitValueInit()) {
        TypeSourceInfo *TSInfo = Entity.getTypeSourceInfo();
        if (!TSInfo)
          TSInfo = S.Context.getTrivialTypeSourceInfo(Step->Type,
                                                    Kind.getRange().getBegin());

        CurInit = new (S.Context) CXXScalarValueInitExpr(
            TSInfo->getType().getNonLValueExprType(S.Context), TSInfo,
            Kind.getRange().getEnd());
      } else {
        CurInit = new (S.Context) ImplicitValueInitExpr(Step->Type);
      }
      break;
    }

    case SK_CAssignment: {
      QualType SourceType = CurInit.get()->getType();
      // Save off the initial CurInit in case we need to emit a diagnostic
      ExprResult InitialCurInit = CurInit;
      ExprResult Result = CurInit;
      Sema::AssignConvertType ConvTy =
        S.CheckSingleAssignmentConstraints(Step->Type, Result, true,
            Entity.getKind() == InitializedEntity::EK_Parameter_CF_Audited);
      if (Result.isInvalid())
        return ExprError();
      CurInit = Result;

      // If this is a call, allow conversion to a transparent union.
      ExprResult CurInitExprRes = CurInit;
      if (ConvTy != Sema::Compatible &&
          Entity.isParameterKind() &&
          S.CheckTransparentUnionArgumentConstraints(Step->Type, CurInitExprRes)
            == Sema::Compatible)
        ConvTy = Sema::Compatible;
      if (CurInitExprRes.isInvalid())
        return ExprError();
      CurInit = CurInitExprRes;

      bool Complained;
      if (S.DiagnoseAssignmentResult(ConvTy, Kind.getLocation(),
                                     Step->Type, SourceType,
                                     InitialCurInit.get(),
                                     getAssignmentAction(Entity, true),
                                     &Complained)) {
        PrintInitLocationNote(S, Entity);
        return ExprError();
      } else if (Complained)
        PrintInitLocationNote(S, Entity);
      break;
    }

    case SK_StringInit: {
      QualType Ty = Step->Type;
      CheckStringInit(CurInit.get(), ResultType ? *ResultType : Ty,
                      S.Context.getAsArrayType(Ty), S);
      break;
    }

    case SK_ObjCObjectConversion:
      CurInit = S.ImpCastExprToType(CurInit.get(), Step->Type,
                          CK_ObjCObjectLValueCast,
                          CurInit.get()->getValueKind());
      break;

    case SK_ArrayInit:
      // Okay: we checked everything before creating this step. Note that
      // this is a GNU extension.
      S.Diag(Kind.getLocation(), diag::ext_array_init_copy)
        << Step->Type << CurInit.get()->getType()
        << CurInit.get()->getSourceRange();

      // If the destination type is an incomplete array type, update the
      // type accordingly.
      if (ResultType) {
        if (const IncompleteArrayType *IncompleteDest
                           = S.Context.getAsIncompleteArrayType(Step->Type)) {
          if (const ConstantArrayType *ConstantSource
                 = S.Context.getAsConstantArrayType(CurInit.get()->getType())) {
            *ResultType = S.Context.getConstantArrayType(
                                             IncompleteDest->getElementType(),
                                             ConstantSource->getSize(),
                                             ArrayType::Normal, 0);
          }
        }
      }
      break;

    case SK_ParenthesizedArrayInit:
      // Okay: we checked everything before creating this step. Note that
      // this is a GNU extension.
      S.Diag(Kind.getLocation(), diag::ext_array_init_parens)
        << CurInit.get()->getSourceRange();
      break;

    case SK_PassByIndirectCopyRestore:
    case SK_PassByIndirectRestore:
      checkIndirectCopyRestoreSource(S, CurInit.get());
      CurInit = new (S.Context) ObjCIndirectCopyRestoreExpr(
          CurInit.get(), Step->Type,
          Step->Kind == SK_PassByIndirectCopyRestore);
      break;

    case SK_ProduceObjCObject:
      CurInit =
          ImplicitCastExpr::Create(S.Context, Step->Type, CK_ARCProduceObject,
                                   CurInit.get(), nullptr, VK_RValue);
      break;

    case SK_StdInitializerList: {
      S.Diag(CurInit.get()->getExprLoc(),
             diag::warn_cxx98_compat_initializer_list_init)
        << CurInit.get()->getSourceRange();

      // Materialize the temporary into memory.
      MaterializeTemporaryExpr *MTE = new (S.Context)
          MaterializeTemporaryExpr(CurInit.get()->getType(), CurInit.get(),
                                   /*BoundToLvalueReference=*/false);

      // Maybe lifetime-extend the array temporary's subobjects to match the
      // entity's lifetime.
      if (const InitializedEntity *ExtendingEntity =
              getEntityForTemporaryLifetimeExtension(&Entity))
        if (performReferenceExtension(MTE, ExtendingEntity))
          warnOnLifetimeExtension(S, Entity, CurInit.get(),
                                  /*IsInitializerList=*/true,
                                  ExtendingEntity->getDecl());

      // Wrap it in a construction of a std::initializer_list<T>.
      CurInit = new (S.Context) CXXStdInitializerListExpr(Step->Type, MTE);

      // Bind the result, in case the library has given initializer_list a
      // non-trivial destructor.
      if (shouldBindAsTemporary(Entity))
        CurInit = S.MaybeBindToTemporary(CurInit.get());
      break;
    }

    case SK_OCLSamplerInit: {
      assert(Step->Type->isSamplerT() && 
             "Sampler initialization on non-sampler type.");

      QualType SourceType = CurInit.get()->getType();

      if (Entity.isParameterKind()) {
        if (!SourceType->isSamplerT())
          S.Diag(Kind.getLocation(), diag::err_sampler_argument_required)
            << SourceType;
      } else if (Entity.getKind() != InitializedEntity::EK_Variable) {
        llvm_unreachable("Invalid EntityKind!");
      }

      break;
    }
    case SK_OCLZeroEvent: {
      assert(Step->Type->isEventT() && 
             "Event initialization on non-event type.");

      CurInit = S.ImpCastExprToType(CurInit.get(), Step->Type,
                                    CK_ZeroToOCLEvent,
                                    CurInit.get()->getValueKind());
      break;
    }
    }
  }

  // Diagnose non-fatal problems with the completed initialization.
  if (Entity.getKind() == InitializedEntity::EK_Member &&
      cast<FieldDecl>(Entity.getDecl())->isBitField())
    S.CheckBitFieldInitialization(Kind.getLocation(),
                                  cast<FieldDecl>(Entity.getDecl()),
                                  CurInit.get());

  // Check for std::move on construction.
  if (const Expr *E = CurInit.get()) {
    CheckMoveOnConstruction(S, E,
                            Entity.getKind() == InitializedEntity::EK_Result);
  }

  return CurInit;
}

/// Somewhere within T there is an uninitialized reference subobject.
/// Dig it out and diagnose it.
static bool DiagnoseUninitializedReference(Sema &S, SourceLocation Loc,
                                           QualType T) {
  if (T->isReferenceType()) {
    S.Diag(Loc, diag::err_reference_without_init)
      << T.getNonReferenceType();
    return true;
  }

  CXXRecordDecl *RD = T->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
  if (!RD || !RD->hasUninitializedReferenceMember())
    return false;

  for (const auto *FI : RD->fields()) {
    if (FI->isUnnamedBitfield())
      continue;

    if (DiagnoseUninitializedReference(S, FI->getLocation(), FI->getType())) {
      S.Diag(Loc, diag::note_value_initialization_here) << RD;
      return true;
    }
  }

  for (const auto &BI : RD->bases()) {
    if (DiagnoseUninitializedReference(S, BI.getLocStart(), BI.getType())) {
      S.Diag(Loc, diag::note_value_initialization_here) << RD;
      return true;
    }
  }

  return false;
}


//===----------------------------------------------------------------------===//
// Diagnose initialization failures
//===----------------------------------------------------------------------===//

/// Emit notes associated with an initialization that failed due to a
/// "simple" conversion failure.
static void emitBadConversionNotes(Sema &S, const InitializedEntity &entity,
                                   Expr *op) {
  QualType destType = entity.getType();
  if (destType.getNonReferenceType()->isObjCObjectPointerType() &&
      op->getType()->isObjCObjectPointerType()) {

    // Emit a possible note about the conversion failing because the
    // operand is a message send with a related result type.
    S.EmitRelatedResultTypeNote(op);

    // Emit a possible note about a return failing because we're
    // expecting a related result type.
    if (entity.getKind() == InitializedEntity::EK_Result)
      S.EmitRelatedResultTypeNoteForReturn(destType);
  }
}

static void diagnoseListInit(Sema &S, const InitializedEntity &Entity,
                             InitListExpr *InitList) {
  QualType DestType = Entity.getType();

  QualType E;
  if (S.getLangOpts().CPlusPlus11 && S.isStdInitializerList(DestType, &E)) {
    QualType ArrayType = S.Context.getConstantArrayType(
        E.withConst(),
        llvm::APInt(S.Context.getTypeSize(S.Context.getSizeType()),
                    InitList->getNumInits()),
        clang::ArrayType::Normal, 0);
    InitializedEntity HiddenArray =
        InitializedEntity::InitializeTemporary(ArrayType);
    return diagnoseListInit(S, HiddenArray, InitList);
  }

  if (DestType->isReferenceType()) {
    // A list-initialization failure for a reference means that we tried to
    // create a temporary of the inner type (per [dcl.init.list]p3.6) and the
    // inner initialization failed.
    QualType T = DestType->getAs<ReferenceType>()->getPointeeType();
    diagnoseListInit(S, InitializedEntity::InitializeTemporary(T), InitList);
    SourceLocation Loc = InitList->getLocStart();
    if (auto *D = Entity.getDecl())
      Loc = D->getLocation();
    S.Diag(Loc, diag::note_in_reference_temporary_list_initializer) << T;
    return;
  }

  InitListChecker DiagnoseInitList(S, Entity, InitList, DestType,
                                   /*VerifyOnly=*/false);
  assert(DiagnoseInitList.HadError() &&
         "Inconsistent init list check result.");
}

bool InitializationSequence::Diagnose(Sema &S,
                                      const InitializedEntity &Entity,
                                      const InitializationKind &Kind,
                                      ArrayRef<Expr *> Args) {
  if (!Failed())
    return false;

  QualType DestType = Entity.getType();
  switch (Failure) {
  case FK_TooManyInitsForReference:
    // FIXME: Customize for the initialized entity?
    if (Args.empty()) {
      // Dig out the reference subobject which is uninitialized and diagnose it.
      // If this is value-initialization, this could be nested some way within
      // the target type.
      assert(Kind.getKind() == InitializationKind::IK_Value ||
             DestType->isReferenceType());
      bool Diagnosed =
        DiagnoseUninitializedReference(S, Kind.getLocation(), DestType);
      assert(Diagnosed && "couldn't find uninitialized reference to diagnose");
      (void)Diagnosed;
    } else  // FIXME: diagnostic below could be better!
      S.Diag(Kind.getLocation(), diag::err_reference_has_multiple_inits)
        << SourceRange(Args.front()->getLocStart(), Args.back()->getLocEnd());
    break;

  case FK_ArrayNeedsInitList:
    S.Diag(Kind.getLocation(), diag::err_array_init_not_init_list) << 0;
    break;
  case FK_ArrayNeedsInitListOrStringLiteral:
    S.Diag(Kind.getLocation(), diag::err_array_init_not_init_list) << 1;
    break;
  case FK_ArrayNeedsInitListOrWideStringLiteral:
    S.Diag(Kind.getLocation(), diag::err_array_init_not_init_list) << 2;
    break;
  case FK_NarrowStringIntoWideCharArray:
    S.Diag(Kind.getLocation(), diag::err_array_init_narrow_string_into_wchar);
    break;
  case FK_WideStringIntoCharArray:
    S.Diag(Kind.getLocation(), diag::err_array_init_wide_string_into_char);
    break;
  case FK_IncompatWideStringIntoWideChar:
    S.Diag(Kind.getLocation(),
           diag::err_array_init_incompat_wide_string_into_wchar);
    break;
  case FK_ArrayTypeMismatch:
  case FK_NonConstantArrayInit:
    S.Diag(Kind.getLocation(),
           (Failure == FK_ArrayTypeMismatch
              ? diag::err_array_init_different_type
              : diag::err_array_init_non_constant_array))
      << DestType.getNonReferenceType()
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;

  case FK_VariableLengthArrayHasInitializer:
    S.Diag(Kind.getLocation(), diag::err_variable_object_no_init)
      << Args[0]->getSourceRange();
    break;

  case FK_AddressOfOverloadFailed: {
    DeclAccessPair Found;
    S.ResolveAddressOfOverloadedFunction(Args[0],
                                         DestType.getNonReferenceType(),
                                         true,
                                         Found);
    break;
  }

  case FK_AddressOfUnaddressableFunction: {
    auto *FD = cast<FunctionDecl>(cast<DeclRefExpr>(Args[0])->getDecl());
    S.checkAddressOfFunctionIsAvailable(FD, /*Complain=*/true,
                                        Args[0]->getLocStart());
    break;
  }

  case FK_ReferenceInitOverloadFailed:
  case FK_UserConversionOverloadFailed:
    switch (FailedOverloadResult) {
    case OR_Ambiguous:
      if (Failure == FK_UserConversionOverloadFailed)
        S.Diag(Kind.getLocation(), diag::err_typecheck_ambiguous_condition)
          << Args[0]->getType() << DestType
          << Args[0]->getSourceRange();
      else
        S.Diag(Kind.getLocation(), diag::err_ref_init_ambiguous)
          << DestType << Args[0]->getType()
          << Args[0]->getSourceRange();

      FailedCandidateSet.NoteCandidates(S, OCD_ViableCandidates, Args);
      break;

    case OR_No_Viable_Function:
      if (!S.RequireCompleteType(Kind.getLocation(),
                                 DestType.getNonReferenceType(),
                          diag::err_typecheck_nonviable_condition_incomplete,
                               Args[0]->getType(), Args[0]->getSourceRange()))
        S.Diag(Kind.getLocation(), diag::err_typecheck_nonviable_condition)
          << (Entity.getKind() == InitializedEntity::EK_Result)
          << Args[0]->getType() << Args[0]->getSourceRange()
          << DestType.getNonReferenceType();

      FailedCandidateSet.NoteCandidates(S, OCD_AllCandidates, Args);
      break;

    case OR_Deleted: {
      S.Diag(Kind.getLocation(), diag::err_typecheck_deleted_function)
        << Args[0]->getType() << DestType.getNonReferenceType()
        << Args[0]->getSourceRange();
      OverloadCandidateSet::iterator Best;
      OverloadingResult Ovl
        = FailedCandidateSet.BestViableFunction(S, Kind.getLocation(), Best,
                                                true);
      if (Ovl == OR_Deleted) {
        S.NoteDeletedFunction(Best->Function);
      } else {
        llvm_unreachable("Inconsistent overload resolution?");
      }
      break;
    }

    case OR_Success:
      llvm_unreachable("Conversion did not fail!");
    }
    break;

  case FK_NonConstLValueReferenceBindingToTemporary:
    if (isa<InitListExpr>(Args[0])) {
      S.Diag(Kind.getLocation(),
             diag::err_lvalue_reference_bind_to_initlist)
      << DestType.getNonReferenceType().isVolatileQualified()
      << DestType.getNonReferenceType()
      << Args[0]->getSourceRange();
      break;
    }
    // Intentional fallthrough

  case FK_NonConstLValueReferenceBindingToUnrelated:
    S.Diag(Kind.getLocation(),
           Failure == FK_NonConstLValueReferenceBindingToTemporary
             ? diag::err_lvalue_reference_bind_to_temporary
             : diag::err_lvalue_reference_bind_to_unrelated)
      << DestType.getNonReferenceType().isVolatileQualified()
      << DestType.getNonReferenceType()
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;

  case FK_RValueReferenceBindingToLValue:
    S.Diag(Kind.getLocation(), diag::err_lvalue_to_rvalue_ref)
      << DestType.getNonReferenceType() << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;

  case FK_ReferenceInitDropsQualifiers: {
    QualType SourceType = Args[0]->getType();
    QualType NonRefType = DestType.getNonReferenceType();
    Qualifiers DroppedQualifiers =
        SourceType.getQualifiers() - NonRefType.getQualifiers();

    S.Diag(Kind.getLocation(), diag::err_reference_bind_drops_quals)
      << SourceType
      << NonRefType
      << DroppedQualifiers.getCVRQualifiers()
      << Args[0]->getSourceRange();
    break;
  }

  case FK_ReferenceInitFailed:
    S.Diag(Kind.getLocation(), diag::err_reference_bind_failed)
      << DestType.getNonReferenceType()
      << Args[0]->isLValue()
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    emitBadConversionNotes(S, Entity, Args[0]);
    break;

  case FK_ConversionFailed: {
    QualType FromType = Args[0]->getType();
    PartialDiagnostic PDiag = S.PDiag(diag::err_init_conversion_failed)
      << (int)Entity.getKind()
      << DestType
      << Args[0]->isLValue()
      << FromType
      << Args[0]->getSourceRange();
    S.HandleFunctionTypeMismatch(PDiag, FromType, DestType);
    S.Diag(Kind.getLocation(), PDiag);
    emitBadConversionNotes(S, Entity, Args[0]);
    break;
  }

  case FK_ConversionFromPropertyFailed:
    // No-op. This error has already been reported.
    break;

  case FK_TooManyInitsForScalar: {
    SourceRange R;

    auto *InitList = dyn_cast<InitListExpr>(Args[0]);
    if (InitList && InitList->getNumInits() >= 1) {
      R = SourceRange(InitList->getInit(0)->getLocEnd(), InitList->getLocEnd());
    } else {
      assert(Args.size() > 1 && "Expected multiple initializers!");
      R = SourceRange(Args.front()->getLocEnd(), Args.back()->getLocEnd());
    }

    R.setBegin(S.getLocForEndOfToken(R.getBegin()));
    if (Kind.isCStyleOrFunctionalCast())
      S.Diag(Kind.getLocation(), diag::err_builtin_func_cast_more_than_one_arg)
        << R;
    else
      S.Diag(Kind.getLocation(), diag::err_excess_initializers)
        << /*scalar=*/2 << R;
    break;
  }

  case FK_ReferenceBindingToInitList:
    S.Diag(Kind.getLocation(), diag::err_reference_bind_init_list)
      << DestType.getNonReferenceType() << Args[0]->getSourceRange();
    break;

  case FK_InitListBadDestinationType:
    S.Diag(Kind.getLocation(), diag::err_init_list_bad_dest_type)
      << (DestType->isRecordType()) << DestType << Args[0]->getSourceRange();
    break;

  case FK_ListConstructorOverloadFailed:
  case FK_ConstructorOverloadFailed: {
    SourceRange ArgsRange;
    if (Args.size())
      ArgsRange = SourceRange(Args.front()->getLocStart(),
                              Args.back()->getLocEnd());

    if (Failure == FK_ListConstructorOverloadFailed) {
      assert(Args.size() == 1 &&
             "List construction from other than 1 argument.");
      InitListExpr *InitList = cast<InitListExpr>(Args[0]);
      Args = MultiExprArg(InitList->getInits(), InitList->getNumInits());
    }

    // FIXME: Using "DestType" for the entity we're printing is probably
    // bad.
    switch (FailedOverloadResult) {
      case OR_Ambiguous:
        S.Diag(Kind.getLocation(), diag::err_ovl_ambiguous_init)
          << DestType << ArgsRange;
        FailedCandidateSet.NoteCandidates(S, OCD_ViableCandidates, Args);
        break;

      case OR_No_Viable_Function:
        if (Kind.getKind() == InitializationKind::IK_Default &&
            (Entity.getKind() == InitializedEntity::EK_Base ||
             Entity.getKind() == InitializedEntity::EK_Member) &&
            isa<CXXConstructorDecl>(S.CurContext)) {
          // This is implicit default initialization of a member or
          // base within a constructor. If no viable function was
          // found, notify the user that she needs to explicitly
          // initialize this base/member.
          CXXConstructorDecl *Constructor
            = cast<CXXConstructorDecl>(S.CurContext);
          if (Entity.getKind() == InitializedEntity::EK_Base) {
            S.Diag(Kind.getLocation(), diag::err_missing_default_ctor)
              << (Constructor->getInheritedConstructor() ? 2 :
                  Constructor->isImplicit() ? 1 : 0)
              << S.Context.getTypeDeclType(Constructor->getParent())
              << /*base=*/0
              << Entity.getType();

            RecordDecl *BaseDecl
              = Entity.getBaseSpecifier()->getType()->getAs<RecordType>()
                                                                  ->getDecl();
            S.Diag(BaseDecl->getLocation(), diag::note_previous_decl)
              << S.Context.getTagDeclType(BaseDecl);
          } else {
            S.Diag(Kind.getLocation(), diag::err_missing_default_ctor)
              << (Constructor->getInheritedConstructor() ? 2 :
                  Constructor->isImplicit() ? 1 : 0)
              << S.Context.getTypeDeclType(Constructor->getParent())
              << /*member=*/1
              << Entity.getName();
            S.Diag(Entity.getDecl()->getLocation(),
                   diag::note_member_declared_at);

            if (const RecordType *Record
                                 = Entity.getType()->getAs<RecordType>())
              S.Diag(Record->getDecl()->getLocation(),
                     diag::note_previous_decl)
                << S.Context.getTagDeclType(Record->getDecl());
          }
          break;
        }

        S.Diag(Kind.getLocation(), diag::err_ovl_no_viable_function_in_init)
          << DestType << ArgsRange;
        FailedCandidateSet.NoteCandidates(S, OCD_AllCandidates, Args);
        break;

      case OR_Deleted: {
        OverloadCandidateSet::iterator Best;
        OverloadingResult Ovl
          = FailedCandidateSet.BestViableFunction(S, Kind.getLocation(), Best);
        if (Ovl != OR_Deleted) {
          S.Diag(Kind.getLocation(), diag::err_ovl_deleted_init)
            << true << DestType << ArgsRange;
          llvm_unreachable("Inconsistent overload resolution?");
          break;
        }
       
        // If this is a defaulted or implicitly-declared function, then
        // it was implicitly deleted. Make it clear that the deletion was
        // implicit.
        if (S.isImplicitlyDeleted(Best->Function))
          S.Diag(Kind.getLocation(), diag::err_ovl_deleted_special_init)
            << S.getSpecialMember(cast<CXXMethodDecl>(Best->Function))
            << DestType << ArgsRange;
        else
          S.Diag(Kind.getLocation(), diag::err_ovl_deleted_init)
            << true << DestType << ArgsRange;

        S.NoteDeletedFunction(Best->Function);
        break;
      }

      case OR_Success:
        llvm_unreachable("Conversion did not fail!");
    }
  }
  break;

  case FK_DefaultInitOfConst:
    if (Entity.getKind() == InitializedEntity::EK_Member &&
        isa<CXXConstructorDecl>(S.CurContext)) {
      // This is implicit default-initialization of a const member in
      // a constructor. Complain that it needs to be explicitly
      // initialized.
      CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(S.CurContext);
      S.Diag(Kind.getLocation(), diag::err_uninitialized_member_in_ctor)
        << (Constructor->getInheritedConstructor() ? 2 :
            Constructor->isImplicit() ? 1 : 0)
        << S.Context.getTypeDeclType(Constructor->getParent())
        << /*const=*/1
        << Entity.getName();
      S.Diag(Entity.getDecl()->getLocation(), diag::note_previous_decl)
        << Entity.getName();
    } else {
      S.Diag(Kind.getLocation(), diag::err_default_init_const)
          << DestType << (bool)DestType->getAs<RecordType>();
    }
    break;

  case FK_Incomplete:
    S.RequireCompleteType(Kind.getLocation(), FailedIncompleteType,
                          diag::err_init_incomplete_type);
    break;

  case FK_ListInitializationFailed: {
    // Run the init list checker again to emit diagnostics.
    InitListExpr *InitList = cast<InitListExpr>(Args[0]);
    diagnoseListInit(S, Entity, InitList);
    break;
  }

  case FK_PlaceholderType: {
    // FIXME: Already diagnosed!
    break;
  }

  case FK_ExplicitConstructor: {
    S.Diag(Kind.getLocation(), diag::err_selected_explicit_constructor)
      << Args[0]->getSourceRange();
    OverloadCandidateSet::iterator Best;
    OverloadingResult Ovl
      = FailedCandidateSet.BestViableFunction(S, Kind.getLocation(), Best);
    (void)Ovl;
    assert(Ovl == OR_Success && "Inconsistent overload resolution");
    CXXConstructorDecl *CtorDecl = cast<CXXConstructorDecl>(Best->Function);
    S.Diag(CtorDecl->getLocation(), diag::note_constructor_declared_here);
    break;
  }
  }

  PrintInitLocationNote(S, Entity);
  return true;
}

void InitializationSequence::dump(raw_ostream &OS) const {
  switch (SequenceKind) {
  case FailedSequence: {
    OS << "Failed sequence: ";
    switch (Failure) {
    case FK_TooManyInitsForReference:
      OS << "too many initializers for reference";
      break;

    case FK_ArrayNeedsInitList:
      OS << "array requires initializer list";
      break;

    case FK_AddressOfUnaddressableFunction:
      OS << "address of unaddressable function was taken";
      break;

    case FK_ArrayNeedsInitListOrStringLiteral:
      OS << "array requires initializer list or string literal";
      break;

    case FK_ArrayNeedsInitListOrWideStringLiteral:
      OS << "array requires initializer list or wide string literal";
      break;

    case FK_NarrowStringIntoWideCharArray:
      OS << "narrow string into wide char array";
      break;

    case FK_WideStringIntoCharArray:
      OS << "wide string into char array";
      break;

    case FK_IncompatWideStringIntoWideChar:
      OS << "incompatible wide string into wide char array";
      break;

    case FK_ArrayTypeMismatch:
      OS << "array type mismatch";
      break;

    case FK_NonConstantArrayInit:
      OS << "non-constant array initializer";
      break;

    case FK_AddressOfOverloadFailed:
      OS << "address of overloaded function failed";
      break;

    case FK_ReferenceInitOverloadFailed:
      OS << "overload resolution for reference initialization failed";
      break;

    case FK_NonConstLValueReferenceBindingToTemporary:
      OS << "non-const lvalue reference bound to temporary";
      break;

    case FK_NonConstLValueReferenceBindingToUnrelated:
      OS << "non-const lvalue reference bound to unrelated type";
      break;

    case FK_RValueReferenceBindingToLValue:
      OS << "rvalue reference bound to an lvalue";
      break;

    case FK_ReferenceInitDropsQualifiers:
      OS << "reference initialization drops qualifiers";
      break;

    case FK_ReferenceInitFailed:
      OS << "reference initialization failed";
      break;

    case FK_ConversionFailed:
      OS << "conversion failed";
      break;

    case FK_ConversionFromPropertyFailed:
      OS << "conversion from property failed";
      break;

    case FK_TooManyInitsForScalar:
      OS << "too many initializers for scalar";
      break;

    case FK_ReferenceBindingToInitList:
      OS << "referencing binding to initializer list";
      break;

    case FK_InitListBadDestinationType:
      OS << "initializer list for non-aggregate, non-scalar type";
      break;

    case FK_UserConversionOverloadFailed:
      OS << "overloading failed for user-defined conversion";
      break;

    case FK_ConstructorOverloadFailed:
      OS << "constructor overloading failed";
      break;

    case FK_DefaultInitOfConst:
      OS << "default initialization of a const variable";
      break;

    case FK_Incomplete:
      OS << "initialization of incomplete type";
      break;

    case FK_ListInitializationFailed:
      OS << "list initialization checker failure";
      break;

    case FK_VariableLengthArrayHasInitializer:
      OS << "variable length array has an initializer";
      break;

    case FK_PlaceholderType:
      OS << "initializer expression isn't contextually valid";
      break;

    case FK_ListConstructorOverloadFailed:
      OS << "list constructor overloading failed";
      break;

    case FK_ExplicitConstructor:
      OS << "list copy initialization chose explicit constructor";
      break;
    }
    OS << '\n';
    return;
  }

  case DependentSequence:
    OS << "Dependent sequence\n";
    return;

  case NormalSequence:
    OS << "Normal sequence: ";
    break;
  }

  for (step_iterator S = step_begin(), SEnd = step_end(); S != SEnd; ++S) {
    if (S != step_begin()) {
      OS << " -> ";
    }

    switch (S->Kind) {
    case SK_ResolveAddressOfOverloadedFunction:
      OS << "resolve address of overloaded function";
      break;

    case SK_CastDerivedToBaseRValue:
      OS << "derived-to-base case (rvalue" << S->Type.getAsString() << ")";
      break;

    case SK_CastDerivedToBaseXValue:
      OS << "derived-to-base case (xvalue" << S->Type.getAsString() << ")";
      break;

    case SK_CastDerivedToBaseLValue:
      OS << "derived-to-base case (lvalue" << S->Type.getAsString() << ")";
      break;

    case SK_BindReference:
      OS << "bind reference to lvalue";
      break;

    case SK_BindReferenceToTemporary:
      OS << "bind reference to a temporary";
      break;

    case SK_ExtraneousCopyToTemporary:
      OS << "extraneous C++03 copy to temporary";
      break;

    case SK_UserConversion:
      OS << "user-defined conversion via " << *S->Function.Function;
      break;

    case SK_QualificationConversionRValue:
      OS << "qualification conversion (rvalue)";
      break;

    case SK_QualificationConversionXValue:
      OS << "qualification conversion (xvalue)";
      break;

    case SK_QualificationConversionLValue:
      OS << "qualification conversion (lvalue)";
      break;

    case SK_AtomicConversion:
      OS << "non-atomic-to-atomic conversion";
      break;

    case SK_LValueToRValue:
      OS << "load (lvalue to rvalue)";
      break;

    case SK_ConversionSequence:
      OS << "implicit conversion sequence (";
      S->ICS->dump(); // FIXME: use OS
      OS << ")";
      break;

    case SK_ConversionSequenceNoNarrowing:
      OS << "implicit conversion sequence with narrowing prohibited (";
      S->ICS->dump(); // FIXME: use OS
      OS << ")";
      break;

    case SK_ListInitialization:
      OS << "list aggregate initialization";
      break;

    case SK_UnwrapInitList:
      OS << "unwrap reference initializer list";
      break;

    case SK_RewrapInitList:
      OS << "rewrap reference initializer list";
      break;

    case SK_ConstructorInitialization:
      OS << "constructor initialization";
      break;

    case SK_ConstructorInitializationFromList:
      OS << "list initialization via constructor";
      break;

    case SK_ZeroInitialization:
      OS << "zero initialization";
      break;

    case SK_CAssignment:
      OS << "C assignment";
      break;

    case SK_StringInit:
      OS << "string initialization";
      break;

    case SK_ObjCObjectConversion:
      OS << "Objective-C object conversion";
      break;

    case SK_ArrayInit:
      OS << "array initialization";
      break;

    case SK_ParenthesizedArrayInit:
      OS << "parenthesized array initialization";
      break;

    case SK_PassByIndirectCopyRestore:
      OS << "pass by indirect copy and restore";
      break;

    case SK_PassByIndirectRestore:
      OS << "pass by indirect restore";
      break;

    case SK_ProduceObjCObject:
      OS << "Objective-C object retension";
      break;

    case SK_StdInitializerList:
      OS << "std::initializer_list from initializer list";
      break;

    case SK_StdInitializerListConstructorCall:
      OS << "list initialization from std::initializer_list";
      break;

    case SK_OCLSamplerInit:
      OS << "OpenCL sampler_t from integer constant";
      break;

    case SK_OCLZeroEvent:
      OS << "OpenCL event_t from zero";
      break;
    }

    OS << " [" << S->Type.getAsString() << ']';
  }

  OS << '\n';
}

void InitializationSequence::dump() const {
  dump(llvm::errs());
}

static void DiagnoseNarrowingInInitList(Sema &S,
                                        const ImplicitConversionSequence &ICS,
                                        QualType PreNarrowingType,
                                        QualType EntityType,
                                        const Expr *PostInit) {
  const StandardConversionSequence *SCS = nullptr;
  switch (ICS.getKind()) {
  case ImplicitConversionSequence::StandardConversion:
    SCS = &ICS.Standard;
    break;
  case ImplicitConversionSequence::UserDefinedConversion:
    SCS = &ICS.UserDefined.After;
    break;
  case ImplicitConversionSequence::AmbiguousConversion:
  case ImplicitConversionSequence::EllipsisConversion:
  case ImplicitConversionSequence::BadConversion:
    return;
  }

  // C++11 [dcl.init.list]p7: Check whether this is a narrowing conversion.
  APValue ConstantValue;
  QualType ConstantType;
  switch (SCS->getNarrowingKind(S.Context, PostInit, ConstantValue,
                                ConstantType)) {
  case NK_Not_Narrowing:
    // No narrowing occurred.
    return;

  case NK_Type_Narrowing:
    // This was a floating-to-integer conversion, which is always considered a
    // narrowing conversion even if the value is a constant and can be
    // represented exactly as an integer.
    S.Diag(PostInit->getLocStart(),
           (S.getLangOpts().MicrosoftExt || !S.getLangOpts().CPlusPlus11)
               ? diag::warn_init_list_type_narrowing
               : diag::ext_init_list_type_narrowing)
      << PostInit->getSourceRange()
      << PreNarrowingType.getLocalUnqualifiedType()
      << EntityType.getLocalUnqualifiedType();
    break;

  case NK_Constant_Narrowing:
    // A constant value was narrowed.
    S.Diag(PostInit->getLocStart(),
           (S.getLangOpts().MicrosoftExt || !S.getLangOpts().CPlusPlus11)
               ? diag::warn_init_list_constant_narrowing
               : diag::ext_init_list_constant_narrowing)
      << PostInit->getSourceRange()
      << ConstantValue.getAsString(S.getASTContext(), ConstantType)
      << EntityType.getLocalUnqualifiedType();
    break;

  case NK_Variable_Narrowing:
    // A variable's value may have been narrowed.
    S.Diag(PostInit->getLocStart(),
           (S.getLangOpts().MicrosoftExt || !S.getLangOpts().CPlusPlus11)
               ? diag::warn_init_list_variable_narrowing
               : diag::ext_init_list_variable_narrowing)
      << PostInit->getSourceRange()
      << PreNarrowingType.getLocalUnqualifiedType()
      << EntityType.getLocalUnqualifiedType();
    break;
  }

  SmallString<128> StaticCast;
  llvm::raw_svector_ostream OS(StaticCast);
  OS << "static_cast<";
  if (const TypedefType *TT = EntityType->getAs<TypedefType>()) {
    // It's important to use the typedef's name if there is one so that the
    // fixit doesn't break code using types like int64_t.
    //
    // FIXME: This will break if the typedef requires qualification.  But
    // getQualifiedNameAsString() includes non-machine-parsable components.
    OS << *TT->getDecl();
  } else if (const BuiltinType *BT = EntityType->getAs<BuiltinType>())
    OS << BT->getName(S.getLangOpts());
  else {
    // Oops, we didn't find the actual type of the variable.  Don't emit a fixit
    // with a broken cast.
    return;
  }
  OS << ">(";
  S.Diag(PostInit->getLocStart(), diag::note_init_list_narrowing_silence)
      << PostInit->getSourceRange()
      << FixItHint::CreateInsertion(PostInit->getLocStart(), OS.str())
      << FixItHint::CreateInsertion(
             S.getLocForEndOfToken(PostInit->getLocEnd()), ")");
}

//===----------------------------------------------------------------------===//
// Initialization helper functions
//===----------------------------------------------------------------------===//
bool
Sema::CanPerformCopyInitialization(const InitializedEntity &Entity,
                                   ExprResult Init) {
  if (Init.isInvalid())
    return false;

  Expr *InitE = Init.get();
  assert(InitE && "No initialization expression");

  InitializationKind Kind
    = InitializationKind::CreateCopy(InitE->getLocStart(), SourceLocation());
  InitializationSequence Seq(*this, Entity, Kind, InitE);
  return !Seq.Failed();
}

ExprResult
Sema::PerformCopyInitialization(const InitializedEntity &Entity,
                                SourceLocation EqualLoc,
                                ExprResult Init,
                                bool TopLevelOfInitList,
                                bool AllowExplicit) {
  if (Init.isInvalid())
    return ExprError();

  Expr *InitE = Init.get();
  assert(InitE && "No initialization expression?");

  if (EqualLoc.isInvalid())
    EqualLoc = InitE->getLocStart();

  InitializationKind Kind = InitializationKind::CreateCopy(InitE->getLocStart(),
                                                           EqualLoc,
                                                           AllowExplicit);
  InitializationSequence Seq(*this, Entity, Kind, InitE, TopLevelOfInitList);

  ExprResult Result = Seq.Perform(*this, Entity, Kind, InitE);

  return Result;
}
