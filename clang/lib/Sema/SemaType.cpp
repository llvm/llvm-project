//===--- SemaType.cpp - Semantic Analysis for Types -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements type-related semantic analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

/// \brief Perform adjustment on the parameter type of a function.
///
/// This routine adjusts the given parameter type @p T to the actual
/// parameter type used by semantic analysis (C99 6.7.5.3p[7,8],
/// C++ [dcl.fct]p3). The adjusted parameter type is returned.
QualType Sema::adjustParameterType(QualType T) {
  // C99 6.7.5.3p7:
  //   A declaration of a parameter as "array of type" shall be
  //   adjusted to "qualified pointer to type", where the type
  //   qualifiers (if any) are those specified within the [ and ] of
  //   the array type derivation.
  if (T->isArrayType())
    return Context.getArrayDecayedType(T);
  
  // C99 6.7.5.3p8:
  //   A declaration of a parameter as "function returning type"
  //   shall be adjusted to "pointer to function returning type", as
  //   in 6.3.2.1.
  if (T->isFunctionType())
    return Context.getPointerType(T);

  return T;
}



/// isOmittedBlockReturnType - Return true if this declarator is missing a
/// return type because this is a omitted return type on a block literal. 
static bool isOmittedBlockReturnType(const Declarator &D) {
  if (D.getContext() != Declarator::BlockLiteralContext ||
      D.getDeclSpec().hasTypeSpecifier())
    return false;
  
  if (D.getNumTypeObjects() == 0)
    return true;   // ^{ ... }
  
  if (D.getNumTypeObjects() == 1 &&
      D.getTypeObject(0).Kind == DeclaratorChunk::Function)
    return true;   // ^(int X, float Y) { ... }
  
  return false;
}

// objc_gc applies to Objective-C pointers or, otherwise, to the
// smallest available pointer type (i.e. 'void*' in 'void**').
#define OBJC_POINTER_TYPE_ATTRS_CASELIST \
    case AttributeList::AT_objc_gc

// Function type attributes.
#define FUNCTION_TYPE_ATTRS_CASELIST \
    case AttributeList::AT_noreturn: \
    case AttributeList::AT_cdecl: \
    case AttributeList::AT_fastcall: \
    case AttributeList::AT_stdcall: \
    case AttributeList::AT_thiscall: \
    case AttributeList::AT_pascal: \
    case AttributeList::AT_regparm

namespace {
  /// An object which stores processing state for the entire
  /// GetTypeForDeclarator process.
  class TypeProcessingState {
    Sema &sema;

    /// The declarator being processed.
    Declarator &declarator;

    /// The index of the declarator chunk we're currently processing.
    /// May be the total number of valid chunks, indicating the
    /// DeclSpec.
    unsigned chunkIndex;

    /// Whether there are non-trivial modifications to the decl spec.
    bool trivial;

    /// The original set of attributes on the DeclSpec.
    llvm::SmallVector<AttributeList*, 2> savedAttrs;

    /// A list of attributes to diagnose the uselessness of when the
    /// processing is complete.
    llvm::SmallVector<AttributeList*, 2> ignoredTypeAttrs;

  public:
    TypeProcessingState(Sema &sema, Declarator &declarator)
      : sema(sema), declarator(declarator),
        chunkIndex(declarator.getNumTypeObjects()),
        trivial(true) {}

    Sema &getSema() const {
      return sema;
    }

    Declarator &getDeclarator() const {
      return declarator;
    }

    unsigned getCurrentChunkIndex() const {
      return chunkIndex;
    }

    void setCurrentChunkIndex(unsigned idx) {
      assert(idx <= declarator.getNumTypeObjects());
      chunkIndex = idx;
    }

    AttributeList *&getCurrentAttrListRef() const {
      assert(chunkIndex <= declarator.getNumTypeObjects());
      if (chunkIndex == declarator.getNumTypeObjects())
        return getMutableDeclSpec().getAttributes().getListRef();
      return declarator.getTypeObject(chunkIndex).getAttrListRef();
    }

    /// Save the current set of attributes on the DeclSpec.
    void saveDeclSpecAttrs() {
      // Don't try to save them multiple times.
      if (!savedAttrs.empty()) return;

      DeclSpec &spec = getMutableDeclSpec();
      for (AttributeList *attr = spec.getAttributes().getList(); attr;
             attr = attr->getNext())
        savedAttrs.push_back(attr);
      trivial &= savedAttrs.empty();
    }

    /// Record that we had nowhere to put the given type attribute.
    /// We will diagnose such attributes later.
    void addIgnoredTypeAttr(AttributeList &attr) {
      ignoredTypeAttrs.push_back(&attr);
    }

    /// Diagnose all the ignored type attributes, given that the
    /// declarator worked out to the given type.
    void diagnoseIgnoredTypeAttrs(QualType type) const {
      for (llvm::SmallVectorImpl<AttributeList*>::const_iterator
             i = ignoredTypeAttrs.begin(), e = ignoredTypeAttrs.end();
           i != e; ++i) {
        AttributeList &attr = **i;
        getSema().Diag(attr.getLoc(), diag::warn_function_attribute_wrong_type)
          << attr.getName() << type;
      }
    }

    ~TypeProcessingState() {
      if (trivial) return;

      restoreDeclSpecAttrs();
    }

  private:
    DeclSpec &getMutableDeclSpec() const {
      return const_cast<DeclSpec&>(declarator.getDeclSpec());
    }

    void restoreDeclSpecAttrs() {
      assert(!savedAttrs.empty());
      getMutableDeclSpec().getAttributes().set(savedAttrs[0]);
      for (unsigned i = 0, e = savedAttrs.size() - 1; i != e; ++i)
        savedAttrs[i]->setNext(savedAttrs[i+1]);
      savedAttrs.back()->setNext(0);
    }
  };

  /// Basically std::pair except that we really want to avoid an
  /// implicit operator= for safety concerns.  It's also a minor
  /// link-time optimization for this to be a private type.
  struct AttrAndList {
    /// The attribute.
    AttributeList &first;

    /// The head of the list the attribute is currently in.
    AttributeList *&second;

    AttrAndList(AttributeList &attr, AttributeList *&head)
      : first(attr), second(head) {}
  };
}

namespace llvm {
  template <> struct isPodLike<AttrAndList> {
    static const bool value = true;
  };
}

static void spliceAttrIntoList(AttributeList &attr, AttributeList *&head) {
  attr.setNext(head);
  head = &attr;
}

static void spliceAttrOutOfList(AttributeList &attr, AttributeList *&head) {
  if (head == &attr) {
    head = attr.getNext();
    return;
  }

  AttributeList *cur = head;
  while (true) {
    assert(cur && cur->getNext() && "ran out of attrs?");
    if (cur->getNext() == &attr) {
      cur->setNext(attr.getNext());
      return;
    }
    cur = cur->getNext();
  }
}

static void moveAttrFromListToList(AttributeList &attr,
                                   AttributeList *&fromList,
                                   AttributeList *&toList) {
  spliceAttrOutOfList(attr, fromList);
  spliceAttrIntoList(attr, toList);
}

static void processTypeAttrs(TypeProcessingState &state,
                             QualType &type, bool isDeclSpec,
                             AttributeList *attrs);

static bool handleFunctionTypeAttr(TypeProcessingState &state,
                                   AttributeList &attr,
                                   QualType &type);

static bool handleObjCGCTypeAttr(TypeProcessingState &state,
                                 AttributeList &attr, QualType &type);

static bool handleObjCPointerTypeAttr(TypeProcessingState &state,
                                      AttributeList &attr, QualType &type) {
  // Right now, we have exactly one of these attributes: objc_gc.
  assert(attr.getKind() == AttributeList::AT_objc_gc);
  return handleObjCGCTypeAttr(state, attr, type);
}

/// Given that an objc_gc attribute was written somewhere on a
/// declaration *other* than on the declarator itself (for which, use
/// distributeObjCPointerTypeAttrFromDeclarator), and given that it
/// didn't apply in whatever position it was written in, try to move
/// it to a more appropriate position.
static void distributeObjCPointerTypeAttr(TypeProcessingState &state,
                                          AttributeList &attr,
                                          QualType type) {
  Declarator &declarator = state.getDeclarator();
  for (unsigned i = state.getCurrentChunkIndex(); i != 0; --i) {
    DeclaratorChunk &chunk = declarator.getTypeObject(i-1);
    switch (chunk.Kind) {
    case DeclaratorChunk::Pointer:
    case DeclaratorChunk::BlockPointer:
      moveAttrFromListToList(attr, state.getCurrentAttrListRef(),
                             chunk.getAttrListRef());
      return;

    case DeclaratorChunk::Paren:
    case DeclaratorChunk::Array:
      continue;

    // Don't walk through these.
    case DeclaratorChunk::Reference:
    case DeclaratorChunk::Function:
    case DeclaratorChunk::MemberPointer:
      goto error;
    }
  }
 error:
  
  state.getSema().Diag(attr.getLoc(), diag::warn_function_attribute_wrong_type)
    << attr.getName() << type;
}

/// Distribute an objc_gc type attribute that was written on the
/// declarator.
static void
distributeObjCPointerTypeAttrFromDeclarator(TypeProcessingState &state,
                                            AttributeList &attr,
                                            QualType &declSpecType) {
  Declarator &declarator = state.getDeclarator();

  // objc_gc goes on the innermost pointer to something that's not a
  // pointer.
  unsigned innermost = -1U;
  bool considerDeclSpec = true;
  for (unsigned i = 0, e = declarator.getNumTypeObjects(); i != e; ++i) {
    DeclaratorChunk &chunk = declarator.getTypeObject(i);
    switch (chunk.Kind) {
    case DeclaratorChunk::Pointer:
    case DeclaratorChunk::BlockPointer:
      innermost = i;
      continue;

    case DeclaratorChunk::Reference:
    case DeclaratorChunk::MemberPointer:
    case DeclaratorChunk::Paren:
    case DeclaratorChunk::Array:
      continue;

    case DeclaratorChunk::Function:
      considerDeclSpec = false;
      goto done;
    }
  }
 done:

  // That might actually be the decl spec if we weren't blocked by
  // anything in the declarator.
  if (considerDeclSpec) {
    if (handleObjCPointerTypeAttr(state, attr, declSpecType))
      return;
  }

  // Otherwise, if we found an appropriate chunk, splice the attribute
  // into it.
  if (innermost != -1U) {
    moveAttrFromListToList(attr, declarator.getAttrListRef(),
                       declarator.getTypeObject(innermost).getAttrListRef());
    return;
  }

  // Otherwise, diagnose when we're done building the type.
  spliceAttrOutOfList(attr, declarator.getAttrListRef());
  state.addIgnoredTypeAttr(attr);
}

/// A function type attribute was written somewhere in a declaration
/// *other* than on the declarator itself or in the decl spec.  Given
/// that it didn't apply in whatever position it was written in, try
/// to move it to a more appropriate position.
static void distributeFunctionTypeAttr(TypeProcessingState &state,
                                       AttributeList &attr,
                                       QualType type) {
  Declarator &declarator = state.getDeclarator();

  // Try to push the attribute from the return type of a function to
  // the function itself.
  for (unsigned i = state.getCurrentChunkIndex(); i != 0; --i) {
    DeclaratorChunk &chunk = declarator.getTypeObject(i-1);
    switch (chunk.Kind) {
    case DeclaratorChunk::Function:
      moveAttrFromListToList(attr, state.getCurrentAttrListRef(),
                             chunk.getAttrListRef());
      return;

    case DeclaratorChunk::Paren:
    case DeclaratorChunk::Pointer:
    case DeclaratorChunk::BlockPointer:
    case DeclaratorChunk::Array:
    case DeclaratorChunk::Reference:
    case DeclaratorChunk::MemberPointer:
      continue;
    }
  }
  
  state.getSema().Diag(attr.getLoc(), diag::warn_function_attribute_wrong_type)
    << attr.getName() << type;
}

/// Try to distribute a function type attribute to the innermost
/// function chunk or type.  Returns true if the attribute was
/// distributed, false if no location was found.
static bool
distributeFunctionTypeAttrToInnermost(TypeProcessingState &state,
                                      AttributeList &attr,
                                      AttributeList *&attrList,
                                      QualType &declSpecType) {
  Declarator &declarator = state.getDeclarator();

  // Put it on the innermost function chunk, if there is one.
  for (unsigned i = 0, e = declarator.getNumTypeObjects(); i != e; ++i) {
    DeclaratorChunk &chunk = declarator.getTypeObject(i);
    if (chunk.Kind != DeclaratorChunk::Function) continue;

    moveAttrFromListToList(attr, attrList, chunk.getAttrListRef());
    return true;
  }

  return handleFunctionTypeAttr(state, attr, declSpecType);
}

/// A function type attribute was written in the decl spec.  Try to
/// apply it somewhere.
static void
distributeFunctionTypeAttrFromDeclSpec(TypeProcessingState &state,
                                       AttributeList &attr,
                                       QualType &declSpecType) {
  state.saveDeclSpecAttrs();

  // Try to distribute to the innermost.
  if (distributeFunctionTypeAttrToInnermost(state, attr,
                                            state.getCurrentAttrListRef(),
                                            declSpecType))
    return;

  // If that failed, diagnose the bad attribute when the declarator is
  // fully built.
  state.addIgnoredTypeAttr(attr);
}

/// A function type attribute was written on the declarator.  Try to
/// apply it somewhere.
static void
distributeFunctionTypeAttrFromDeclarator(TypeProcessingState &state,
                                         AttributeList &attr,
                                         QualType &declSpecType) {
  Declarator &declarator = state.getDeclarator();

  // Try to distribute to the innermost.
  if (distributeFunctionTypeAttrToInnermost(state, attr,
                                            declarator.getAttrListRef(),
                                            declSpecType))
    return;

  // If that failed, diagnose the bad attribute when the declarator is
  // fully built.
  spliceAttrOutOfList(attr, declarator.getAttrListRef());
  state.addIgnoredTypeAttr(attr);
}

/// \brief Given that there are attributes written on the declarator
/// itself, try to distribute any type attributes to the appropriate
/// declarator chunk.
///
/// These are attributes like the following:
///   int f ATTR;
///   int (f ATTR)();
/// but not necessarily this:
///   int f() ATTR;
static void distributeTypeAttrsFromDeclarator(TypeProcessingState &state,
                                              QualType &declSpecType) {
  // Collect all the type attributes from the declarator itself.
  assert(state.getDeclarator().getAttributes() && "declarator has no attrs!");
  AttributeList *attr = state.getDeclarator().getAttributes();
  AttributeList *next;
  do {
    next = attr->getNext();

    switch (attr->getKind()) {
    OBJC_POINTER_TYPE_ATTRS_CASELIST:
      distributeObjCPointerTypeAttrFromDeclarator(state, *attr, declSpecType);
      break;

    FUNCTION_TYPE_ATTRS_CASELIST:
      distributeFunctionTypeAttrFromDeclarator(state, *attr, declSpecType);
      break;

    default:
      break;
    }
  } while ((attr = next));
}

/// Add a synthetic '()' to a block-literal declarator if it is
/// required, given the return type.
static void maybeSynthesizeBlockSignature(TypeProcessingState &state,
                                          QualType declSpecType) {
  Declarator &declarator = state.getDeclarator();

  // First, check whether the declarator would produce a function,
  // i.e. whether the innermost semantic chunk is a function.
  if (declarator.isFunctionDeclarator()) {
    // If so, make that declarator a prototyped declarator.
    declarator.getFunctionTypeInfo().hasPrototype = true;
    return;
  }

  // If there are any type objects, the type as written won't name a
  // function, regardless of the decl spec type.  This is because a
  // block signature declarator is always an abstract-declarator, and
  // abstract-declarators can't just be parentheses chunks.  Therefore
  // we need to build a function chunk unless there are no type
  // objects and the decl spec type is a function.
  if (!declarator.getNumTypeObjects() && declSpecType->isFunctionType())
    return;

  // Note that there *are* cases with invalid declarators where
  // declarators consist solely of parentheses.  In general, these
  // occur only in failed efforts to make function declarators, so
  // faking up the function chunk is still the right thing to do.

  // Otherwise, we need to fake up a function declarator.
  SourceLocation loc = declarator.getSourceRange().getBegin();

  // ...and *prepend* it to the declarator.
  declarator.AddInnermostTypeInfo(DeclaratorChunk::getFunction(
                             ParsedAttributes(),
                             /*proto*/ true,
                             /*variadic*/ false, SourceLocation(),
                             /*args*/ 0, 0,
                             /*type quals*/ 0,
                             /*ref-qualifier*/true, SourceLocation(),
                             /*EH*/ false, SourceLocation(), false, 0, 0, 0,
                             /*parens*/ loc, loc,
                             declarator));

  // For consistency, make sure the state still has us as processing
  // the decl spec.
  assert(state.getCurrentChunkIndex() == declarator.getNumTypeObjects() - 1);
  state.setCurrentChunkIndex(declarator.getNumTypeObjects());
}

/// \brief Convert the specified declspec to the appropriate type
/// object.
/// \param D  the declarator containing the declaration specifier.
/// \returns The type described by the declaration specifiers.  This function
/// never returns null.
static QualType ConvertDeclSpecToType(Sema &S, TypeProcessingState &state) {
  // FIXME: Should move the logic from DeclSpec::Finish to here for validity
  // checking.

  Declarator &declarator = state.getDeclarator();
  const DeclSpec &DS = declarator.getDeclSpec();
  SourceLocation DeclLoc = declarator.getIdentifierLoc();
  if (DeclLoc.isInvalid())
    DeclLoc = DS.getSourceRange().getBegin();
  
  ASTContext &Context = S.Context;

  QualType Result;
  switch (DS.getTypeSpecType()) {
  case DeclSpec::TST_void:
    Result = Context.VoidTy;
    break;
  case DeclSpec::TST_char:
    if (DS.getTypeSpecSign() == DeclSpec::TSS_unspecified)
      Result = Context.CharTy;
    else if (DS.getTypeSpecSign() == DeclSpec::TSS_signed)
      Result = Context.SignedCharTy;
    else {
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unsigned &&
             "Unknown TSS value");
      Result = Context.UnsignedCharTy;
    }
    break;
  case DeclSpec::TST_wchar:
    if (DS.getTypeSpecSign() == DeclSpec::TSS_unspecified)
      Result = Context.WCharTy;
    else if (DS.getTypeSpecSign() == DeclSpec::TSS_signed) {
      S.Diag(DS.getTypeSpecSignLoc(), diag::ext_invalid_sign_spec)
        << DS.getSpecifierName(DS.getTypeSpecType());
      Result = Context.getSignedWCharType();
    } else {
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unsigned &&
        "Unknown TSS value");
      S.Diag(DS.getTypeSpecSignLoc(), diag::ext_invalid_sign_spec)
        << DS.getSpecifierName(DS.getTypeSpecType());
      Result = Context.getUnsignedWCharType();
    }
    break;
  case DeclSpec::TST_char16:
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unspecified &&
        "Unknown TSS value");
      Result = Context.Char16Ty;
    break;
  case DeclSpec::TST_char32:
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unspecified &&
        "Unknown TSS value");
      Result = Context.Char32Ty;
    break;
  case DeclSpec::TST_unspecified:
    // "<proto1,proto2>" is an objc qualified ID with a missing id.
    if (DeclSpec::ProtocolQualifierListTy PQ = DS.getProtocolQualifiers()) {
      Result = Context.getObjCObjectType(Context.ObjCBuiltinIdTy,
                                         (ObjCProtocolDecl**)PQ,
                                         DS.getNumProtocolQualifiers());
      Result = Context.getObjCObjectPointerType(Result);
      break;
    }
    
    // If this is a missing declspec in a block literal return context, then it
    // is inferred from the return statements inside the block.
    if (isOmittedBlockReturnType(declarator)) {
      Result = Context.DependentTy;
      break;
    }

    // Unspecified typespec defaults to int in C90.  However, the C90 grammar
    // [C90 6.5] only allows a decl-spec if there was *some* type-specifier,
    // type-qualifier, or storage-class-specifier.  If not, emit an extwarn.
    // Note that the one exception to this is function definitions, which are
    // allowed to be completely missing a declspec.  This is handled in the
    // parser already though by it pretending to have seen an 'int' in this
    // case.
    if (S.getLangOptions().ImplicitInt) {
      // In C89 mode, we only warn if there is a completely missing declspec
      // when one is not allowed.
      if (DS.isEmpty()) {
        S.Diag(DeclLoc, diag::ext_missing_declspec)
          << DS.getSourceRange()
        << FixItHint::CreateInsertion(DS.getSourceRange().getBegin(), "int");
      }
    } else if (!DS.hasTypeSpecifier()) {
      // C99 and C++ require a type specifier.  For example, C99 6.7.2p2 says:
      // "At least one type specifier shall be given in the declaration
      // specifiers in each declaration, and in the specifier-qualifier list in
      // each struct declaration and type name."
      // FIXME: Does Microsoft really have the implicit int extension in C++?
      if (S.getLangOptions().CPlusPlus &&
          !S.getLangOptions().Microsoft) {
        S.Diag(DeclLoc, diag::err_missing_type_specifier)
          << DS.getSourceRange();

        // When this occurs in C++ code, often something is very broken with the
        // value being declared, poison it as invalid so we don't get chains of
        // errors.
        declarator.setInvalidType(true);
      } else {
        S.Diag(DeclLoc, diag::ext_missing_type_specifier)
          << DS.getSourceRange();
      }
    }

    // FALL THROUGH.
  case DeclSpec::TST_int: {
    if (DS.getTypeSpecSign() != DeclSpec::TSS_unsigned) {
      switch (DS.getTypeSpecWidth()) {
      case DeclSpec::TSW_unspecified: Result = Context.IntTy; break;
      case DeclSpec::TSW_short:       Result = Context.ShortTy; break;
      case DeclSpec::TSW_long:        Result = Context.LongTy; break;
      case DeclSpec::TSW_longlong:
        Result = Context.LongLongTy;
          
        // long long is a C99 feature.
        if (!S.getLangOptions().C99 &&
            !S.getLangOptions().CPlusPlus0x)
          S.Diag(DS.getTypeSpecWidthLoc(), diag::ext_longlong);
        break;
      }
    } else {
      switch (DS.getTypeSpecWidth()) {
      case DeclSpec::TSW_unspecified: Result = Context.UnsignedIntTy; break;
      case DeclSpec::TSW_short:       Result = Context.UnsignedShortTy; break;
      case DeclSpec::TSW_long:        Result = Context.UnsignedLongTy; break;
      case DeclSpec::TSW_longlong:
        Result = Context.UnsignedLongLongTy;
          
        // long long is a C99 feature.
        if (!S.getLangOptions().C99 &&
            !S.getLangOptions().CPlusPlus0x)
          S.Diag(DS.getTypeSpecWidthLoc(), diag::ext_longlong);
        break;
      }
    }
    break;
  }
  case DeclSpec::TST_float: Result = Context.FloatTy; break;
  case DeclSpec::TST_double:
    if (DS.getTypeSpecWidth() == DeclSpec::TSW_long)
      Result = Context.LongDoubleTy;
    else
      Result = Context.DoubleTy;

    if (S.getLangOptions().OpenCL && !S.getOpenCLOptions().cl_khr_fp64) {
      S.Diag(DS.getTypeSpecTypeLoc(), diag::err_double_requires_fp64);
      declarator.setInvalidType(true);
    }
    break;
  case DeclSpec::TST_bool: Result = Context.BoolTy; break; // _Bool or bool
  case DeclSpec::TST_decimal32:    // _Decimal32
  case DeclSpec::TST_decimal64:    // _Decimal64
  case DeclSpec::TST_decimal128:   // _Decimal128
    S.Diag(DS.getTypeSpecTypeLoc(), diag::err_decimal_unsupported);
    Result = Context.IntTy;
    declarator.setInvalidType(true);
    break;
  case DeclSpec::TST_class:
  case DeclSpec::TST_enum:
  case DeclSpec::TST_union:
  case DeclSpec::TST_struct: {
    TypeDecl *D = dyn_cast_or_null<TypeDecl>(DS.getRepAsDecl());
    if (!D) {
      // This can happen in C++ with ambiguous lookups.
      Result = Context.IntTy;
      declarator.setInvalidType(true);
      break;
    }

    // If the type is deprecated or unavailable, diagnose it.
    S.DiagnoseUseOfDecl(D, DS.getTypeSpecTypeLoc());
    
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 && "No qualifiers on tag names!");
    
    // TypeQuals handled by caller.
    Result = Context.getTypeDeclType(D);

    // In C++, make an ElaboratedType.
    if (S.getLangOptions().CPlusPlus) {
      ElaboratedTypeKeyword Keyword
        = ElaboratedType::getKeywordForTypeSpec(DS.getTypeSpecType());
      Result = S.getElaboratedType(Keyword, DS.getTypeSpecScope(), Result);
    }
    if (D->isInvalidDecl())
      declarator.setInvalidType(true);
    break;
  }
  case DeclSpec::TST_typename: {
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 &&
           "Can't handle qualifiers on typedef names yet!");
    Result = S.GetTypeFromParser(DS.getRepAsType());
    if (Result.isNull())
      declarator.setInvalidType(true);
    else if (DeclSpec::ProtocolQualifierListTy PQ
               = DS.getProtocolQualifiers()) {
      if (const ObjCObjectType *ObjT = Result->getAs<ObjCObjectType>()) {
        // Silently drop any existing protocol qualifiers.
        // TODO: determine whether that's the right thing to do.
        if (ObjT->getNumProtocols())
          Result = ObjT->getBaseType();

        if (DS.getNumProtocolQualifiers())
          Result = Context.getObjCObjectType(Result,
                                             (ObjCProtocolDecl**) PQ,
                                             DS.getNumProtocolQualifiers());
      } else if (Result->isObjCIdType()) {
        // id<protocol-list>
        Result = Context.getObjCObjectType(Context.ObjCBuiltinIdTy,
                                           (ObjCProtocolDecl**) PQ,
                                           DS.getNumProtocolQualifiers());
        Result = Context.getObjCObjectPointerType(Result);
      } else if (Result->isObjCClassType()) {
        // Class<protocol-list>
        Result = Context.getObjCObjectType(Context.ObjCBuiltinClassTy,
                                           (ObjCProtocolDecl**) PQ,
                                           DS.getNumProtocolQualifiers());
        Result = Context.getObjCObjectPointerType(Result);
      } else {
        S.Diag(DeclLoc, diag::err_invalid_protocol_qualifiers)
          << DS.getSourceRange();
        declarator.setInvalidType(true);
      }
    }

    // TypeQuals handled by caller.
    break;
  }
  case DeclSpec::TST_typeofType:
    // FIXME: Preserve type source info.
    Result = S.GetTypeFromParser(DS.getRepAsType());
    assert(!Result.isNull() && "Didn't get a type for typeof?");
    if (!Result->isDependentType())
      if (const TagType *TT = Result->getAs<TagType>())
        S.DiagnoseUseOfDecl(TT->getDecl(), DS.getTypeSpecTypeLoc());
    // TypeQuals handled by caller.
    Result = Context.getTypeOfType(Result);
    break;
  case DeclSpec::TST_typeofExpr: {
    Expr *E = DS.getRepAsExpr();
    assert(E && "Didn't get an expression for typeof?");
    // TypeQuals handled by caller.
    Result = S.BuildTypeofExprType(E, DS.getTypeSpecTypeLoc());
    if (Result.isNull()) {
      Result = Context.IntTy;
      declarator.setInvalidType(true);
    }
    break;
  }
  case DeclSpec::TST_decltype: {
    Expr *E = DS.getRepAsExpr();
    assert(E && "Didn't get an expression for decltype?");
    // TypeQuals handled by caller.
    Result = S.BuildDecltypeType(E, DS.getTypeSpecTypeLoc());
    if (Result.isNull()) {
      Result = Context.IntTy;
      declarator.setInvalidType(true);
    }
    break;
  }
  case DeclSpec::TST_auto: {
    // TypeQuals handled by caller.
    Result = Context.getAutoType(QualType());
    break;
  }

  case DeclSpec::TST_error:
    Result = Context.IntTy;
    declarator.setInvalidType(true);
    break;
  }

  // Handle complex types.
  if (DS.getTypeSpecComplex() == DeclSpec::TSC_complex) {
    if (S.getLangOptions().Freestanding)
      S.Diag(DS.getTypeSpecComplexLoc(), diag::ext_freestanding_complex);
    Result = Context.getComplexType(Result);
  } else if (DS.isTypeAltiVecVector()) {
    unsigned typeSize = static_cast<unsigned>(Context.getTypeSize(Result));
    assert(typeSize > 0 && "type size for vector must be greater than 0 bits");
    VectorType::VectorKind VecKind = VectorType::AltiVecVector;
    if (DS.isTypeAltiVecPixel())
      VecKind = VectorType::AltiVecPixel;
    else if (DS.isTypeAltiVecBool())
      VecKind = VectorType::AltiVecBool;
    Result = Context.getVectorType(Result, 128/typeSize, VecKind);
  }

  // FIXME: Imaginary.
  if (DS.getTypeSpecComplex() == DeclSpec::TSC_imaginary)
    S.Diag(DS.getTypeSpecComplexLoc(), diag::err_imaginary_not_supported);

  // Before we process any type attributes, synthesize a block literal
  // function declarator if necessary.
  if (declarator.getContext() == Declarator::BlockLiteralContext)
    maybeSynthesizeBlockSignature(state, Result);

  // Apply any type attributes from the decl spec.  This may cause the
  // list of type attributes to be temporarily saved while the type
  // attributes are pushed around.
  if (AttributeList *attrs = DS.getAttributes().getList())
    processTypeAttrs(state, Result, true, attrs);

  // Apply const/volatile/restrict qualifiers to T.
  if (unsigned TypeQuals = DS.getTypeQualifiers()) {

    // Enforce C99 6.7.3p2: "Types other than pointer types derived from object
    // or incomplete types shall not be restrict-qualified."  C++ also allows
    // restrict-qualified references.
    if (TypeQuals & DeclSpec::TQ_restrict) {
      if (Result->isAnyPointerType() || Result->isReferenceType()) {
        QualType EltTy;
        if (Result->isObjCObjectPointerType())
          EltTy = Result;
        else
          EltTy = Result->isPointerType() ?
                    Result->getAs<PointerType>()->getPointeeType() :
                    Result->getAs<ReferenceType>()->getPointeeType();

        // If we have a pointer or reference, the pointee must have an object
        // incomplete type.
        if (!EltTy->isIncompleteOrObjectType()) {
          S.Diag(DS.getRestrictSpecLoc(),
               diag::err_typecheck_invalid_restrict_invalid_pointee)
            << EltTy << DS.getSourceRange();
          TypeQuals &= ~DeclSpec::TQ_restrict; // Remove the restrict qualifier.
        }
      } else {
        S.Diag(DS.getRestrictSpecLoc(),
               diag::err_typecheck_invalid_restrict_not_pointer)
          << Result << DS.getSourceRange();
        TypeQuals &= ~DeclSpec::TQ_restrict; // Remove the restrict qualifier.
      }
    }

    // Warn about CV qualifiers on functions: C99 6.7.3p8: "If the specification
    // of a function type includes any type qualifiers, the behavior is
    // undefined."
    if (Result->isFunctionType() && TypeQuals) {
      // Get some location to point at, either the C or V location.
      SourceLocation Loc;
      if (TypeQuals & DeclSpec::TQ_const)
        Loc = DS.getConstSpecLoc();
      else if (TypeQuals & DeclSpec::TQ_volatile)
        Loc = DS.getVolatileSpecLoc();
      else {
        assert((TypeQuals & DeclSpec::TQ_restrict) &&
               "Has CVR quals but not C, V, or R?");
        Loc = DS.getRestrictSpecLoc();
      }
      S.Diag(Loc, diag::warn_typecheck_function_qualifiers)
        << Result << DS.getSourceRange();
    }

    // C++ [dcl.ref]p1:
    //   Cv-qualified references are ill-formed except when the
    //   cv-qualifiers are introduced through the use of a typedef
    //   (7.1.3) or of a template type argument (14.3), in which
    //   case the cv-qualifiers are ignored.
    // FIXME: Shouldn't we be checking SCS_typedef here?
    if (DS.getTypeSpecType() == DeclSpec::TST_typename &&
        TypeQuals && Result->isReferenceType()) {
      TypeQuals &= ~DeclSpec::TQ_const;
      TypeQuals &= ~DeclSpec::TQ_volatile;
    }

    Qualifiers Quals = Qualifiers::fromCVRMask(TypeQuals);
    Result = Context.getQualifiedType(Result, Quals);
  }

  return Result;
}

static std::string getPrintableNameForEntity(DeclarationName Entity) {
  if (Entity)
    return Entity.getAsString();

  return "type name";
}

QualType Sema::BuildQualifiedType(QualType T, SourceLocation Loc,
                                  Qualifiers Qs) {
  // Enforce C99 6.7.3p2: "Types other than pointer types derived from
  // object or incomplete types shall not be restrict-qualified."
  if (Qs.hasRestrict()) {
    unsigned DiagID = 0;
    QualType ProblemTy;

    const Type *Ty = T->getCanonicalTypeInternal().getTypePtr();
    if (const ReferenceType *RTy = dyn_cast<ReferenceType>(Ty)) {
      if (!RTy->getPointeeType()->isIncompleteOrObjectType()) {
        DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
        ProblemTy = T->getAs<ReferenceType>()->getPointeeType();
      }
    } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
      if (!PTy->getPointeeType()->isIncompleteOrObjectType()) {
        DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
        ProblemTy = T->getAs<PointerType>()->getPointeeType();
      }
    } else if (const MemberPointerType *PTy = dyn_cast<MemberPointerType>(Ty)) {
      if (!PTy->getPointeeType()->isIncompleteOrObjectType()) {
        DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
        ProblemTy = T->getAs<PointerType>()->getPointeeType();
      }      
    } else if (!Ty->isDependentType()) {
      // FIXME: this deserves a proper diagnostic
      DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
      ProblemTy = T;
    }

    if (DiagID) {
      Diag(Loc, DiagID) << ProblemTy;
      Qs.removeRestrict();
    }
  }

  return Context.getQualifiedType(T, Qs);
}

/// \brief Build a paren type including \p T.
QualType Sema::BuildParenType(QualType T) {
  return Context.getParenType(T);
}

/// \brief Build a pointer type.
///
/// \param T The type to which we'll be building a pointer.
///
/// \param Loc The location of the entity whose type involves this
/// pointer type or, if there is no such entity, the location of the
/// type that will have pointer type.
///
/// \param Entity The name of the entity that involves the pointer
/// type, if known.
///
/// \returns A suitable pointer type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildPointerType(QualType T,
                                SourceLocation Loc, DeclarationName Entity) {
  if (T->isReferenceType()) {
    // C++ 8.3.2p4: There shall be no ... pointers to references ...
    Diag(Loc, diag::err_illegal_decl_pointer_to_reference)
      << getPrintableNameForEntity(Entity) << T;
    return QualType();
  }

  assert(!T->isObjCObjectType() && "Should build ObjCObjectPointerType");

  // Build the pointer type.
  return Context.getPointerType(T);
}

/// \brief Build a reference type.
///
/// \param T The type to which we'll be building a reference.
///
/// \param Loc The location of the entity whose type involves this
/// reference type or, if there is no such entity, the location of the
/// type that will have reference type.
///
/// \param Entity The name of the entity that involves the reference
/// type, if known.
///
/// \returns A suitable reference type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildReferenceType(QualType T, bool SpelledAsLValue,
                                  SourceLocation Loc,
                                  DeclarationName Entity) {
  // C++0x [dcl.ref]p6:
  //   If a typedef (7.1.3), a type template-parameter (14.3.1), or a 
  //   decltype-specifier (7.1.6.2) denotes a type TR that is a reference to a 
  //   type T, an attempt to create the type "lvalue reference to cv TR" creates 
  //   the type "lvalue reference to T", while an attempt to create the type 
  //   "rvalue reference to cv TR" creates the type TR.
  bool LValueRef = SpelledAsLValue || T->getAs<LValueReferenceType>();

  // C++ [dcl.ref]p4: There shall be no references to references.
  //
  // According to C++ DR 106, references to references are only
  // diagnosed when they are written directly (e.g., "int & &"),
  // but not when they happen via a typedef:
  //
  //   typedef int& intref;
  //   typedef intref& intref2;
  //
  // Parser::ParseDeclaratorInternal diagnoses the case where
  // references are written directly; here, we handle the
  // collapsing of references-to-references as described in C++0x.
  // DR 106 and 540 introduce reference-collapsing into C++98/03.

  // C++ [dcl.ref]p1:
  //   A declarator that specifies the type "reference to cv void"
  //   is ill-formed.
  if (T->isVoidType()) {
    Diag(Loc, diag::err_reference_to_void);
    return QualType();
  }

  // Handle restrict on references.
  if (LValueRef)
    return Context.getLValueReferenceType(T, SpelledAsLValue);
  return Context.getRValueReferenceType(T);
}

/// \brief Build an array type.
///
/// \param T The type of each element in the array.
///
/// \param ASM C99 array size modifier (e.g., '*', 'static').
///
/// \param ArraySize Expression describing the size of the array.
///
/// \param Loc The location of the entity whose type involves this
/// array type or, if there is no such entity, the location of the
/// type that will have array type.
///
/// \param Entity The name of the entity that involves the array
/// type, if known.
///
/// \returns A suitable array type, if there are no errors. Otherwise,
/// returns a NULL type.
QualType Sema::BuildArrayType(QualType T, ArrayType::ArraySizeModifier ASM,
                              Expr *ArraySize, unsigned Quals,
                              SourceRange Brackets, DeclarationName Entity) {

  SourceLocation Loc = Brackets.getBegin();
  if (getLangOptions().CPlusPlus) {
    // C++ [dcl.array]p1:
    //   T is called the array element type; this type shall not be a reference
    //   type, the (possibly cv-qualified) type void, a function type or an 
    //   abstract class type.
    //
    // Note: function types are handled in the common path with C.
    if (T->isReferenceType()) {
      Diag(Loc, diag::err_illegal_decl_array_of_references)
      << getPrintableNameForEntity(Entity) << T;
      return QualType();
    }
    
    if (T->isVoidType()) {
      Diag(Loc, diag::err_illegal_decl_array_incomplete_type) << T;
      return QualType();
    }
    
    if (RequireNonAbstractType(Brackets.getBegin(), T, 
                               diag::err_array_of_abstract_type))
      return QualType();
    
  } else {
    // C99 6.7.5.2p1: If the element type is an incomplete or function type,
    // reject it (e.g. void ary[7], struct foo ary[7], void ary[7]())
    if (RequireCompleteType(Loc, T,
                            diag::err_illegal_decl_array_incomplete_type))
      return QualType();
  }

  if (T->isFunctionType()) {
    Diag(Loc, diag::err_illegal_decl_array_of_functions)
      << getPrintableNameForEntity(Entity) << T;
    return QualType();
  }

  if (T->getContainedAutoType()) {
    Diag(Loc, diag::err_illegal_decl_array_of_auto)
      << getPrintableNameForEntity(Entity) << T;
    return QualType();
  }

  if (const RecordType *EltTy = T->getAs<RecordType>()) {
    // If the element type is a struct or union that contains a variadic
    // array, accept it as a GNU extension: C99 6.7.2.1p2.
    if (EltTy->getDecl()->hasFlexibleArrayMember())
      Diag(Loc, diag::ext_flexible_array_in_array) << T;
  } else if (T->isObjCObjectType()) {
    Diag(Loc, diag::err_objc_array_of_interfaces) << T;
    return QualType();
  }

  // Do lvalue-to-rvalue conversions on the array size expression.
  if (ArraySize && !ArraySize->isRValue())
    DefaultLvalueConversion(ArraySize);

  // C99 6.7.5.2p1: The size expression shall have integer type.
  // TODO: in theory, if we were insane, we could allow contextual
  // conversions to integer type here.
  if (ArraySize && !ArraySize->isTypeDependent() &&
      !ArraySize->getType()->isIntegralOrUnscopedEnumerationType()) {
    Diag(ArraySize->getLocStart(), diag::err_array_size_non_int)
      << ArraySize->getType() << ArraySize->getSourceRange();
    return QualType();
  }
  llvm::APSInt ConstVal(Context.getTypeSize(Context.getSizeType()));
  if (!ArraySize) {
    if (ASM == ArrayType::Star)
      T = Context.getVariableArrayType(T, 0, ASM, Quals, Brackets);
    else
      T = Context.getIncompleteArrayType(T, ASM, Quals);
  } else if (ArraySize->isTypeDependent() || ArraySize->isValueDependent()) {
    T = Context.getDependentSizedArrayType(T, ArraySize, ASM, Quals, Brackets);
  } else if (!ArraySize->isIntegerConstantExpr(ConstVal, Context) ||
             (!T->isDependentType() && !T->isIncompleteType() &&
              !T->isConstantSizeType())) {
    // Per C99, a variable array is an array with either a non-constant
    // size or an element type that has a non-constant-size
    T = Context.getVariableArrayType(T, ArraySize, ASM, Quals, Brackets);
  } else {
    // C99 6.7.5.2p1: If the expression is a constant expression, it shall
    // have a value greater than zero.
    if (ConstVal.isSigned() && ConstVal.isNegative()) {
      if (Entity)
        Diag(ArraySize->getLocStart(), diag::err_decl_negative_array_size)
          << getPrintableNameForEntity(Entity) << ArraySize->getSourceRange();
      else
        Diag(ArraySize->getLocStart(), diag::err_typecheck_negative_array_size)
          << ArraySize->getSourceRange();
      return QualType();
    }
    if (ConstVal == 0) {
      // GCC accepts zero sized static arrays. We allow them when
      // we're not in a SFINAE context.
      Diag(ArraySize->getLocStart(), 
           isSFINAEContext()? diag::err_typecheck_zero_array_size
                            : diag::ext_typecheck_zero_array_size)
        << ArraySize->getSourceRange();
    } else if (!T->isDependentType() && !T->isVariablyModifiedType() && 
               !T->isIncompleteType()) {
      // Is the array too large?      
      unsigned ActiveSizeBits
        = ConstantArrayType::getNumAddressingBits(Context, T, ConstVal);
      if (ActiveSizeBits > ConstantArrayType::getMaxSizeBits(Context))
        Diag(ArraySize->getLocStart(), diag::err_array_too_large)
          << ConstVal.toString(10)
          << ArraySize->getSourceRange();
    }
    
    T = Context.getConstantArrayType(T, ConstVal, ASM, Quals);
  }
  // If this is not C99, extwarn about VLA's and C99 array size modifiers.
  if (!getLangOptions().C99) {
    if (T->isVariableArrayType()) {
      // Prohibit the use of non-POD types in VLAs.
      if (!T->isDependentType() && 
          !Context.getBaseElementType(T)->isPODType()) {
        Diag(Loc, diag::err_vla_non_pod)
          << Context.getBaseElementType(T);
        return QualType();
      } 
      // Prohibit the use of VLAs during template argument deduction.
      else if (isSFINAEContext()) {
        Diag(Loc, diag::err_vla_in_sfinae);
        return QualType();
      }
      // Just extwarn about VLAs.
      else
        Diag(Loc, diag::ext_vla);
    } else if (ASM != ArrayType::Normal || Quals != 0)
      Diag(Loc, 
           getLangOptions().CPlusPlus? diag::err_c99_array_usage_cxx
                                     : diag::ext_c99_array_usage);
  }

  return T;
}

/// \brief Build an ext-vector type.
///
/// Run the required checks for the extended vector type.
QualType Sema::BuildExtVectorType(QualType T, Expr *ArraySize,
                                  SourceLocation AttrLoc) {
  // unlike gcc's vector_size attribute, we do not allow vectors to be defined
  // in conjunction with complex types (pointers, arrays, functions, etc.).
  if (!T->isDependentType() &&
      !T->isIntegerType() && !T->isRealFloatingType()) {
    Diag(AttrLoc, diag::err_attribute_invalid_vector_type) << T;
    return QualType();
  }

  if (!ArraySize->isTypeDependent() && !ArraySize->isValueDependent()) {
    llvm::APSInt vecSize(32);
    if (!ArraySize->isIntegerConstantExpr(vecSize, Context)) {
      Diag(AttrLoc, diag::err_attribute_argument_not_int)
        << "ext_vector_type" << ArraySize->getSourceRange();
      return QualType();
    }

    // unlike gcc's vector_size attribute, the size is specified as the
    // number of elements, not the number of bytes.
    unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue());

    if (vectorSize == 0) {
      Diag(AttrLoc, diag::err_attribute_zero_size)
      << ArraySize->getSourceRange();
      return QualType();
    }

    if (!T->isDependentType())
      return Context.getExtVectorType(T, vectorSize);
  }

  return Context.getDependentSizedExtVectorType(T, ArraySize, AttrLoc);
}

/// \brief Build a function type.
///
/// This routine checks the function type according to C++ rules and
/// under the assumption that the result type and parameter types have
/// just been instantiated from a template. It therefore duplicates
/// some of the behavior of GetTypeForDeclarator, but in a much
/// simpler form that is only suitable for this narrow use case.
///
/// \param T The return type of the function.
///
/// \param ParamTypes The parameter types of the function. This array
/// will be modified to account for adjustments to the types of the
/// function parameters.
///
/// \param NumParamTypes The number of parameter types in ParamTypes.
///
/// \param Variadic Whether this is a variadic function type.
///
/// \param Quals The cvr-qualifiers to be applied to the function type.
///
/// \param Loc The location of the entity whose type involves this
/// function type or, if there is no such entity, the location of the
/// type that will have function type.
///
/// \param Entity The name of the entity that involves the function
/// type, if known.
///
/// \returns A suitable function type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildFunctionType(QualType T,
                                 QualType *ParamTypes,
                                 unsigned NumParamTypes,
                                 bool Variadic, unsigned Quals,
                                 RefQualifierKind RefQualifier,
                                 SourceLocation Loc, DeclarationName Entity,
                                 FunctionType::ExtInfo Info) {
  if (T->isArrayType() || T->isFunctionType()) {
    Diag(Loc, diag::err_func_returning_array_function) 
      << T->isFunctionType() << T;
    return QualType();
  }
       
  bool Invalid = false;
  for (unsigned Idx = 0; Idx < NumParamTypes; ++Idx) {
    QualType ParamType = adjustParameterType(ParamTypes[Idx]);
    if (ParamType->isVoidType()) {
      Diag(Loc, diag::err_param_with_void_type);
      Invalid = true;
    }

    ParamTypes[Idx] = ParamType;
  }

  if (Invalid)
    return QualType();

  FunctionProtoType::ExtProtoInfo EPI;
  EPI.Variadic = Variadic;
  EPI.TypeQuals = Quals;
  EPI.RefQualifier = RefQualifier;
  EPI.ExtInfo = Info;

  return Context.getFunctionType(T, ParamTypes, NumParamTypes, EPI);
}

/// \brief Build a member pointer type \c T Class::*.
///
/// \param T the type to which the member pointer refers.
/// \param Class the class type into which the member pointer points.
/// \param CVR Qualifiers applied to the member pointer type
/// \param Loc the location where this type begins
/// \param Entity the name of the entity that will have this member pointer type
///
/// \returns a member pointer type, if successful, or a NULL type if there was
/// an error.
QualType Sema::BuildMemberPointerType(QualType T, QualType Class,
                                      SourceLocation Loc,
                                      DeclarationName Entity) {
  // Verify that we're not building a pointer to pointer to function with
  // exception specification.
  if (CheckDistantExceptionSpec(T)) {
    Diag(Loc, diag::err_distant_exception_spec);

    // FIXME: If we're doing this as part of template instantiation,
    // we should return immediately.

    // Build the type anyway, but use the canonical type so that the
    // exception specifiers are stripped off.
    T = Context.getCanonicalType(T);
  }

  // C++ 8.3.3p3: A pointer to member shall not point to ... a member
  //   with reference type, or "cv void."
  if (T->isReferenceType()) {
    Diag(Loc, diag::err_illegal_decl_mempointer_to_reference)
      << (Entity? Entity.getAsString() : "type name") << T;
    return QualType();
  }

  if (T->isVoidType()) {
    Diag(Loc, diag::err_illegal_decl_mempointer_to_void)
      << (Entity? Entity.getAsString() : "type name");
    return QualType();
  }

  if (!Class->isDependentType() && !Class->isRecordType()) {
    Diag(Loc, diag::err_mempointer_in_nonclass_type) << Class;
    return QualType();
  }

  // In the Microsoft ABI, the class is allowed to be an incomplete
  // type. In such cases, the compiler makes a worst-case assumption.
  // We make no such assumption right now, so emit an error if the
  // class isn't a complete type.
  if (Context.Target.getCXXABI() == CXXABI_Microsoft &&
      RequireCompleteType(Loc, Class, diag::err_incomplete_type))
    return QualType();

  return Context.getMemberPointerType(T, Class.getTypePtr());
}

/// \brief Build a block pointer type.
///
/// \param T The type to which we'll be building a block pointer.
///
/// \param CVR The cvr-qualifiers to be applied to the block pointer type.
///
/// \param Loc The location of the entity whose type involves this
/// block pointer type or, if there is no such entity, the location of the
/// type that will have block pointer type.
///
/// \param Entity The name of the entity that involves the block pointer
/// type, if known.
///
/// \returns A suitable block pointer type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildBlockPointerType(QualType T, 
                                     SourceLocation Loc,
                                     DeclarationName Entity) {
  if (!T->isFunctionType()) {
    Diag(Loc, diag::err_nonfunction_block_type);
    return QualType();
  }

  return Context.getBlockPointerType(T);
}

QualType Sema::GetTypeFromParser(ParsedType Ty, TypeSourceInfo **TInfo) {
  QualType QT = Ty.get();
  if (QT.isNull()) {
    if (TInfo) *TInfo = 0;
    return QualType();
  }

  TypeSourceInfo *DI = 0;
  if (const LocInfoType *LIT = dyn_cast<LocInfoType>(QT)) {
    QT = LIT->getType();
    DI = LIT->getTypeSourceInfo();
  }

  if (TInfo) *TInfo = DI;
  return QT;
}

static void DiagnoseIgnoredQualifiers(unsigned Quals,
                                      SourceLocation ConstQualLoc,
                                      SourceLocation VolatileQualLoc,
                                      SourceLocation RestrictQualLoc,
                                      Sema& S) {
  std::string QualStr;
  unsigned NumQuals = 0;
  SourceLocation Loc;

  FixItHint ConstFixIt;
  FixItHint VolatileFixIt;
  FixItHint RestrictFixIt;

  // FIXME: The locations here are set kind of arbitrarily. It'd be nicer to
  // find a range and grow it to encompass all the qualifiers, regardless of
  // the order in which they textually appear.
  if (Quals & Qualifiers::Const) {
    ConstFixIt = FixItHint::CreateRemoval(ConstQualLoc);
    Loc = ConstQualLoc;
    ++NumQuals;
    QualStr = "const";
  }
  if (Quals & Qualifiers::Volatile) {
    VolatileFixIt = FixItHint::CreateRemoval(VolatileQualLoc);
    if (NumQuals == 0) {
      Loc = VolatileQualLoc;
      QualStr = "volatile";
    } else {
      QualStr += " volatile";
    }
    ++NumQuals;
  }
  if (Quals & Qualifiers::Restrict) {
    RestrictFixIt = FixItHint::CreateRemoval(RestrictQualLoc);
    if (NumQuals == 0) {
      Loc = RestrictQualLoc;
      QualStr = "restrict";
    } else {
      QualStr += " restrict";
    }
    ++NumQuals;
  }

  assert(NumQuals > 0 && "No known qualifiers?");

  S.Diag(Loc, diag::warn_qual_return_type)
    << QualStr << NumQuals
    << ConstFixIt << VolatileFixIt << RestrictFixIt;
}

/// GetTypeForDeclarator - Convert the type for the specified
/// declarator to Type instances.
///
/// If OwnedDecl is non-NULL, and this declarator's decl-specifier-seq
/// owns the declaration of a type (e.g., the definition of a struct
/// type), then *OwnedDecl will receive the owned declaration.
///
/// The result of this call will never be null, but the associated
/// type may be a null type if there's an unrecoverable error.
TypeSourceInfo *Sema::GetTypeForDeclarator(Declarator &D, Scope *S,
                                           TagDecl **OwnedDecl,
                                           bool AutoAllowedInTypeName) {
  // Determine the type of the declarator. Not all forms of declarator
  // have a type.
  QualType T;
  TypeSourceInfo *ReturnTypeInfo = 0;

  TypeProcessingState state(*this, D);

  switch (D.getName().getKind()) {
  case UnqualifiedId::IK_Identifier:
  case UnqualifiedId::IK_OperatorFunctionId:
  case UnqualifiedId::IK_LiteralOperatorId:
  case UnqualifiedId::IK_TemplateId:
    T = ConvertDeclSpecToType(*this, state);
    
    if (!D.isInvalidType() && D.getDeclSpec().isTypeSpecOwned()) {
      TagDecl* Owned = cast<TagDecl>(D.getDeclSpec().getRepAsDecl());
      // Owned is embedded if it was defined here, or if it is the
      // very first (i.e., canonical) declaration of this tag type.
      Owned->setEmbeddedInDeclarator(Owned->isDefinition() ||
                                     Owned->isCanonicalDecl());
      if (OwnedDecl) *OwnedDecl = Owned;
    }
    break;

  case UnqualifiedId::IK_ConstructorName:
  case UnqualifiedId::IK_ConstructorTemplateId:
  case UnqualifiedId::IK_DestructorName:
    // Constructors and destructors don't have return types. Use
    // "void" instead. 
    T = Context.VoidTy;
    break;

  case UnqualifiedId::IK_ConversionFunctionId:
    // The result type of a conversion function is the type that it
    // converts to.
    T = GetTypeFromParser(D.getName().ConversionFunctionId, 
                          &ReturnTypeInfo);
    break;
  }

  if (D.getAttributes())
    distributeTypeAttrsFromDeclarator(state, T);

  // C++0x [dcl.spec.auto]p5: reject 'auto' if it is not in an allowed context.
  // In C++0x, a function declarator using 'auto' must have a trailing return
  // type (this is checked later) and we can skip this. In other languages
  // using auto, we need to check regardless.
  if (D.getDeclSpec().getTypeSpecType() == DeclSpec::TST_auto &&
      (!getLangOptions().CPlusPlus0x || !D.isFunctionDeclarator())) {
    int Error = -1;

    switch (D.getContext()) {
    case Declarator::KNRTypeListContext:
      assert(0 && "K&R type lists aren't allowed in C++");
      break;
    case Declarator::PrototypeContext:
      Error = 0; // Function prototype
      break;
    case Declarator::MemberContext:
      switch (cast<TagDecl>(CurContext)->getTagKind()) {
      case TTK_Enum: assert(0 && "unhandled tag kind"); break;
      case TTK_Struct: Error = 1; /* Struct member */ break;
      case TTK_Union:  Error = 2; /* Union member */ break;
      case TTK_Class:  Error = 3; /* Class member */ break;
      }
      break;
    case Declarator::CXXCatchContext:
      Error = 4; // Exception declaration
      break;
    case Declarator::TemplateParamContext:
      Error = 5; // Template parameter
      break;
    case Declarator::BlockLiteralContext:
      Error = 6; // Block literal
      break;
    case Declarator::TemplateTypeArgContext:
      Error = 7; // Template type argument
      break;
    case Declarator::TypeNameContext:
      if (!AutoAllowedInTypeName)
        Error = 10; // Generic
      break;
    case Declarator::FileContext:
    case Declarator::BlockContext:
    case Declarator::ForContext:
    case Declarator::ConditionContext:
      break;
    }

    if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
      Error = 8;

    // In Objective-C it is an error to use 'auto' on a function declarator.
    if (D.isFunctionDeclarator())
      Error = 9;

    // C++0x [dcl.spec.auto]p2: 'auto' is always fine if the declarator
    // contains a trailing return type. That is only legal at the outermost
    // level. Check all declarator chunks (outermost first) anyway, to give
    // better diagnostics.
    if (getLangOptions().CPlusPlus0x && Error != -1) {
      for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
        unsigned chunkIndex = e - i - 1;
        state.setCurrentChunkIndex(chunkIndex);
        DeclaratorChunk &DeclType = D.getTypeObject(chunkIndex);
        if (DeclType.Kind == DeclaratorChunk::Function) {
          const DeclaratorChunk::FunctionTypeInfo &FTI = DeclType.Fun;
          if (FTI.TrailingReturnType) {
            Error = -1;
            break;
          }
        }
      }
    }

    if (Error != -1) {
      Diag(D.getDeclSpec().getTypeSpecTypeLoc(), diag::err_auto_not_allowed)
        << Error;
      T = Context.IntTy;
      D.setInvalidType(true);
    }
  }
  
  if (T.isNull())
    return Context.getNullTypeSourceInfo();

  // The name we're declaring, if any.
  DeclarationName Name;
  if (D.getIdentifier())
    Name = D.getIdentifier();

  // Walk the DeclTypeInfo, building the recursive type as we go.
  // DeclTypeInfos are ordered from the identifier out, which is
  // opposite of what we want :).
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
    unsigned chunkIndex = e - i - 1;
    state.setCurrentChunkIndex(chunkIndex);
    DeclaratorChunk &DeclType = D.getTypeObject(chunkIndex);
    switch (DeclType.Kind) {
    default: assert(0 && "Unknown decltype!");
    case DeclaratorChunk::Paren:
      T = BuildParenType(T);
      break;
    case DeclaratorChunk::BlockPointer:
      // If blocks are disabled, emit an error.
      if (!LangOpts.Blocks)
        Diag(DeclType.Loc, diag::err_blocks_disable);

      T = BuildBlockPointerType(T, D.getIdentifierLoc(), Name);
      if (DeclType.Cls.TypeQuals)
        T = BuildQualifiedType(T, DeclType.Loc, DeclType.Cls.TypeQuals);
      break;
    case DeclaratorChunk::Pointer:
      // Verify that we're not building a pointer to pointer to function with
      // exception specification.
      if (getLangOptions().CPlusPlus && CheckDistantExceptionSpec(T)) {
        Diag(D.getIdentifierLoc(), diag::err_distant_exception_spec);
        D.setInvalidType(true);
        // Build the type anyway.
      }
      if (getLangOptions().ObjC1 && T->getAs<ObjCObjectType>()) {
        T = Context.getObjCObjectPointerType(T);
        if (DeclType.Ptr.TypeQuals)
          T = BuildQualifiedType(T, DeclType.Loc, DeclType.Ptr.TypeQuals);
        break;
      }
      T = BuildPointerType(T, DeclType.Loc, Name);
      if (DeclType.Ptr.TypeQuals)
        T = BuildQualifiedType(T, DeclType.Loc, DeclType.Ptr.TypeQuals);

      break;
    case DeclaratorChunk::Reference: {
      // Verify that we're not building a reference to pointer to function with
      // exception specification.
      if (getLangOptions().CPlusPlus && CheckDistantExceptionSpec(T)) {
        Diag(D.getIdentifierLoc(), diag::err_distant_exception_spec);
        D.setInvalidType(true);
        // Build the type anyway.
      }
      T = BuildReferenceType(T, DeclType.Ref.LValueRef, DeclType.Loc, Name);

      Qualifiers Quals;
      if (DeclType.Ref.HasRestrict)
        T = BuildQualifiedType(T, DeclType.Loc, Qualifiers::Restrict);
      break;
    }
    case DeclaratorChunk::Array: {
      // Verify that we're not building an array of pointers to function with
      // exception specification.
      if (getLangOptions().CPlusPlus && CheckDistantExceptionSpec(T)) {
        Diag(D.getIdentifierLoc(), diag::err_distant_exception_spec);
        D.setInvalidType(true);
        // Build the type anyway.
      }
      DeclaratorChunk::ArrayTypeInfo &ATI = DeclType.Arr;
      Expr *ArraySize = static_cast<Expr*>(ATI.NumElts);
      ArrayType::ArraySizeModifier ASM;
      if (ATI.isStar)
        ASM = ArrayType::Star;
      else if (ATI.hasStatic)
        ASM = ArrayType::Static;
      else
        ASM = ArrayType::Normal;
      if (ASM == ArrayType::Star &&
          D.getContext() != Declarator::PrototypeContext) {
        // FIXME: This check isn't quite right: it allows star in prototypes
        // for function definitions, and disallows some edge cases detailed
        // in http://gcc.gnu.org/ml/gcc-patches/2009-02/msg00133.html
        Diag(DeclType.Loc, diag::err_array_star_outside_prototype);
        ASM = ArrayType::Normal;
        D.setInvalidType(true);
      }
      T = BuildArrayType(T, ASM, ArraySize,
                         Qualifiers::fromCVRMask(ATI.TypeQuals),
                         SourceRange(DeclType.Loc, DeclType.EndLoc), Name);
      break;
    }
    case DeclaratorChunk::Function: {
      // If the function declarator has a prototype (i.e. it is not () and
      // does not have a K&R-style identifier list), then the arguments are part
      // of the type, otherwise the argument list is ().
      const DeclaratorChunk::FunctionTypeInfo &FTI = DeclType.Fun;

      // Check for auto functions and trailing return type and adjust the
      // return type accordingly.
      if (!D.isInvalidType()) {
        // trailing-return-type is only required if we're declaring a function,
        // and not, for instance, a pointer to a function.
        if (D.getDeclSpec().getTypeSpecType() == DeclSpec::TST_auto &&
            !FTI.TrailingReturnType && chunkIndex == 0) {
          Diag(D.getDeclSpec().getTypeSpecTypeLoc(),
               diag::err_auto_missing_trailing_return);
          T = Context.IntTy;
          D.setInvalidType(true);
        } else if (FTI.TrailingReturnType) {
          // T must be exactly 'auto' at this point. See CWG issue 681.
          if (isa<ParenType>(T)) {
            Diag(D.getDeclSpec().getTypeSpecTypeLoc(),
                 diag::err_trailing_return_in_parens)
              << T << D.getDeclSpec().getSourceRange();
            D.setInvalidType(true);
          } else if (T.hasQualifiers() || !isa<AutoType>(T)) {
            Diag(D.getDeclSpec().getTypeSpecTypeLoc(),
                 diag::err_trailing_return_without_auto)
              << T << D.getDeclSpec().getSourceRange();
            D.setInvalidType(true);
          }

          T = GetTypeFromParser(
            ParsedType::getFromOpaquePtr(FTI.TrailingReturnType),
            &ReturnTypeInfo);
        }
      }

      // C99 6.7.5.3p1: The return type may not be a function or array type.
      // For conversion functions, we'll diagnose this particular error later.
      if ((T->isArrayType() || T->isFunctionType()) &&
          (D.getName().getKind() != UnqualifiedId::IK_ConversionFunctionId)) {
        unsigned diagID = diag::err_func_returning_array_function;
        // Last processing chunk in block context means this function chunk
        // represents the block.
        if (chunkIndex == 0 &&
            D.getContext() == Declarator::BlockLiteralContext)
          diagID = diag::err_block_returning_array_function;
        Diag(DeclType.Loc, diagID) << T->isFunctionType() << T;
        T = Context.IntTy;
        D.setInvalidType(true);
      }

      // cv-qualifiers on return types are pointless except when the type is a
      // class type in C++.
      if (isa<PointerType>(T) && T.getLocalCVRQualifiers() &&
          (D.getName().getKind() != UnqualifiedId::IK_ConversionFunctionId) &&
          (!getLangOptions().CPlusPlus || !T->isDependentType())) {
        assert(chunkIndex + 1 < e && "No DeclaratorChunk for the return type?");
        DeclaratorChunk ReturnTypeChunk = D.getTypeObject(chunkIndex + 1);
        assert(ReturnTypeChunk.Kind == DeclaratorChunk::Pointer);

        DeclaratorChunk::PointerTypeInfo &PTI = ReturnTypeChunk.Ptr;

        DiagnoseIgnoredQualifiers(PTI.TypeQuals,
            SourceLocation::getFromRawEncoding(PTI.ConstQualLoc),
            SourceLocation::getFromRawEncoding(PTI.VolatileQualLoc),
            SourceLocation::getFromRawEncoding(PTI.RestrictQualLoc),
            *this);

      } else if (T.getCVRQualifiers() && D.getDeclSpec().getTypeQualifiers() &&
          (!getLangOptions().CPlusPlus ||
           (!T->isDependentType() && !T->isRecordType()))) {

        DiagnoseIgnoredQualifiers(D.getDeclSpec().getTypeQualifiers(),
                                  D.getDeclSpec().getConstSpecLoc(),
                                  D.getDeclSpec().getVolatileSpecLoc(),
                                  D.getDeclSpec().getRestrictSpecLoc(),
                                  *this);
      }

      if (getLangOptions().CPlusPlus && D.getDeclSpec().isTypeSpecOwned()) {
        // C++ [dcl.fct]p6:
        //   Types shall not be defined in return or parameter types.
        TagDecl *Tag = cast<TagDecl>(D.getDeclSpec().getRepAsDecl());
        if (Tag->isDefinition())
          Diag(Tag->getLocation(), diag::err_type_defined_in_result_type)
            << Context.getTypeDeclType(Tag);
      }

      // Exception specs are not allowed in typedefs. Complain, but add it
      // anyway.
      if (FTI.hasExceptionSpec &&
          D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
        Diag(FTI.getThrowLoc(), diag::err_exception_spec_in_typedef);

      if (!FTI.NumArgs && !FTI.isVariadic && !getLangOptions().CPlusPlus) {
        // Simple void foo(), where the incoming T is the result type.
        T = Context.getFunctionNoProtoType(T);
      } else {
        // We allow a zero-parameter variadic function in C if the
        // function is marked with the "overloadable" attribute. Scan
        // for this attribute now.
        if (!FTI.NumArgs && FTI.isVariadic && !getLangOptions().CPlusPlus) {
          bool Overloadable = false;
          for (const AttributeList *Attrs = D.getAttributes();
               Attrs; Attrs = Attrs->getNext()) {
            if (Attrs->getKind() == AttributeList::AT_overloadable) {
              Overloadable = true;
              break;
            }
          }

          if (!Overloadable)
            Diag(FTI.getEllipsisLoc(), diag::err_ellipsis_first_arg);
        }

        if (FTI.NumArgs && FTI.ArgInfo[0].Param == 0) {
          // C99 6.7.5.3p3: Reject int(x,y,z) when it's not a function
          // definition.
          Diag(FTI.ArgInfo[0].IdentLoc, diag::err_ident_list_in_fn_declaration);
          D.setInvalidType(true);
          break;
        }

        FunctionProtoType::ExtProtoInfo EPI;
        EPI.Variadic = FTI.isVariadic;
        EPI.TypeQuals = FTI.TypeQuals;
        EPI.RefQualifier = !FTI.hasRefQualifier()? RQ_None
                    : FTI.RefQualifierIsLValueRef? RQ_LValue
                    : RQ_RValue;
        
        // Otherwise, we have a function with an argument list that is
        // potentially variadic.
        llvm::SmallVector<QualType, 16> ArgTys;
        ArgTys.reserve(FTI.NumArgs);

        for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
          ParmVarDecl *Param = cast<ParmVarDecl>(FTI.ArgInfo[i].Param);
          QualType ArgTy = Param->getType();
          assert(!ArgTy.isNull() && "Couldn't parse type?");

          // Adjust the parameter type.
          assert((ArgTy == adjustParameterType(ArgTy)) && "Unadjusted type?");

          // Look for 'void'.  void is allowed only as a single argument to a
          // function with no other parameters (C99 6.7.5.3p10).  We record
          // int(void) as a FunctionProtoType with an empty argument list.
          if (ArgTy->isVoidType()) {
            // If this is something like 'float(int, void)', reject it.  'void'
            // is an incomplete type (C99 6.2.5p19) and function decls cannot
            // have arguments of incomplete type.
            if (FTI.NumArgs != 1 || FTI.isVariadic) {
              Diag(DeclType.Loc, diag::err_void_only_param);
              ArgTy = Context.IntTy;
              Param->setType(ArgTy);
            } else if (FTI.ArgInfo[i].Ident) {
              // Reject, but continue to parse 'int(void abc)'.
              Diag(FTI.ArgInfo[i].IdentLoc,
                   diag::err_param_with_void_type);
              ArgTy = Context.IntTy;
              Param->setType(ArgTy);
            } else {
              // Reject, but continue to parse 'float(const void)'.
              if (ArgTy.hasQualifiers())
                Diag(DeclType.Loc, diag::err_void_param_qualified);

              // Do not add 'void' to the ArgTys list.
              break;
            }
          } else if (!FTI.hasPrototype) {
            if (ArgTy->isPromotableIntegerType()) {
              ArgTy = Context.getPromotedIntegerType(ArgTy);
            } else if (const BuiltinType* BTy = ArgTy->getAs<BuiltinType>()) {
              if (BTy->getKind() == BuiltinType::Float)
                ArgTy = Context.DoubleTy;
            }
          }

          ArgTys.push_back(ArgTy);
        }

        llvm::SmallVector<QualType, 4> Exceptions;
        if (FTI.hasExceptionSpec) {
          EPI.HasExceptionSpec = FTI.hasExceptionSpec;
          EPI.HasAnyExceptionSpec = FTI.hasAnyExceptionSpec;
          Exceptions.reserve(FTI.NumExceptions);
          for (unsigned ei = 0, ee = FTI.NumExceptions; ei != ee; ++ei) {
            // FIXME: Preserve type source info.
            QualType ET = GetTypeFromParser(FTI.Exceptions[ei].Ty);
            // Check that the type is valid for an exception spec, and
            // drop it if not.
            if (!CheckSpecifiedExceptionType(ET, FTI.Exceptions[ei].Range))
              Exceptions.push_back(ET);
          }
          EPI.NumExceptions = Exceptions.size();
          EPI.Exceptions = Exceptions.data();
        }

        T = Context.getFunctionType(T, ArgTys.data(), ArgTys.size(), EPI);
      }

      break;
    }
    case DeclaratorChunk::MemberPointer:
      // The scope spec must refer to a class, or be dependent.
      CXXScopeSpec &SS = DeclType.Mem.Scope();
      QualType ClsType;
      if (SS.isInvalid()) {
        // Avoid emitting extra errors if we already errored on the scope.
        D.setInvalidType(true);
      } else if (isDependentScopeSpecifier(SS) ||
                 dyn_cast_or_null<CXXRecordDecl>(computeDeclContext(SS))) {
        NestedNameSpecifier *NNS
          = static_cast<NestedNameSpecifier*>(SS.getScopeRep());
        NestedNameSpecifier *NNSPrefix = NNS->getPrefix();
        switch (NNS->getKind()) {
        case NestedNameSpecifier::Identifier:
          ClsType = Context.getDependentNameType(ETK_None, NNSPrefix,
                                                 NNS->getAsIdentifier());
          break;

        case NestedNameSpecifier::Namespace:
        case NestedNameSpecifier::NamespaceAlias:
        case NestedNameSpecifier::Global:
          llvm_unreachable("Nested-name-specifier must name a type");
          break;

        case NestedNameSpecifier::TypeSpec:
        case NestedNameSpecifier::TypeSpecWithTemplate:
          ClsType = QualType(NNS->getAsType(), 0);
          // Note: if NNS is dependent, then its prefix (if any) is already
          // included in ClsType; this does not hold if the NNS is
          // nondependent: in this case (if there is indeed a prefix)
          // ClsType needs to be wrapped into an elaborated type.
          if (NNSPrefix && !NNS->isDependent())
            ClsType = Context.getElaboratedType(ETK_None, NNSPrefix, ClsType);
          break;
        }
      } else {
        Diag(DeclType.Mem.Scope().getBeginLoc(),
             diag::err_illegal_decl_mempointer_in_nonclass)
          << (D.getIdentifier() ? D.getIdentifier()->getName() : "type name")
          << DeclType.Mem.Scope().getRange();
        D.setInvalidType(true);
      }

      if (!ClsType.isNull())
        T = BuildMemberPointerType(T, ClsType, DeclType.Loc, D.getIdentifier());
      if (T.isNull()) {
        T = Context.IntTy;
        D.setInvalidType(true);
      } else if (DeclType.Mem.TypeQuals) {
        T = BuildQualifiedType(T, DeclType.Loc, DeclType.Mem.TypeQuals);
      }
      break;
    }

    if (T.isNull()) {
      D.setInvalidType(true);
      T = Context.IntTy;
    }

    // See if there are any attributes on this declarator chunk.
    if (AttributeList *attrs = const_cast<AttributeList*>(DeclType.getAttrs()))
      processTypeAttrs(state, T, false, attrs);
  }

  if (getLangOptions().CPlusPlus && T->isFunctionType()) {
    const FunctionProtoType *FnTy = T->getAs<FunctionProtoType>();
    assert(FnTy && "Why oh why is there not a FunctionProtoType here?");

    // C++ 8.3.5p4: 
    //   A cv-qualifier-seq shall only be part of the function type
    //   for a nonstatic member function, the function type to which a pointer
    //   to member refers, or the top-level function type of a function typedef
    //   declaration.
    //
    // Core issue 547 also allows cv-qualifiers on function types that are
    // top-level template type arguments.
    bool FreeFunction;
    if (!D.getCXXScopeSpec().isSet()) {
      FreeFunction = (D.getContext() != Declarator::MemberContext ||
                      D.getDeclSpec().isFriendSpecified());
    } else {
      DeclContext *DC = computeDeclContext(D.getCXXScopeSpec());
      FreeFunction = (DC && !DC->isRecord());
    }

    // C++0x [dcl.fct]p6:
    //   A ref-qualifier shall only be part of the function type for a
    //   non-static member function, the function type to which a pointer to
    //   member refers, or the top-level function type of a function typedef 
    //   declaration.
    if ((FnTy->getTypeQuals() != 0 || FnTy->getRefQualifier()) &&
        !(D.getContext() == Declarator::TemplateTypeArgContext &&
          !D.isFunctionDeclarator()) &&
        D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef &&
        (FreeFunction ||
         D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_static)) {
      if (D.getContext() == Declarator::TemplateTypeArgContext) {
        // Accept qualified function types as template type arguments as a GNU
        // extension. This is also the subject of C++ core issue 547.
        std::string Quals;
        if (FnTy->getTypeQuals() != 0)
          Quals = Qualifiers::fromCVRMask(FnTy->getTypeQuals()).getAsString();
        
        switch (FnTy->getRefQualifier()) {
        case RQ_None:
          break;
            
        case RQ_LValue:
          if (!Quals.empty())
            Quals += ' ';
          Quals += '&';
          break;
          
        case RQ_RValue:
          if (!Quals.empty())
            Quals += ' ';
          Quals += "&&";
          break;
        }
        
        Diag(D.getIdentifierLoc(), 
             diag::ext_qualified_function_type_template_arg)
          << Quals;
      } else {
        if (FnTy->getTypeQuals() != 0) {
          if (D.isFunctionDeclarator())
            Diag(D.getIdentifierLoc(), 
                 diag::err_invalid_qualified_function_type);
          else
            Diag(D.getIdentifierLoc(),
                 diag::err_invalid_qualified_typedef_function_type_use)
              << FreeFunction;
        }
          
        if (FnTy->getRefQualifier()) {
          if (D.isFunctionDeclarator()) {
            SourceLocation Loc = D.getIdentifierLoc();
            for (unsigned I = 0, N = D.getNumTypeObjects(); I != N; ++I) {
              const DeclaratorChunk &Chunk = D.getTypeObject(N-I-1);
              if (Chunk.Kind == DeclaratorChunk::Function &&
                  Chunk.Fun.hasRefQualifier()) {
                Loc = Chunk.Fun.getRefQualifierLoc();
                break;
              }
            }

            Diag(Loc, diag::err_invalid_ref_qualifier_function_type)
              << (FnTy->getRefQualifier() == RQ_LValue)
              << FixItHint::CreateRemoval(Loc);
          } else {
            Diag(D.getIdentifierLoc(), 
                 diag::err_invalid_ref_qualifier_typedef_function_type_use)
              << FreeFunction
              << (FnTy->getRefQualifier() == RQ_LValue);
          }
        }
          
        // Strip the cv-qualifiers and ref-qualifiers from the type.
        FunctionProtoType::ExtProtoInfo EPI = FnTy->getExtProtoInfo();
        EPI.TypeQuals = 0;
        EPI.RefQualifier = RQ_None;
          
        T = Context.getFunctionType(FnTy->getResultType(), 
                                    FnTy->arg_type_begin(),
                                    FnTy->getNumArgs(), EPI);
      }
    }
  }

  // Apply any undistributed attributes from the declarator.
  if (!T.isNull())
    if (AttributeList *attrs = D.getAttributes())
      processTypeAttrs(state, T, false, attrs);

  // Diagnose any ignored type attributes.
  if (!T.isNull()) state.diagnoseIgnoredTypeAttrs(T);

  // If there's a constexpr specifier, treat it as a top-level const.
  if (D.getDeclSpec().isConstexprSpecified()) {
    T.addConst();
  }

  // If there was an ellipsis in the declarator, the declaration declares a 
  // parameter pack whose type may be a pack expansion type.
  if (D.hasEllipsis() && !T.isNull()) {
    // C++0x [dcl.fct]p13:
    //   A declarator-id or abstract-declarator containing an ellipsis shall 
    //   only be used in a parameter-declaration. Such a parameter-declaration
    //   is a parameter pack (14.5.3). [...]
    switch (D.getContext()) {
    case Declarator::PrototypeContext:
      // C++0x [dcl.fct]p13:
      //   [...] When it is part of a parameter-declaration-clause, the 
      //   parameter pack is a function parameter pack (14.5.3). The type T 
      //   of the declarator-id of the function parameter pack shall contain
      //   a template parameter pack; each template parameter pack in T is 
      //   expanded by the function parameter pack.
      //
      // We represent function parameter packs as function parameters whose
      // type is a pack expansion.
      if (!T->containsUnexpandedParameterPack()) {
        Diag(D.getEllipsisLoc(), 
             diag::err_function_parameter_pack_without_parameter_packs)
          << T <<  D.getSourceRange();
        D.setEllipsisLoc(SourceLocation());
      } else {
        T = Context.getPackExpansionType(T, llvm::Optional<unsigned>());
      }
      break;
        
    case Declarator::TemplateParamContext:
      // C++0x [temp.param]p15:
      //   If a template-parameter is a [...] is a parameter-declaration that 
      //   declares a parameter pack (8.3.5), then the template-parameter is a
      //   template parameter pack (14.5.3).
      //
      // Note: core issue 778 clarifies that, if there are any unexpanded
      // parameter packs in the type of the non-type template parameter, then
      // it expands those parameter packs.
      if (T->containsUnexpandedParameterPack())
        T = Context.getPackExpansionType(T, llvm::Optional<unsigned>());
      else if (!getLangOptions().CPlusPlus0x)
        Diag(D.getEllipsisLoc(), diag::ext_variadic_templates);
      break;
    
    case Declarator::FileContext:
    case Declarator::KNRTypeListContext:
    case Declarator::TypeNameContext:
    case Declarator::MemberContext:
    case Declarator::BlockContext:
    case Declarator::ForContext:
    case Declarator::ConditionContext:
    case Declarator::CXXCatchContext:
    case Declarator::BlockLiteralContext:
    case Declarator::TemplateTypeArgContext:
      // FIXME: We may want to allow parameter packs in block-literal contexts
      // in the future.
      Diag(D.getEllipsisLoc(), diag::err_ellipsis_in_declarator_not_parameter);
      D.setEllipsisLoc(SourceLocation());
      break;
    }
  }

  if (T.isNull())
    return Context.getNullTypeSourceInfo();
  else if (D.isInvalidType())
    return Context.getTrivialTypeSourceInfo(T);
  return GetTypeSourceInfoForDeclarator(D, T, ReturnTypeInfo);
}

/// Map an AttributedType::Kind to an AttributeList::Kind.
static AttributeList::Kind getAttrListKind(AttributedType::Kind kind) {
  switch (kind) {
  case AttributedType::attr_address_space:
    return AttributeList::AT_address_space;
  case AttributedType::attr_regparm:
    return AttributeList::AT_regparm;
  case AttributedType::attr_vector_size:
    return AttributeList::AT_vector_size;
  case AttributedType::attr_neon_vector_type:
    return AttributeList::AT_neon_vector_type;
  case AttributedType::attr_neon_polyvector_type:
    return AttributeList::AT_neon_polyvector_type;
  case AttributedType::attr_objc_gc:
    return AttributeList::AT_objc_gc;
  case AttributedType::attr_noreturn:
    return AttributeList::AT_noreturn;
  case AttributedType::attr_cdecl:
    return AttributeList::AT_cdecl;
  case AttributedType::attr_fastcall:
    return AttributeList::AT_fastcall;
  case AttributedType::attr_stdcall:
    return AttributeList::AT_stdcall;
  case AttributedType::attr_thiscall:
    return AttributeList::AT_thiscall;
  case AttributedType::attr_pascal:
    return AttributeList::AT_pascal;
  }
  llvm_unreachable("unexpected attribute kind!");
  return AttributeList::Kind();
}

static void fillAttributedTypeLoc(AttributedTypeLoc TL,
                                  const AttributeList *attrs) {
  AttributedType::Kind kind = TL.getAttrKind();

  assert(attrs && "no type attributes in the expected location!");
  AttributeList::Kind parsedKind = getAttrListKind(kind);
  while (attrs->getKind() != parsedKind) {
    attrs = attrs->getNext();
    assert(attrs && "no matching attribute in expected location!");
  }

  TL.setAttrNameLoc(attrs->getLoc());
  if (TL.hasAttrExprOperand())
    TL.setAttrExprOperand(attrs->getArg(0));
  else if (TL.hasAttrEnumOperand())
    TL.setAttrEnumOperandLoc(attrs->getParameterLoc());

  // FIXME: preserve this information to here.
  if (TL.hasAttrOperand())
    TL.setAttrOperandParensRange(SourceRange());
}

namespace {
  class TypeSpecLocFiller : public TypeLocVisitor<TypeSpecLocFiller> {
    ASTContext &Context;
    const DeclSpec &DS;

  public:
    TypeSpecLocFiller(ASTContext &Context, const DeclSpec &DS) 
      : Context(Context), DS(DS) {}

    void VisitAttributedTypeLoc(AttributedTypeLoc TL) {
      fillAttributedTypeLoc(TL, DS.getAttributes().getList());
      Visit(TL.getModifiedLoc());
    }
    void VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
      Visit(TL.getUnqualifiedLoc());
    }
    void VisitTypedefTypeLoc(TypedefTypeLoc TL) {
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }
    void VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }
    void VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
      // Handle the base type, which might not have been written explicitly.
      if (DS.getTypeSpecType() == DeclSpec::TST_unspecified) {
        TL.setHasBaseTypeAsWritten(false);
        TL.getBaseLoc().initialize(Context, SourceLocation());
      } else {
        TL.setHasBaseTypeAsWritten(true);
        Visit(TL.getBaseLoc());
      }

      // Protocol qualifiers.
      if (DS.getProtocolQualifiers()) {
        assert(TL.getNumProtocols() > 0);
        assert(TL.getNumProtocols() == DS.getNumProtocolQualifiers());
        TL.setLAngleLoc(DS.getProtocolLAngleLoc());
        TL.setRAngleLoc(DS.getSourceRange().getEnd());
        for (unsigned i = 0, e = DS.getNumProtocolQualifiers(); i != e; ++i)
          TL.setProtocolLoc(i, DS.getProtocolLocs()[i]);
      } else {
        assert(TL.getNumProtocols() == 0);
        TL.setLAngleLoc(SourceLocation());
        TL.setRAngleLoc(SourceLocation());
      }
    }
    void VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
      TL.setStarLoc(SourceLocation());
      Visit(TL.getPointeeLoc());
    }
    void VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL) {
      TypeSourceInfo *TInfo = 0;
      Sema::GetTypeFromParser(DS.getRepAsType(), &TInfo);

      // If we got no declarator info from previous Sema routines,
      // just fill with the typespec loc.
      if (!TInfo) {
        TL.initialize(Context, DS.getTypeSpecTypeLoc());
        return;
      }

      TypeLoc OldTL = TInfo->getTypeLoc();
      if (TInfo->getType()->getAs<ElaboratedType>()) {
        ElaboratedTypeLoc ElabTL = cast<ElaboratedTypeLoc>(OldTL);
        TemplateSpecializationTypeLoc NamedTL =
          cast<TemplateSpecializationTypeLoc>(ElabTL.getNamedTypeLoc());
        TL.copy(NamedTL);
      }
      else
        TL.copy(cast<TemplateSpecializationTypeLoc>(OldTL));
    }
    void VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
      assert(DS.getTypeSpecType() == DeclSpec::TST_typeofExpr);
      TL.setTypeofLoc(DS.getTypeSpecTypeLoc());
      TL.setParensRange(DS.getTypeofParensRange());
    }
    void VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
      assert(DS.getTypeSpecType() == DeclSpec::TST_typeofType);
      TL.setTypeofLoc(DS.getTypeSpecTypeLoc());
      TL.setParensRange(DS.getTypeofParensRange());
      assert(DS.getRepAsType());
      TypeSourceInfo *TInfo = 0;
      Sema::GetTypeFromParser(DS.getRepAsType(), &TInfo);
      TL.setUnderlyingTInfo(TInfo);
    }
    void VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
      // By default, use the source location of the type specifier.
      TL.setBuiltinLoc(DS.getTypeSpecTypeLoc());
      if (TL.needsExtraLocalData()) {
        // Set info for the written builtin specifiers.
        TL.getWrittenBuiltinSpecs() = DS.getWrittenBuiltinSpecs();
        // Try to have a meaningful source location.
        if (TL.getWrittenSignSpec() != TSS_unspecified)
          // Sign spec loc overrides the others (e.g., 'unsigned long').
          TL.setBuiltinLoc(DS.getTypeSpecSignLoc());
        else if (TL.getWrittenWidthSpec() != TSW_unspecified)
          // Width spec loc overrides type spec loc (e.g., 'short int').
          TL.setBuiltinLoc(DS.getTypeSpecWidthLoc());
      }
    }
    void VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
      ElaboratedTypeKeyword Keyword
        = TypeWithKeyword::getKeywordForTypeSpec(DS.getTypeSpecType());
      if (DS.getTypeSpecType() == TST_typename) {
        TypeSourceInfo *TInfo = 0;
        Sema::GetTypeFromParser(DS.getRepAsType(), &TInfo);
        if (TInfo) {
          TL.copy(cast<ElaboratedTypeLoc>(TInfo->getTypeLoc()));
          return;
        }
      }
      TL.setKeywordLoc(Keyword != ETK_None
                       ? DS.getTypeSpecTypeLoc()
                       : SourceLocation());
      const CXXScopeSpec& SS = DS.getTypeSpecScope();
      TL.setQualifierLoc(SS.getWithLocInContext(Context));
      Visit(TL.getNextTypeLoc().getUnqualifiedLoc());
    }
    void VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
      ElaboratedTypeKeyword Keyword
        = TypeWithKeyword::getKeywordForTypeSpec(DS.getTypeSpecType());
      if (DS.getTypeSpecType() == TST_typename) {
        TypeSourceInfo *TInfo = 0;
        Sema::GetTypeFromParser(DS.getRepAsType(), &TInfo);
        if (TInfo) {
          TL.copy(cast<DependentNameTypeLoc>(TInfo->getTypeLoc()));
          return;
        }
      }
      TL.setKeywordLoc(Keyword != ETK_None
                       ? DS.getTypeSpecTypeLoc()
                       : SourceLocation());
      const CXXScopeSpec& SS = DS.getTypeSpecScope();
      TL.setQualifierLoc(SS.getWithLocInContext(Context));
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }
    void VisitDependentTemplateSpecializationTypeLoc(
                                 DependentTemplateSpecializationTypeLoc TL) {
      ElaboratedTypeKeyword Keyword
        = TypeWithKeyword::getKeywordForTypeSpec(DS.getTypeSpecType());
      if (Keyword == ETK_Typename) {
        TypeSourceInfo *TInfo = 0;
        Sema::GetTypeFromParser(DS.getRepAsType(), &TInfo);
        if (TInfo) {
          TL.copy(cast<DependentTemplateSpecializationTypeLoc>(
                    TInfo->getTypeLoc()));
          return;
        }
      }
      TL.initializeLocal(Context, SourceLocation());
      TL.setKeywordLoc(Keyword != ETK_None
                       ? DS.getTypeSpecTypeLoc()
                       : SourceLocation());
      const CXXScopeSpec& SS = DS.getTypeSpecScope();
      TL.setQualifierLoc(SS.getWithLocInContext(Context));
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }

    void VisitTypeLoc(TypeLoc TL) {
      // FIXME: add other typespec types and change this to an assert.
      TL.initialize(Context, DS.getTypeSpecTypeLoc());
    }
  };

  class DeclaratorLocFiller : public TypeLocVisitor<DeclaratorLocFiller> {
    ASTContext &Context;
    const DeclaratorChunk &Chunk;

  public:
    DeclaratorLocFiller(ASTContext &Context, const DeclaratorChunk &Chunk)
      : Context(Context), Chunk(Chunk) {}

    void VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
      llvm_unreachable("qualified type locs not expected here!");
    }

    void VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::BlockPointer);
      TL.setCaretLoc(Chunk.Loc);
    }
    void VisitPointerTypeLoc(PointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Pointer);
      TL.setStarLoc(Chunk.Loc);
    }
    void VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Pointer);
      TL.setStarLoc(Chunk.Loc);
    }
    void VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::MemberPointer);
      const CXXScopeSpec& SS = Chunk.Mem.Scope();
      NestedNameSpecifierLoc NNSLoc = SS.getWithLocInContext(Context);

      const Type* ClsTy = TL.getClass();
      QualType ClsQT = QualType(ClsTy, 0);
      TypeSourceInfo *ClsTInfo = Context.CreateTypeSourceInfo(ClsQT, 0);
      // Now copy source location info into the type loc component.
      TypeLoc ClsTL = ClsTInfo->getTypeLoc();
      switch (NNSLoc.getNestedNameSpecifier()->getKind()) {
      case NestedNameSpecifier::Identifier:
        assert(isa<DependentNameType>(ClsTy) && "Unexpected TypeLoc");
        {
          DependentNameTypeLoc DNTLoc = cast<DependentNameTypeLoc>(ClsTL);
          DNTLoc.setKeywordLoc(SourceLocation());
          DNTLoc.setQualifierLoc(NNSLoc.getPrefix());
          DNTLoc.setNameLoc(NNSLoc.getLocalBeginLoc());
        }
        break;

      case NestedNameSpecifier::TypeSpec:
      case NestedNameSpecifier::TypeSpecWithTemplate:
        if (isa<ElaboratedType>(ClsTy)) {
          ElaboratedTypeLoc ETLoc = *cast<ElaboratedTypeLoc>(&ClsTL);
          ETLoc.setKeywordLoc(SourceLocation());
          ETLoc.setQualifierLoc(NNSLoc.getPrefix());
          TypeLoc NamedTL = ETLoc.getNamedTypeLoc();
          NamedTL.initializeFullCopy(NNSLoc.getTypeLoc());
        } else {
          ClsTL.initializeFullCopy(NNSLoc.getTypeLoc());
        }
        break;

      case NestedNameSpecifier::Namespace:
      case NestedNameSpecifier::NamespaceAlias:
      case NestedNameSpecifier::Global:
        llvm_unreachable("Nested-name-specifier must name a type");
        break;
      }

      // Finally fill in MemberPointerLocInfo fields.
      TL.setStarLoc(Chunk.Loc);
      TL.setClassTInfo(ClsTInfo);
    }
    void VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Reference);
      // 'Amp' is misleading: this might have been originally
      /// spelled with AmpAmp.
      TL.setAmpLoc(Chunk.Loc);
    }
    void VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Reference);
      assert(!Chunk.Ref.LValueRef);
      TL.setAmpAmpLoc(Chunk.Loc);
    }
    void VisitArrayTypeLoc(ArrayTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Array);
      TL.setLBracketLoc(Chunk.Loc);
      TL.setRBracketLoc(Chunk.EndLoc);
      TL.setSizeExpr(static_cast<Expr*>(Chunk.Arr.NumElts));
    }
    void VisitFunctionTypeLoc(FunctionTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Function);
      TL.setLParenLoc(Chunk.Loc);
      TL.setRParenLoc(Chunk.EndLoc);
      TL.setTrailingReturn(!!Chunk.Fun.TrailingReturnType);

      const DeclaratorChunk::FunctionTypeInfo &FTI = Chunk.Fun;
      for (unsigned i = 0, e = TL.getNumArgs(), tpi = 0; i != e; ++i) {
        ParmVarDecl *Param = cast<ParmVarDecl>(FTI.ArgInfo[i].Param);
        TL.setArg(tpi++, Param);
      }
      // FIXME: exception specs
    }
    void VisitParenTypeLoc(ParenTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Paren);
      TL.setLParenLoc(Chunk.Loc);
      TL.setRParenLoc(Chunk.EndLoc);
    }

    void VisitTypeLoc(TypeLoc TL) {
      llvm_unreachable("unsupported TypeLoc kind in declarator!");
    }
  };
}

/// \brief Create and instantiate a TypeSourceInfo with type source information.
///
/// \param T QualType referring to the type as written in source code.
///
/// \param ReturnTypeInfo For declarators whose return type does not show
/// up in the normal place in the declaration specifiers (such as a C++
/// conversion function), this pointer will refer to a type source information
/// for that return type.
TypeSourceInfo *
Sema::GetTypeSourceInfoForDeclarator(Declarator &D, QualType T,
                                     TypeSourceInfo *ReturnTypeInfo) {
  TypeSourceInfo *TInfo = Context.CreateTypeSourceInfo(T);
  UnqualTypeLoc CurrTL = TInfo->getTypeLoc().getUnqualifiedLoc();

  // Handle parameter packs whose type is a pack expansion.
  if (isa<PackExpansionType>(T)) {
    cast<PackExpansionTypeLoc>(CurrTL).setEllipsisLoc(D.getEllipsisLoc());
    CurrTL = CurrTL.getNextTypeLoc().getUnqualifiedLoc();    
  }
  
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
    while (isa<AttributedTypeLoc>(CurrTL)) {
      AttributedTypeLoc TL = cast<AttributedTypeLoc>(CurrTL);
      fillAttributedTypeLoc(TL, D.getTypeObject(i).getAttrs());
      CurrTL = TL.getNextTypeLoc().getUnqualifiedLoc();
    }

    DeclaratorLocFiller(Context, D.getTypeObject(i)).Visit(CurrTL);
    CurrTL = CurrTL.getNextTypeLoc().getUnqualifiedLoc();
  }
  
  // If we have different source information for the return type, use
  // that.  This really only applies to C++ conversion functions.
  if (ReturnTypeInfo) {
    TypeLoc TL = ReturnTypeInfo->getTypeLoc();
    assert(TL.getFullDataSize() == CurrTL.getFullDataSize());
    memcpy(CurrTL.getOpaqueData(), TL.getOpaqueData(), TL.getFullDataSize());
  } else {
    TypeSpecLocFiller(Context, D.getDeclSpec()).Visit(CurrTL);
  }
      
  return TInfo;
}

/// \brief Create a LocInfoType to hold the given QualType and TypeSourceInfo.
ParsedType Sema::CreateParsedType(QualType T, TypeSourceInfo *TInfo) {
  // FIXME: LocInfoTypes are "transient", only needed for passing to/from Parser
  // and Sema during declaration parsing. Try deallocating/caching them when
  // it's appropriate, instead of allocating them and keeping them around.
  LocInfoType *LocT = (LocInfoType*)BumpAlloc.Allocate(sizeof(LocInfoType), 
                                                       TypeAlignment);
  new (LocT) LocInfoType(T, TInfo);
  assert(LocT->getTypeClass() != T->getTypeClass() &&
         "LocInfoType's TypeClass conflicts with an existing Type class");
  return ParsedType::make(QualType(LocT, 0));
}

void LocInfoType::getAsStringInternal(std::string &Str,
                                      const PrintingPolicy &Policy) const {
  assert(false && "LocInfoType leaked into the type system; an opaque TypeTy*"
         " was used directly instead of getting the QualType through"
         " GetTypeFromParser");
}

TypeResult Sema::ActOnTypeName(Scope *S, Declarator &D) {
  // C99 6.7.6: Type names have no identifier.  This is already validated by
  // the parser.
  assert(D.getIdentifier() == 0 && "Type name should have no identifier!");

  TagDecl *OwnedTag = 0;
  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, S, &OwnedTag);
  QualType T = TInfo->getType();
  if (D.isInvalidType())
    return true;

  if (getLangOptions().CPlusPlus) {
    // Check that there are no default arguments (C++ only).
    CheckExtraCXXDefaultArguments(D);

    // C++0x [dcl.type]p3:
    //   A type-specifier-seq shall not define a class or enumeration
    //   unless it appears in the type-id of an alias-declaration
    //   (7.1.3).
    if (OwnedTag && OwnedTag->isDefinition())
      Diag(OwnedTag->getLocation(), diag::err_type_defined_in_type_specifier)
        << Context.getTypeDeclType(OwnedTag);
  }

  return CreateParsedType(T, TInfo);
}



//===----------------------------------------------------------------------===//
// Type Attribute Processing
//===----------------------------------------------------------------------===//

/// HandleAddressSpaceTypeAttribute - Process an address_space attribute on the
/// specified type.  The attribute contains 1 argument, the id of the address
/// space for the type.
static void HandleAddressSpaceTypeAttribute(QualType &Type,
                                            const AttributeList &Attr, Sema &S){

  // If this type is already address space qualified, reject it.
  // Clause 6.7.3 - Type qualifiers: "No type shall be qualified by qualifiers
  // for two or more different address spaces."
  if (Type.getAddressSpace()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_multiple_qualifiers);
    Attr.setInvalid();
    return;
  }

  // Check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  Expr *ASArgExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt addrSpace(32);
  if (ASArgExpr->isTypeDependent() || ASArgExpr->isValueDependent() ||
      !ASArgExpr->isIntegerConstantExpr(addrSpace, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_space_not_int)
      << ASArgExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }

  // Bounds checking.
  if (addrSpace.isSigned()) {
    if (addrSpace.isNegative()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_address_space_negative)
        << ASArgExpr->getSourceRange();
      Attr.setInvalid();
      return;
    }
    addrSpace.setIsSigned(false);
  }
  llvm::APSInt max(addrSpace.getBitWidth());
  max = Qualifiers::MaxAddressSpace;
  if (addrSpace > max) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_space_too_high)
      << Qualifiers::MaxAddressSpace << ASArgExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }

  unsigned ASIdx = static_cast<unsigned>(addrSpace.getZExtValue());
  Type = S.Context.getAddrSpaceQualType(Type, ASIdx);
}

/// handleObjCGCTypeAttr - Process the __attribute__((objc_gc)) type
/// attribute on the specified type.  Returns true to indicate that
/// the attribute was handled, false to indicate that the type does
/// not permit the attribute.
static bool handleObjCGCTypeAttr(TypeProcessingState &state,
                                 AttributeList &attr,
                                 QualType &type) {
  Sema &S = state.getSema();

  // Delay if this isn't some kind of pointer.
  if (!type->isPointerType() &&
      !type->isObjCObjectPointerType() &&
      !type->isBlockPointerType())
    return false;

  if (type.getObjCGCAttr() != Qualifiers::GCNone) {
    S.Diag(attr.getLoc(), diag::err_attribute_multiple_objc_gc);
    attr.setInvalid();
    return true;
  }

  // Check the attribute arguments.
  if (!attr.getParameterName()) {
    S.Diag(attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "objc_gc" << 1;
    attr.setInvalid();
    return true;
  }
  Qualifiers::GC GCAttr;
  if (attr.getNumArgs() != 0) {
    S.Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    attr.setInvalid();
    return true;
  }
  if (attr.getParameterName()->isStr("weak"))
    GCAttr = Qualifiers::Weak;
  else if (attr.getParameterName()->isStr("strong"))
    GCAttr = Qualifiers::Strong;
  else {
    S.Diag(attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "objc_gc" << attr.getParameterName();
    attr.setInvalid();
    return true;
  }

  QualType origType = type;
  type = S.Context.getObjCGCQualType(origType, GCAttr);

  // Make an attributed type to preserve the source information.
  if (attr.getLoc().isValid())
    type = S.Context.getAttributedType(AttributedType::attr_objc_gc,
                                       origType, type);

  return true;
}

namespace {
  /// A helper class to unwrap a type down to a function for the
  /// purposes of applying attributes there.
  ///
  /// Use:
  ///   FunctionTypeUnwrapper unwrapped(SemaRef, T);
  ///   if (unwrapped.isFunctionType()) {
  ///     const FunctionType *fn = unwrapped.get();
  ///     // change fn somehow
  ///     T = unwrapped.wrap(fn);
  ///   }
  struct FunctionTypeUnwrapper {
    enum WrapKind {
      Desugar,
      Parens,
      Pointer,
      BlockPointer,
      Reference,
      MemberPointer
    };

    QualType Original;
    const FunctionType *Fn;
    llvm::SmallVector<unsigned char /*WrapKind*/, 8> Stack;

    FunctionTypeUnwrapper(Sema &S, QualType T) : Original(T) {
      while (true) {
        const Type *Ty = T.getTypePtr();
        if (isa<FunctionType>(Ty)) {
          Fn = cast<FunctionType>(Ty);
          return;
        } else if (isa<ParenType>(Ty)) {
          T = cast<ParenType>(Ty)->getInnerType();
          Stack.push_back(Parens);
        } else if (isa<PointerType>(Ty)) {
          T = cast<PointerType>(Ty)->getPointeeType();
          Stack.push_back(Pointer);
        } else if (isa<BlockPointerType>(Ty)) {
          T = cast<BlockPointerType>(Ty)->getPointeeType();
          Stack.push_back(BlockPointer);
        } else if (isa<MemberPointerType>(Ty)) {
          T = cast<MemberPointerType>(Ty)->getPointeeType();
          Stack.push_back(MemberPointer);
        } else if (isa<ReferenceType>(Ty)) {
          T = cast<ReferenceType>(Ty)->getPointeeType();
          Stack.push_back(Reference);
        } else {
          const Type *DTy = Ty->getUnqualifiedDesugaredType();
          if (Ty == DTy) {
            Fn = 0;
            return;
          }

          T = QualType(DTy, 0);
          Stack.push_back(Desugar);
        }
      }
    }

    bool isFunctionType() const { return (Fn != 0); }
    const FunctionType *get() const { return Fn; }

    QualType wrap(Sema &S, const FunctionType *New) {
      // If T wasn't modified from the unwrapped type, do nothing.
      if (New == get()) return Original;

      Fn = New;
      return wrap(S.Context, Original, 0);
    }

  private:
    QualType wrap(ASTContext &C, QualType Old, unsigned I) {
      if (I == Stack.size())
        return C.getQualifiedType(Fn, Old.getQualifiers());

      // Build up the inner type, applying the qualifiers from the old
      // type to the new type.
      SplitQualType SplitOld = Old.split();

      // As a special case, tail-recurse if there are no qualifiers.
      if (SplitOld.second.empty())
        return wrap(C, SplitOld.first, I);
      return C.getQualifiedType(wrap(C, SplitOld.first, I), SplitOld.second);
    }

    QualType wrap(ASTContext &C, const Type *Old, unsigned I) {
      if (I == Stack.size()) return QualType(Fn, 0);

      switch (static_cast<WrapKind>(Stack[I++])) {
      case Desugar:
        // This is the point at which we potentially lose source
        // information.
        return wrap(C, Old->getUnqualifiedDesugaredType(), I);

      case Parens: {
        QualType New = wrap(C, cast<ParenType>(Old)->getInnerType(), I);
        return C.getParenType(New);
      }

      case Pointer: {
        QualType New = wrap(C, cast<PointerType>(Old)->getPointeeType(), I);
        return C.getPointerType(New);
      }

      case BlockPointer: {
        QualType New = wrap(C, cast<BlockPointerType>(Old)->getPointeeType(),I);
        return C.getBlockPointerType(New);
      }

      case MemberPointer: {
        const MemberPointerType *OldMPT = cast<MemberPointerType>(Old);
        QualType New = wrap(C, OldMPT->getPointeeType(), I);
        return C.getMemberPointerType(New, OldMPT->getClass());
      }

      case Reference: {
        const ReferenceType *OldRef = cast<ReferenceType>(Old);
        QualType New = wrap(C, OldRef->getPointeeType(), I);
        if (isa<LValueReferenceType>(OldRef))
          return C.getLValueReferenceType(New, OldRef->isSpelledAsLValue());
        else
          return C.getRValueReferenceType(New);
      }
      }

      llvm_unreachable("unknown wrapping kind");
      return QualType();
    }
  };
}

/// Process an individual function attribute.  Returns true to
/// indicate that the attribute was handled, false if it wasn't.
static bool handleFunctionTypeAttr(TypeProcessingState &state,
                                   AttributeList &attr,
                                   QualType &type) {
  Sema &S = state.getSema();

  FunctionTypeUnwrapper unwrapped(S, type);

  if (attr.getKind() == AttributeList::AT_noreturn) {
    if (S.CheckNoReturnAttr(attr))
      return true;

    // Delay if this is not a function type.
    if (!unwrapped.isFunctionType())
      return false;

    // Otherwise we can process right away.
    FunctionType::ExtInfo EI = unwrapped.get()->getExtInfo().withNoReturn(true);
    type = unwrapped.wrap(S, S.Context.adjustFunctionType(unwrapped.get(), EI));
    return true;
  }

  if (attr.getKind() == AttributeList::AT_regparm) {
    unsigned value;
    if (S.CheckRegparmAttr(attr, value))
      return true;

    // Delay if this is not a function type.
    if (!unwrapped.isFunctionType())
      return false;

    // Diagnose regparm with fastcall.
    const FunctionType *fn = unwrapped.get();
    CallingConv CC = fn->getCallConv();
    if (CC == CC_X86FastCall) {
      S.Diag(attr.getLoc(), diag::err_attributes_are_not_compatible)
        << FunctionType::getNameForCallConv(CC)
        << "regparm";
      attr.setInvalid();
      return true;
    }

    FunctionType::ExtInfo EI = 
      unwrapped.get()->getExtInfo().withRegParm(value);
    type = unwrapped.wrap(S, S.Context.adjustFunctionType(unwrapped.get(), EI));
    return true;
  }

  // Otherwise, a calling convention.
  CallingConv CC;
  if (S.CheckCallingConvAttr(attr, CC))
    return true;

  // Delay if the type didn't work out to a function.
  if (!unwrapped.isFunctionType()) return false;

  const FunctionType *fn = unwrapped.get();
  CallingConv CCOld = fn->getCallConv();
  if (S.Context.getCanonicalCallConv(CC) ==
      S.Context.getCanonicalCallConv(CCOld)) {
    FunctionType::ExtInfo EI= unwrapped.get()->getExtInfo().withCallingConv(CC);
    type = unwrapped.wrap(S, S.Context.adjustFunctionType(unwrapped.get(), EI));
    return true;
  }

  if (CCOld != CC_Default) {
    // Should we diagnose reapplications of the same convention?
    S.Diag(attr.getLoc(), diag::err_attributes_are_not_compatible)
      << FunctionType::getNameForCallConv(CC)
      << FunctionType::getNameForCallConv(CCOld);
    attr.setInvalid();
    return true;
  }

  // Diagnose the use of X86 fastcall on varargs or unprototyped functions.
  if (CC == CC_X86FastCall) {
    if (isa<FunctionNoProtoType>(fn)) {
      S.Diag(attr.getLoc(), diag::err_cconv_knr)
        << FunctionType::getNameForCallConv(CC);
      attr.setInvalid();
      return true;
    }

    const FunctionProtoType *FnP = cast<FunctionProtoType>(fn);
    if (FnP->isVariadic()) {
      S.Diag(attr.getLoc(), diag::err_cconv_varargs)
        << FunctionType::getNameForCallConv(CC);
      attr.setInvalid();
      return true;
    }

    // Also diagnose fastcall with regparm.
    if (fn->getRegParmType()) {
      S.Diag(attr.getLoc(), diag::err_attributes_are_not_compatible)
        << "regparm"
        << FunctionType::getNameForCallConv(CC);
      attr.setInvalid();
      return true;
    }
  }

  FunctionType::ExtInfo EI = unwrapped.get()->getExtInfo().withCallingConv(CC);
  type = unwrapped.wrap(S, S.Context.adjustFunctionType(unwrapped.get(), EI));
  return true;
}

/// HandleVectorSizeAttribute - this attribute is only applicable to integral
/// and float scalars, although arrays, pointers, and function return values are
/// allowed in conjunction with this construct. Aggregates with this attribute
/// are invalid, even if they are of the same size as a corresponding scalar.
/// The raw attribute should contain precisely 1 argument, the vector size for
/// the variable, measured in bytes. If curType and rawAttr are well formed,
/// this routine will return a new vector type.
static void HandleVectorSizeAttr(QualType& CurType, const AttributeList &Attr,
                                 Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (sizeExpr->isTypeDependent() || sizeExpr->isValueDependent() ||
      !sizeExpr->isIntegerConstantExpr(vecSize, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "vector_size" << sizeExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  // the base type must be integer or float, and can't already be a vector.
  if (!CurType->isIntegerType() && !CurType->isRealFloatingType()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type) << CurType;
    Attr.setInvalid();
    return;
  }
  unsigned typeSize = static_cast<unsigned>(S.Context.getTypeSize(CurType));
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue() * 8);

  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_size)
      << sizeExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  if (vectorSize == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_zero_size)
      << sizeExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }

  // Success! Instantiate the vector type, the number of elements is > 0, and
  // not required to be a power of 2, unlike GCC.
  CurType = S.Context.getVectorType(CurType, vectorSize/typeSize,
                                    VectorType::GenericVector);
}

/// HandleNeonVectorTypeAttr - The "neon_vector_type" and
/// "neon_polyvector_type" attributes are used to create vector types that
/// are mangled according to ARM's ABI.  Otherwise, these types are identical
/// to those created with the "vector_size" attribute.  Unlike "vector_size"
/// the argument to these Neon attributes is the number of vector elements,
/// not the vector size in bytes.  The vector width and element type must
/// match one of the standard Neon vector types.
static void HandleNeonVectorTypeAttr(QualType& CurType,
                                     const AttributeList &Attr, Sema &S,
                                     VectorType::VectorKind VecKind,
                                     const char *AttrName) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  // The number of elements must be an ICE.
  Expr *numEltsExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt numEltsInt(32);
  if (numEltsExpr->isTypeDependent() || numEltsExpr->isValueDependent() ||
      !numEltsExpr->isIntegerConstantExpr(numEltsInt, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << AttrName << numEltsExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  // Only certain element types are supported for Neon vectors.
  const BuiltinType* BTy = CurType->getAs<BuiltinType>();
  if (!BTy ||
      (VecKind == VectorType::NeonPolyVector &&
       BTy->getKind() != BuiltinType::SChar &&
       BTy->getKind() != BuiltinType::Short) ||
      (BTy->getKind() != BuiltinType::SChar &&
       BTy->getKind() != BuiltinType::UChar &&
       BTy->getKind() != BuiltinType::Short &&
       BTy->getKind() != BuiltinType::UShort &&
       BTy->getKind() != BuiltinType::Int &&
       BTy->getKind() != BuiltinType::UInt &&
       BTy->getKind() != BuiltinType::LongLong &&
       BTy->getKind() != BuiltinType::ULongLong &&
       BTy->getKind() != BuiltinType::Float)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type) <<CurType;
    Attr.setInvalid();
    return;
  }
  // The total size of the vector must be 64 or 128 bits.
  unsigned typeSize = static_cast<unsigned>(S.Context.getTypeSize(CurType));
  unsigned numElts = static_cast<unsigned>(numEltsInt.getZExtValue());
  unsigned vecSize = typeSize * numElts;
  if (vecSize != 64 && vecSize != 128) {
    S.Diag(Attr.getLoc(), diag::err_attribute_bad_neon_vector_size) << CurType;
    Attr.setInvalid();
    return;
  }

  CurType = S.Context.getVectorType(CurType, numElts, VecKind);
}

static void processTypeAttrs(TypeProcessingState &state, QualType &type,
                             bool isDeclSpec, AttributeList *attrs) {
  // Scan through and apply attributes to this type where it makes sense.  Some
  // attributes (such as __address_space__, __vector_size__, etc) apply to the
  // type, but others can be present in the type specifiers even though they
  // apply to the decl.  Here we apply type attributes and ignore the rest.

  AttributeList *next;
  do {
    AttributeList &attr = *attrs;
    next = attr.getNext();

    // Skip attributes that were marked to be invalid.
    if (attr.isInvalid())
      continue;

    // If this is an attribute we can handle, do so now,
    // otherwise, add it to the FnAttrs list for rechaining.
    switch (attr.getKind()) {
    default: break;

    case AttributeList::AT_address_space:
      HandleAddressSpaceTypeAttribute(type, attr, state.getSema());
      break;
    OBJC_POINTER_TYPE_ATTRS_CASELIST:
      if (!handleObjCPointerTypeAttr(state, attr, type))
        distributeObjCPointerTypeAttr(state, attr, type);
      break;
    case AttributeList::AT_vector_size:
      HandleVectorSizeAttr(type, attr, state.getSema());
      break;
    case AttributeList::AT_neon_vector_type:
      HandleNeonVectorTypeAttr(type, attr, state.getSema(),
                               VectorType::NeonVector, "neon_vector_type");
      break;
    case AttributeList::AT_neon_polyvector_type:
      HandleNeonVectorTypeAttr(type, attr, state.getSema(),
                               VectorType::NeonPolyVector,
                               "neon_polyvector_type");
      break;

    FUNCTION_TYPE_ATTRS_CASELIST:
      // Never process function type attributes as part of the
      // declaration-specifiers.
      if (isDeclSpec)
        distributeFunctionTypeAttrFromDeclSpec(state, attr, type);

      // Otherwise, handle the possible delays.
      else if (!handleFunctionTypeAttr(state, attr, type))
        distributeFunctionTypeAttr(state, attr, type);
      break;
    }
  } while ((attrs = next));
}

/// @brief Ensure that the type T is a complete type.
///
/// This routine checks whether the type @p T is complete in any
/// context where a complete type is required. If @p T is a complete
/// type, returns false. If @p T is a class template specialization,
/// this routine then attempts to perform class template
/// instantiation. If instantiation fails, or if @p T is incomplete
/// and cannot be completed, issues the diagnostic @p diag (giving it
/// the type @p T) and returns true.
///
/// @param Loc  The location in the source that the incomplete type
/// diagnostic should refer to.
///
/// @param T  The type that this routine is examining for completeness.
///
/// @param PD The partial diagnostic that will be printed out if T is not a
/// complete type.
///
/// @returns @c true if @p T is incomplete and a diagnostic was emitted,
/// @c false otherwise.
bool Sema::RequireCompleteType(SourceLocation Loc, QualType T,
                               const PartialDiagnostic &PD,
                               std::pair<SourceLocation, 
                                         PartialDiagnostic> Note) {
  unsigned diag = PD.getDiagID();

  // FIXME: Add this assertion to make sure we always get instantiation points.
  //  assert(!Loc.isInvalid() && "Invalid location in RequireCompleteType");
  // FIXME: Add this assertion to help us flush out problems with
  // checking for dependent types and type-dependent expressions.
  //
  //  assert(!T->isDependentType() &&
  //         "Can't ask whether a dependent type is complete");

  // If we have a complete type, we're done.
  if (!T->isIncompleteType())
    return false;

  // If we have a class template specialization or a class member of a
  // class template specialization, or an array with known size of such,
  // try to instantiate it.
  QualType MaybeTemplate = T;
  if (const ConstantArrayType *Array = Context.getAsConstantArrayType(T))
    MaybeTemplate = Array->getElementType();
  if (const RecordType *Record = MaybeTemplate->getAs<RecordType>()) {
    if (ClassTemplateSpecializationDecl *ClassTemplateSpec
          = dyn_cast<ClassTemplateSpecializationDecl>(Record->getDecl())) {
      if (ClassTemplateSpec->getSpecializationKind() == TSK_Undeclared)
        return InstantiateClassTemplateSpecialization(Loc, ClassTemplateSpec,
                                                      TSK_ImplicitInstantiation,
                                                      /*Complain=*/diag != 0);
    } else if (CXXRecordDecl *Rec
                 = dyn_cast<CXXRecordDecl>(Record->getDecl())) {
      if (CXXRecordDecl *Pattern = Rec->getInstantiatedFromMemberClass()) {
        MemberSpecializationInfo *MSInfo = Rec->getMemberSpecializationInfo();
        assert(MSInfo && "Missing member specialization information?");
        // This record was instantiated from a class within a template.
        if (MSInfo->getTemplateSpecializationKind() 
                                               != TSK_ExplicitSpecialization)
          return InstantiateClass(Loc, Rec, Pattern,
                                  getTemplateInstantiationArgs(Rec),
                                  TSK_ImplicitInstantiation,
                                  /*Complain=*/diag != 0);
      }
    }
  }

  if (diag == 0)
    return true;

  const TagType *Tag = T->getAs<TagType>();

  // Avoid diagnosing invalid decls as incomplete.
  if (Tag && Tag->getDecl()->isInvalidDecl())
    return true;

  // Give the external AST source a chance to complete the type.
  if (Tag && Tag->getDecl()->hasExternalLexicalStorage()) {
    Context.getExternalSource()->CompleteType(Tag->getDecl());
    if (!Tag->isIncompleteType())
      return false;
  }

  // We have an incomplete type. Produce a diagnostic.
  Diag(Loc, PD) << T;

  // If we have a note, produce it.
  if (!Note.first.isInvalid())
    Diag(Note.first, Note.second);
    
  // If the type was a forward declaration of a class/struct/union
  // type, produce a note.
  if (Tag && !Tag->getDecl()->isInvalidDecl())
    Diag(Tag->getDecl()->getLocation(),
         Tag->isBeingDefined() ? diag::note_type_being_defined
                               : diag::note_forward_declaration)
        << QualType(Tag, 0);

  return true;
}

bool Sema::RequireCompleteType(SourceLocation Loc, QualType T,
                               const PartialDiagnostic &PD) {
  return RequireCompleteType(Loc, T, PD, 
                             std::make_pair(SourceLocation(), PDiag(0)));
}
  
bool Sema::RequireCompleteType(SourceLocation Loc, QualType T,
                               unsigned DiagID) {
  return RequireCompleteType(Loc, T, PDiag(DiagID),
                             std::make_pair(SourceLocation(), PDiag(0)));
}

/// \brief Retrieve a version of the type 'T' that is elaborated by Keyword
/// and qualified by the nested-name-specifier contained in SS.
QualType Sema::getElaboratedType(ElaboratedTypeKeyword Keyword,
                                 const CXXScopeSpec &SS, QualType T) {
  if (T.isNull())
    return T;
  NestedNameSpecifier *NNS;
  if (SS.isValid())
    NNS = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  else {
    if (Keyword == ETK_None)
      return T;
    NNS = 0;
  }
  return Context.getElaboratedType(Keyword, NNS, T);
}

QualType Sema::BuildTypeofExprType(Expr *E, SourceLocation Loc) {
  ExprResult ER = CheckPlaceholderExpr(E, Loc);
  if (ER.isInvalid()) return QualType();
  E = ER.take();

  if (!E->isTypeDependent()) {
    QualType T = E->getType();
    if (const TagType *TT = T->getAs<TagType>())
      DiagnoseUseOfDecl(TT->getDecl(), E->getExprLoc());
  }
  return Context.getTypeOfExprType(E);
}

QualType Sema::BuildDecltypeType(Expr *E, SourceLocation Loc) {
  ExprResult ER = CheckPlaceholderExpr(E, Loc);
  if (ER.isInvalid()) return QualType();
  E = ER.take();
  
  return Context.getDecltypeType(E);
}
