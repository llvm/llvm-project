//===------- QualTypeNames.cpp - Generate Complete QualType Names ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/QualTypeNames.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"

namespace clang {

namespace TypeName {

/// Create a NestedNameSpecifier for Namesp and its enclosing
/// scopes.
///
/// \param[in] Ctx - the AST Context to be used.
/// \param[in] Namesp - the NamespaceDecl for which a NestedNameSpecifier
/// is requested.
/// \param[in] WithGlobalNsPrefix - Indicate whether the global namespace
/// specifier "::" should be prepended or not.
static NestedNameSpecifier
createNestedNameSpecifier(const ASTContext &Ctx, const NamespaceDecl *Namesp,
                          bool WithGlobalNsPrefix);

/// Create a NestedNameSpecifier for TagDecl and its enclosing
/// scopes.
///
/// \param[in] Ctx - the AST Context to be used.
/// \param[in] TD - the TagDecl for which a NestedNameSpecifier is
/// requested.
/// \param[in] FullyQualify - Convert all template arguments into fully
/// qualified names.
/// \param[in] WithGlobalNsPrefix - Indicate whether the global namespace
/// specifier "::" should be prepended or not.
static NestedNameSpecifier createNestedNameSpecifier(const ASTContext &Ctx,
                                                     const TypeDecl *TD,
                                                     bool FullyQualify,
                                                     bool WithGlobalNsPrefix);

static NestedNameSpecifier
createNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Decl *decl,
                                    bool FullyQualified,
                                    bool WithGlobalNsPrefix);

static NestedNameSpecifier getFullyQualifiedNestedNameSpecifier(
    const ASTContext &Ctx, NestedNameSpecifier NNS, bool WithGlobalNsPrefix);

static bool getFullyQualifiedTemplateName(const ASTContext &Ctx,
                                          TemplateName &TName,
                                          bool WithGlobalNsPrefix) {
  bool Changed = false;
  NestedNameSpecifier NNS = std::nullopt;

  TemplateDecl *ArgTDecl = TName.getAsTemplateDecl();
  if (!ArgTDecl) // ArgTDecl can be null in dependent contexts.
    return false;

  QualifiedTemplateName *QTName = TName.getAsQualifiedTemplateName();

  if (QTName &&
      !QTName->hasTemplateKeyword() &&
      (NNS = QTName->getQualifier())) {
    NestedNameSpecifier QNNS =
        getFullyQualifiedNestedNameSpecifier(Ctx, NNS, WithGlobalNsPrefix);
    if (QNNS != NNS) {
      Changed = true;
      NNS = QNNS;
    } else {
      NNS = std::nullopt;
    }
  } else {
    NNS = createNestedNameSpecifierForScopeOf(
        Ctx, ArgTDecl, true, WithGlobalNsPrefix);
  }
  if (NNS) {
    TemplateName UnderlyingTN(ArgTDecl);
    if (UsingShadowDecl *USD = TName.getAsUsingShadowDecl())
      UnderlyingTN = TemplateName(USD);
    TName =
        Ctx.getQualifiedTemplateName(NNS,
                                     /*TemplateKeyword=*/false, UnderlyingTN);
    Changed = true;
  }
  return Changed;
}

static bool getFullyQualifiedTemplateArgument(const ASTContext &Ctx,
                                              TemplateArgument &Arg,
                                              bool WithGlobalNsPrefix) {
  bool Changed = false;

  // Note: we do not handle TemplateArgument::Expression, to replace it
  // we need the information for the template instance decl.

  if (Arg.getKind() == TemplateArgument::Template) {
    TemplateName TName = Arg.getAsTemplate();
    Changed = getFullyQualifiedTemplateName(Ctx, TName, WithGlobalNsPrefix);
    if (Changed) {
      Arg = TemplateArgument(TName);
    }
  } else if (Arg.getKind() == TemplateArgument::Type) {
    QualType SubTy = Arg.getAsType();
    // Check if the type needs more desugaring and recurse.
    QualType QTFQ = getFullyQualifiedType(SubTy, Ctx, WithGlobalNsPrefix);
    if (QTFQ != SubTy) {
      Arg = TemplateArgument(QTFQ);
      Changed = true;
    }
  }
  return Changed;
}

static const Type *getFullyQualifiedTemplateType(const ASTContext &Ctx,
                                                 const TagType *TSTRecord,
                                                 ElaboratedTypeKeyword Keyword,
                                                 NestedNameSpecifier Qualifier,
                                                 bool WithGlobalNsPrefix) {
  // We are asked to fully qualify and we have a Record Type,
  // which can point to a template instantiation with no sugar in any of
  // its template argument, however we still need to fully qualify them.

  const auto *TD = TSTRecord->getDecl();
  const auto *TSTDecl = dyn_cast<ClassTemplateSpecializationDecl>(TD);
  if (!TSTDecl)
    return Ctx.getTagType(Keyword, Qualifier, TD, /*OwnsTag=*/false)
        .getTypePtr();

  const TemplateArgumentList &TemplateArgs = TSTDecl->getTemplateArgs();

  bool MightHaveChanged = false;
  SmallVector<TemplateArgument, 4> FQArgs;
  for (unsigned int I = 0, E = TemplateArgs.size(); I != E; ++I) {
    // cheap to copy and potentially modified by
    // getFullyQualifedTemplateArgument
    TemplateArgument Arg(TemplateArgs[I]);
    MightHaveChanged |=
        getFullyQualifiedTemplateArgument(Ctx, Arg, WithGlobalNsPrefix);
    FQArgs.push_back(Arg);
  }

  if (!MightHaveChanged)
    return Ctx.getTagType(Keyword, Qualifier, TD, /*OwnsTag=*/false)
        .getTypePtr();
  // If a fully qualified arg is different from the unqualified arg,
  // allocate new type in the AST.
  TemplateName TN = Ctx.getQualifiedTemplateName(
      Qualifier, /*TemplateKeyword=*/false,
      TemplateName(TSTDecl->getSpecializedTemplate()));
  QualType QT = Ctx.getTemplateSpecializationType(
      Keyword, TN, FQArgs,
      /*CanonicalArgs=*/{}, TSTRecord->getCanonicalTypeInternal());
  // getTemplateSpecializationType returns a fully qualified
  // version of the specialization itself, so no need to qualify
  // it.
  return QT.getTypePtr();
}

static const Type *
getFullyQualifiedTemplateType(const ASTContext &Ctx,
                              const TemplateSpecializationType *TST,
                              bool WithGlobalNsPrefix) {
  TemplateName TName = TST->getTemplateName();
  bool MightHaveChanged =
      getFullyQualifiedTemplateName(Ctx, TName, WithGlobalNsPrefix);
  SmallVector<TemplateArgument, 4> FQArgs;
  // Cheap to copy and potentially modified by
  // getFullyQualifedTemplateArgument.
  for (TemplateArgument Arg : TST->template_arguments()) {
    MightHaveChanged |=
        getFullyQualifiedTemplateArgument(Ctx, Arg, WithGlobalNsPrefix);
    FQArgs.push_back(Arg);
  }

  if (!MightHaveChanged)
    return TST;

  QualType NewQT =
      Ctx.getTemplateSpecializationType(TST->getKeyword(), TName, FQArgs,
                                        /*CanonicalArgs=*/{}, TST->desugar());
  // getTemplateSpecializationType returns a fully qualified
  // version of the specialization itself, so no need to qualify
  // it.
  return NewQT.getTypePtr();
}

static NestedNameSpecifier createOuterNNS(const ASTContext &Ctx, const Decl *D,
                                          bool FullyQualify,
                                          bool WithGlobalNsPrefix) {
  const DeclContext *DC = D->getDeclContext();
  if (const auto *NS = dyn_cast<NamespaceDecl>(DC)) {
    while (NS && NS->isInline()) {
      // Ignore inline namespace;
      NS = dyn_cast<NamespaceDecl>(NS->getDeclContext());
    }
    if (NS && NS->getDeclName()) {
      return createNestedNameSpecifier(Ctx, NS, WithGlobalNsPrefix);
    }
    return std::nullopt; // no starting '::', no anonymous
  }
  if (const auto *TD = dyn_cast<TagDecl>(DC))
    return createNestedNameSpecifier(Ctx, TD, FullyQualify, WithGlobalNsPrefix);
  if (const auto *TDD = dyn_cast<TypedefNameDecl>(DC))
    return createNestedNameSpecifier(Ctx, TDD, FullyQualify,
                                     WithGlobalNsPrefix);
  if (WithGlobalNsPrefix && DC->isTranslationUnit())
    return NestedNameSpecifier::getGlobal();
  return std::nullopt; // no starting '::' if |WithGlobalNsPrefix| is false
}

/// Return a fully qualified version of this name specifier.
static NestedNameSpecifier getFullyQualifiedNestedNameSpecifier(
    const ASTContext &Ctx, NestedNameSpecifier Scope, bool WithGlobalNsPrefix) {
  switch (Scope.getKind()) {
  case NestedNameSpecifier::Kind::Null:
    llvm_unreachable("can't fully qualify the empty nested name specifier");
  case NestedNameSpecifier::Kind::Global:
  case NestedNameSpecifier::Kind::MicrosoftSuper:
    // Already fully qualified
    return Scope;
  case NestedNameSpecifier::Kind::Namespace:
    return TypeName::createNestedNameSpecifier(
        Ctx, Scope.getAsNamespaceAndPrefix().Namespace->getNamespace(),
        WithGlobalNsPrefix);
  case NestedNameSpecifier::Kind::Type: {
    const Type *Type = Scope.getAsType();
    // Find decl context.
    const TypeDecl *TD;
    if (const TagType *TagDeclType = Type->getAs<TagType>())
      TD = TagDeclType->getDecl();
    else if (const auto *D = dyn_cast<TypedefType>(Type))
      TD = D->getDecl();
    else
      return Scope;
    return TypeName::createNestedNameSpecifier(Ctx, TD, /*FullyQualify=*/true,
                                               WithGlobalNsPrefix);
  }
  }
  llvm_unreachable("bad NNS kind");
}

/// Create a nested name specifier for the declaring context of
/// the type.
static NestedNameSpecifier
createNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Decl *Decl,
                                    bool FullyQualified,
                                    bool WithGlobalNsPrefix) {
  assert(Decl);

  // Some declaration cannot be qualified.
  if (Decl->isTemplateParameter())
    return std::nullopt;
  const DeclContext *DC = Decl->getDeclContext()->getRedeclContext();
  const auto *Outer = dyn_cast<NamedDecl>(DC);
  const auto *OuterNS = dyn_cast<NamespaceDecl>(DC);
  if (OuterNS && OuterNS->isAnonymousNamespace())
    OuterNS = dyn_cast<NamespaceDecl>(OuterNS->getParent());
  if (Outer) {
    if (const auto *CxxDecl = dyn_cast<CXXRecordDecl>(DC)) {
      if (ClassTemplateDecl *ClassTempl =
              CxxDecl->getDescribedClassTemplate()) {
        // We are in the case of a type(def) that was declared in a
        // class template but is *not* type dependent.  In clang, it
        // gets attached to the class template declaration rather than
        // any specific class template instantiation.  This result in
        // 'odd' fully qualified typename:
        //
        //    vector<_Tp,_Alloc>::size_type
        //
        // Make the situation is 'useable' but looking a bit odd by
        // picking a random instance as the declaring context.
        if (!ClassTempl->specializations().empty()) {
          Decl = *(ClassTempl->spec_begin());
          Outer = dyn_cast<NamedDecl>(Decl);
          OuterNS = dyn_cast<NamespaceDecl>(Decl);
        }
      }
    }

    if (OuterNS) {
      return createNestedNameSpecifier(Ctx, OuterNS, WithGlobalNsPrefix);
    } else if (const auto *TD = dyn_cast<TagDecl>(Outer)) {
      return createNestedNameSpecifier(
          Ctx, TD, FullyQualified, WithGlobalNsPrefix);
    } else if (isa<TranslationUnitDecl>(Outer)) {
      // Context is the TU. Nothing needs to be done.
      return std::nullopt;
    } else {
      // Decl's context was neither the TU, a namespace, nor a
      // TagDecl, which means it is a type local to a scope, and not
      // accessible at the end of the TU.
      return std::nullopt;
    }
  } else if (WithGlobalNsPrefix && DC->isTranslationUnit()) {
    return NestedNameSpecifier::getGlobal();
  }
  return std::nullopt;
}

/// Create a nested name specifier for the declaring context of
/// the type.
static NestedNameSpecifier
createNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Type *TypePtr,
                                    bool FullyQualified,
                                    bool WithGlobalNsPrefix) {
  if (!TypePtr)
    return std::nullopt;

  Decl *Decl = nullptr;
  // There are probably other cases ...
  if (const auto *TDT = dyn_cast<TypedefType>(TypePtr)) {
    Decl = TDT->getDecl();
  } else if (const auto *TagDeclType = dyn_cast<TagType>(TypePtr)) {
    Decl = TagDeclType->getDecl();
  } else if (const auto *TST = dyn_cast<TemplateSpecializationType>(TypePtr)) {
    Decl = TST->getTemplateName().getAsTemplateDecl();
  } else {
    Decl = TypePtr->getAsCXXRecordDecl();
  }

  if (!Decl)
    return std::nullopt;

  return createNestedNameSpecifierForScopeOf(
      Ctx, Decl, FullyQualified, WithGlobalNsPrefix);
}

static NestedNameSpecifier
createNestedNameSpecifier(const ASTContext &Ctx, const NamespaceDecl *Namespace,
                          bool WithGlobalNsPrefix) {
  while (Namespace && Namespace->isInline()) {
    // Ignore inline namespace;
    Namespace = dyn_cast<NamespaceDecl>(Namespace->getDeclContext());
  }
  if (!Namespace)
    return std::nullopt;

  bool FullyQualify = true; // doesn't matter, DeclContexts are namespaces
  return NestedNameSpecifier(
      Ctx, Namespace,
      createOuterNNS(Ctx, Namespace, FullyQualify, WithGlobalNsPrefix));
}

NestedNameSpecifier createNestedNameSpecifier(const ASTContext &Ctx,
                                              const TypeDecl *TD,
                                              bool FullyQualify,
                                              bool WithGlobalNsPrefix) {
  const Type *TypePtr = Ctx.getTypeDeclType(TD).getTypePtr();
  if (auto *RD = dyn_cast<TagType>(TypePtr)) {
    // We are asked to fully qualify and we have a Record Type (which
    // may point to a template specialization) or Template
    // Specialization Type. We need to fully qualify their arguments.
    TypePtr = getFullyQualifiedTemplateType(
        Ctx, RD, ElaboratedTypeKeyword::None,
        createOuterNNS(Ctx, TD, FullyQualify, WithGlobalNsPrefix),
        WithGlobalNsPrefix);
  } else if (auto *TST = dyn_cast<TemplateSpecializationType>(TypePtr)) {
    TypePtr = getFullyQualifiedTemplateType(Ctx, TST, WithGlobalNsPrefix);
  }
  return NestedNameSpecifier(TypePtr);
}

/// Return the fully qualified type, including fully-qualified
/// versions of any template parameters.
QualType getFullyQualifiedType(QualType QT, const ASTContext &Ctx,
                               bool WithGlobalNsPrefix) {
  // In case of myType* we need to strip the pointer first, fully
  // qualify and attach the pointer once again.
  if (isa<PointerType>(QT.getTypePtr())) {
    // Get the qualifiers.
    Qualifiers Quals = QT.getQualifiers();
    QT = getFullyQualifiedType(QT->getPointeeType(), Ctx, WithGlobalNsPrefix);
    QT = Ctx.getPointerType(QT);
    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
    return QT;
  }

  if (auto *MPT = dyn_cast<MemberPointerType>(QT.getTypePtr())) {
    // Get the qualifiers.
    Qualifiers Quals = QT.getQualifiers();
    // Fully qualify the pointee and class types.
    QT = getFullyQualifiedType(QT->getPointeeType(), Ctx, WithGlobalNsPrefix);
    NestedNameSpecifier Qualifier = getFullyQualifiedNestedNameSpecifier(
        Ctx, MPT->getQualifier(), WithGlobalNsPrefix);
    QT = Ctx.getMemberPointerType(QT, Qualifier,
                                  MPT->getMostRecentCXXRecordDecl());
    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
    return QT;
  }

  // In case of myType& we need to strip the reference first, fully
  // qualify and attach the reference once again.
  if (isa<ReferenceType>(QT.getTypePtr())) {
    // Get the qualifiers.
    bool IsLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
    Qualifiers Quals = QT.getQualifiers();
    QT = getFullyQualifiedType(QT->getPointeeType(), Ctx, WithGlobalNsPrefix);
    // Add the r- or l-value reference type back to the fully
    // qualified one.
    if (IsLValueRefTy)
      QT = Ctx.getLValueReferenceType(QT);
    else
      QT = Ctx.getRValueReferenceType(QT);
    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
    return QT;
  }

  // Handle types with attributes such as `unique_ptr<int> _Nonnull`.
  if (auto *AT = dyn_cast<AttributedType>(QT.getTypePtr())) {
    QualType NewModified =
        getFullyQualifiedType(AT->getModifiedType(), Ctx, WithGlobalNsPrefix);
    QualType NewEquivalent =
        getFullyQualifiedType(AT->getEquivalentType(), Ctx, WithGlobalNsPrefix);
    Qualifiers Qualifiers = QT.getLocalQualifiers();
    return Ctx.getQualifiedType(
        Ctx.getAttributedType(AT->getAttrKind(), NewModified, NewEquivalent),
        Qualifiers);
  }

  // Remove the part of the type related to the type being a template
  // parameter (we won't report it as part of the 'type name' and it
  // is actually make the code below to be more complex (to handle
  // those)
  while (isa<SubstTemplateTypeParmType>(QT.getTypePtr())) {
    // Get the qualifiers.
    Qualifiers Quals = QT.getQualifiers();

    QT = cast<SubstTemplateTypeParmType>(QT.getTypePtr())->desugar();

    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
  }

  if (const auto *TST =
          dyn_cast<const TemplateSpecializationType>(QT.getTypePtr())) {

    const Type *T = getFullyQualifiedTemplateType(Ctx, TST, WithGlobalNsPrefix);
    if (T == TST)
      return QT;
    return Ctx.getQualifiedType(T, QT.getQualifiers());
  }

  // Local qualifiers are attached to the QualType outside of the
  // elaborated type.  Retrieve them before descending into the
  // elaborated type.
  Qualifiers PrefixQualifiers = QT.getLocalQualifiers();
  QT = QualType(QT.getTypePtr(), 0);

  // We don't consider the alias introduced by `using a::X` as a new type.
  // The qualified name is still a::X.
  if (const auto *UT = QT->getAs<UsingType>()) {
    QT = Ctx.getQualifiedType(UT->desugar(), PrefixQualifiers);
    return getFullyQualifiedType(QT, Ctx, WithGlobalNsPrefix);
  }

  // Create a nested name specifier if needed.
  NestedNameSpecifier Prefix = createNestedNameSpecifierForScopeOf(
      Ctx, QT.getTypePtr(), true /*FullyQualified*/, WithGlobalNsPrefix);

  // In case of template specializations iterate over the arguments and
  // fully qualify them as well.
  if (const auto *TT = dyn_cast<TagType>(QT.getTypePtr())) {
    // We are asked to fully qualify and we have a Record Type (which
    // may point to a template specialization) or Template
    // Specialization Type. We need to fully qualify their arguments.

    const Type *TypePtr = getFullyQualifiedTemplateType(
        Ctx, TT, TT->getKeyword(), Prefix, WithGlobalNsPrefix);
    QT = QualType(TypePtr, 0);
  } else if (const auto *TT = dyn_cast<TypedefType>(QT.getTypePtr())) {
    QT = Ctx.getTypedefType(
        TT->getKeyword(), Prefix, TT->getDecl(),
        getFullyQualifiedType(TT->desugar(), Ctx, WithGlobalNsPrefix));
  } else {
    assert(!Prefix && "Unhandled type node");
  }
  QT = Ctx.getQualifiedType(QT, PrefixQualifiers);
  return QT;
}

std::string getFullyQualifiedName(QualType QT,
                                  const ASTContext &Ctx,
                                  const PrintingPolicy &Policy,
                                  bool WithGlobalNsPrefix) {
  QualType FQQT = getFullyQualifiedType(QT, Ctx, WithGlobalNsPrefix);
  return FQQT.getAsString(Policy);
}

NestedNameSpecifier getFullyQualifiedDeclaredContext(const ASTContext &Ctx,
                                                     const Decl *Decl,
                                                     bool WithGlobalNsPrefix) {
  return createNestedNameSpecifierForScopeOf(Ctx, Decl, /*FullyQualified=*/true,
                                             WithGlobalNsPrefix);
}

}  // end namespace TypeName
}  // end namespace clang
