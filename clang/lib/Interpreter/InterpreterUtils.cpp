//===--- InterpreterUtils.cpp - Incremental Utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some common utils used in the incremental library.
//
//===----------------------------------------------------------------------===//

#include "InterpreterUtils.h"

namespace clang {

IntegerLiteral *IntegerLiteralExpr(ASTContext &C, uint64_t Val) {
  return IntegerLiteral::Create(C, llvm::APSInt::getUnsigned(Val),
                                C.UnsignedLongLongTy, SourceLocation());
}

Expr *CStyleCastPtrExpr(Sema &S, QualType Ty, Expr *E) {
  ASTContext &Ctx = S.getASTContext();
  if (!Ty->isPointerType())
    Ty = Ctx.getPointerType(Ty);

  TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ty, SourceLocation());
  Expr *Result =
      S.BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), E).get();
  assert(Result && "Cannot create CStyleCastPtrExpr");
  return Result;
}

Expr *CStyleCastPtrExpr(Sema &S, QualType Ty, uintptr_t Ptr) {
  ASTContext &Ctx = S.getASTContext();
  return CStyleCastPtrExpr(S, Ty, IntegerLiteralExpr(Ctx, (uint64_t)Ptr));
}

Sema::DeclGroupPtrTy CreateDGPtrFrom(Sema &S, Decl *D) {
  SmallVector<Decl *, 1> DeclsInGroup;
  DeclsInGroup.push_back(D);
  Sema::DeclGroupPtrTy DeclGroupPtr = S.BuildDeclaratorGroup(DeclsInGroup);
  return DeclGroupPtr;
}

NamespaceDecl *LookupNamespace(Sema &S, llvm::StringRef Name,
                               const DeclContext *Within) {
  DeclarationName DName = &S.Context.Idents.get(Name);
  LookupResult R(S, DName, SourceLocation(),
                 Sema::LookupNestedNameSpecifierName);
  R.suppressDiagnostics();
  if (!Within)
    S.LookupName(R, S.TUScope);
  else {
    if (const auto *TD = dyn_cast<clang::TagDecl>(Within);
        TD && !TD->getDefinition())
      // No definition, no lookup result.
      return nullptr;

    S.LookupQualifiedName(R, const_cast<DeclContext *>(Within));
  }

  if (R.empty())
    return nullptr;

  R.resolveKind();

  return dyn_cast<NamespaceDecl>(R.getFoundDecl());
}

NamedDecl *LookupNamed(Sema &S, llvm::StringRef Name,
                       const DeclContext *Within) {
  DeclarationName DName = &S.Context.Idents.get(Name);
  LookupResult R(S, DName, SourceLocation(), Sema::LookupOrdinaryName,
                 RedeclarationKind::ForVisibleRedeclaration);

  R.suppressDiagnostics();

  if (!Within)
    S.LookupName(R, S.TUScope);
  else {
    const DeclContext *PrimaryWithin = nullptr;
    if (const auto *TD = dyn_cast<TagDecl>(Within))
      PrimaryWithin = dyn_cast_if_present<DeclContext>(TD->getDefinition());
    else
      PrimaryWithin = Within->getPrimaryContext();

    // No definition, no lookup result.
    if (!PrimaryWithin)
      return nullptr;

    S.LookupQualifiedName(R, const_cast<DeclContext *>(PrimaryWithin));
  }

  if (R.empty())
    return nullptr;
  R.resolveKind();

  if (R.isSingleResult())
    return dyn_cast<NamedDecl>(R.getFoundDecl());

  return nullptr;
}

static NestedNameSpecifier *CreateOuterNNS(const ASTContext &Ctx, const Decl *D,
                                           bool FullyQualify) {
  const DeclContext *DC = D->getDeclContext();
  if (const auto *NS = dyn_cast<NamespaceDecl>(DC)) {
    while (NS && NS->isInline()) {
      // Ignore inline namespace;
      NS = dyn_cast_if_present<NamespaceDecl>(NS->getDeclContext());
    }
    if (NS && NS->getDeclName())
      return CreateNestedNameSpecifier(Ctx, NS);
    return nullptr; // no starting '::', no anonymous
  }
  if (const auto *TD = dyn_cast<TagDecl>(DC))
    return CreateNestedNameSpecifier(Ctx, TD, FullyQualify);
  if (const auto *TDD = dyn_cast<TypedefNameDecl>(DC))
    return CreateNestedNameSpecifier(Ctx, TDD, FullyQualify);
  return nullptr; // no starting '::'
}

static NestedNameSpecifier *
CreateNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Decl *D,
                                    bool FullyQualified) {
  // Create a nested name specifier for the declaring context of the type.

  assert(D);

  const auto *Outer = dyn_cast_if_present<NamedDecl>(D->getDeclContext());
  const auto *OuterNs = dyn_cast_if_present<NamespaceDecl>(D->getDeclContext());
  if (Outer && !(OuterNs && OuterNs->isAnonymousNamespace())) {

    if (const auto *CXXD = dyn_cast<CXXRecordDecl>(D->getDeclContext())) {

      if (ClassTemplateDecl *CTD = CXXD->getDescribedClassTemplate()) {
        // We are in the case of a type(def) that was declared in a
        // class template but is *not* type dependent.  In clang, it gets
        // attached to the class template declaration rather than any
        // specific class template instantiation.   This result in 'odd'
        // fully qualified typename:
        //    vector<_Tp,_Alloc>::size_type
        // Make the situation is 'useable' but looking a bit odd by
        // picking a random instance as the declaring context.
        // FIXME: We should not use the iterators here to check if we are in
        // a template specialization. clTempl != cxxdecl already tell us that
        // is the case. It seems that we rely on a side-effect from triggering
        // deserializations to support 'some' use-case. See ROOT-9709.
        if (CTD->spec_begin() != CTD->spec_end()) {
          D = *(CTD->spec_begin());
          Outer = dyn_cast<NamedDecl>(D);
          OuterNs = dyn_cast<NamespaceDecl>(D);
        }
      }
    }

    if (OuterNs)
      return CreateNestedNameSpecifier(Ctx, OuterNs);

    if (const auto *TD = dyn_cast<TagDecl>(Outer))
      return CreateNestedNameSpecifier(Ctx, TD, FullyQualified);
  }
  return nullptr;
}

static NestedNameSpecifier *
CreateNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Type *TypePtr,
                                    bool FullyQualified) {
  // Create a nested name specifier for the declaring context of the type.

  if (!TypePtr)
    return nullptr;

  Decl *D = nullptr;
  if (const auto *TDT = dyn_cast<TypedefType>(TypePtr)) {
    D = TDT->getDecl();
  } else {
    // There are probably other cases ...
    if (const auto *TT = dyn_cast_if_present<TagType>(TypePtr))
      D = TT->getDecl();
    else
      D = TypePtr->getAsCXXRecordDecl();
  }

  if (!D)
    return nullptr;

  return CreateNestedNameSpecifierForScopeOf(Ctx, D, FullyQualified);
}

static NestedNameSpecifier *
GetFullyQualifiedNameSpecifier(const ASTContext &Ctx,
                               NestedNameSpecifier *Scope) {
  // Return a fully qualified version of this name specifier
  if (Scope->getKind() == NestedNameSpecifier::Global) {
    // Already fully qualified.
    return Scope;
  }

  if (const Type *Type = Scope->getAsType()) {
    // Find decl context.
    const TagDecl *TD = nullptr;
    if (const auto *TT = dyn_cast<TagType>(Type))
      TD = TT->getDecl();
    else
      TD = Type->getAsCXXRecordDecl();

    if (TD)
      return CreateNestedNameSpecifier(Ctx, TD, true /*FullyQualified*/);

    if (const auto *TDD = dyn_cast<TypedefType>(Type))
      return CreateNestedNameSpecifier(Ctx, TDD->getDecl(),
                                       true /*FullyQualified*/);
  } else if (const NamespaceDecl *NS = Scope->getAsNamespace())
    return CreateNestedNameSpecifier(Ctx, NS);
  else if (const auto *Alias = Scope->getAsNamespaceAlias())
    return CreateNestedNameSpecifier(Ctx,
                                     Alias->getNamespace()->getCanonicalDecl());

  return Scope;
}

static bool GetFullyQualifiedTemplateName(const ASTContext &Ctx,
                                          TemplateName &Name) {

  bool Changed = false;
  NestedNameSpecifier *NNS = nullptr;

  TemplateDecl *TD = Name.getAsTemplateDecl();
  QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName();

  if (QTN && !QTN->hasTemplateKeyword()) {
    NNS = QTN->getQualifier();
    NestedNameSpecifier *QNNS = GetFullyQualifiedNameSpecifier(Ctx, NNS);
    if (QNNS != NNS) {
      Changed = true;
      NNS = QNNS;
    } else {
      NNS = nullptr;
    }
  } else {
    NNS = CreateNestedNameSpecifierForScopeOf(Ctx, TD, true);
  }
  if (NNS) {
    Name = Ctx.getQualifiedTemplateName(NNS,
                                        /*TemplateKeyword=*/false,
                                        TemplateName(TD));
    Changed = true;
  }
  return Changed;
}

static bool GetFullyQualifiedTemplateArgument(const ASTContext &Ctx,
                                              TemplateArgument &Arg) {
  bool Changed = false;

  // Note: we do not handle TemplateArgument::Expression, to replace it
  // we need the information for the template instance decl.
  // See GetPartiallyDesugaredTypeImpl

  if (Arg.getKind() == TemplateArgument::Template) {
    TemplateName Name = Arg.getAsTemplate();
    Changed = GetFullyQualifiedTemplateName(Ctx, Name);
    if (Changed) {
      Arg = TemplateArgument(Name);
    }
  } else if (Arg.getKind() == TemplateArgument::Type) {
    QualType SubTy = Arg.getAsType();
    // Check if the type needs more desugaring and recurse.
    QualType QTFQ = GetFullyQualifiedType(SubTy, Ctx);
    if (QTFQ != SubTy) {
      Arg = TemplateArgument(QTFQ);
      Changed = true;
    }
  } else if (Arg.getKind() == TemplateArgument::Pack) {
    SmallVector<TemplateArgument, 2> desArgs;
    for (auto I = Arg.pack_begin(), E = Arg.pack_end(); I != E; ++I) {
      TemplateArgument PackArg(*I);
      Changed = GetFullyQualifiedTemplateArgument(Ctx, PackArg);
      desArgs.push_back(PackArg);
    }
    if (Changed) {
      // The allocator in ASTContext is mutable ...
      // Keep the argument const to be inline will all the other interfaces
      // like:  NestedNameSpecifier::Create
      ASTContext &MutableCtx(const_cast<ASTContext &>(Ctx));
      Arg = TemplateArgument::CreatePackCopy(MutableCtx, desArgs);
    }
  }
  return Changed;
}

static const Type *GetFullyQualifiedLocalType(const ASTContext &Ctx,
                                              const Type *Typeptr) {
  // We really just want to handle the template parameter if any ....
  // In case of template specializations iterate over the arguments and
  // fully qualify them as well.
  if (const auto *TST = dyn_cast<const TemplateSpecializationType>(Typeptr)) {

    bool MightHaveChanged = false;
    llvm::SmallVector<TemplateArgument, 4> DesArgs;
    for (auto &I : TST->template_arguments()) {

      // cheap to copy and potentially modified by
      // GetFullyQualifedTemplateArgument
      TemplateArgument Arg(I);
      MightHaveChanged |= GetFullyQualifiedTemplateArgument(Ctx, Arg);
      DesArgs.push_back(Arg);
    }

    // If desugaring happened allocate new type in the AST.
    if (MightHaveChanged) {
      QualType QT = Ctx.getTemplateSpecializationType(
          TST->getTemplateName(), DesArgs, TST->getCanonicalTypeInternal());
      return QT.getTypePtr();
    }
  } else if (const auto *TSTRecord = dyn_cast<const RecordType>(Typeptr)) {
    // We are asked to fully qualify and we have a Record Type,
    // which can point to a template instantiation with no sugar in any of
    // its template argument, however we still need to fully qualify them.

    if (const auto *TSTdecl =
            dyn_cast<ClassTemplateSpecializationDecl>(TSTRecord->getDecl())) {
      const TemplateArgumentList &TemplateArgs = TSTdecl->getTemplateArgs();

      bool MightHaveChanged = false;
      llvm::SmallVector<TemplateArgument, 4> DesArgs;
      for (unsigned int I = 0, E = TemplateArgs.size(); I != E; ++I) {

        // cheap to copy and potentially modified by
        // GetFullyQualifedTemplateArgument
        TemplateArgument Arg(TemplateArgs[I]);
        MightHaveChanged |= GetFullyQualifiedTemplateArgument(Ctx, Arg);
        DesArgs.push_back(Arg);
      }

      // If desugaring happened allocate new type in the AST.
      if (MightHaveChanged) {
        TemplateName TN(TSTdecl->getSpecializedTemplate());
        QualType QT = Ctx.getTemplateSpecializationType(
            TN, DesArgs, TSTRecord->getCanonicalTypeInternal());
        return QT.getTypePtr();
      }
    }
  }
  return Typeptr;
}

NestedNameSpecifier *CreateNestedNameSpecifier(const ASTContext &Ctx,
                                               const NamespaceDecl *NSD) {
  while (NSD && NSD->isInline()) {
    // Ignore inline namespace;
    NSD = dyn_cast_if_present<NamespaceDecl>(NSD->getDeclContext());
  }
  if (!NSD)
    return nullptr;

  bool FullyQualified = true; // doesn't matter, DeclContexts are namespaces
  return NestedNameSpecifier::Create(
      Ctx, CreateOuterNNS(Ctx, NSD, FullyQualified), NSD);
}

NestedNameSpecifier *CreateNestedNameSpecifier(const ASTContext &Ctx,
                                               const TypedefNameDecl *TD,
                                               bool FullyQualify) {
  return NestedNameSpecifier::Create(Ctx, CreateOuterNNS(Ctx, TD, FullyQualify),
                                     /*Template=*/true, TD->getTypeForDecl());
}

NestedNameSpecifier *CreateNestedNameSpecifier(const ASTContext &Ctx,
                                               const TagDecl *TD,
                                               bool FullyQualify) {
  const Type *Ty = Ctx.getTypeDeclType(TD).getTypePtr();
  if (FullyQualify)
    Ty = GetFullyQualifiedLocalType(Ctx, Ty);
  return NestedNameSpecifier::Create(Ctx, CreateOuterNNS(Ctx, TD, FullyQualify),
                                     /*Template=*/false, Ty);
}

QualType GetFullyQualifiedType(QualType QT, const ASTContext &Ctx) {
  // Return the fully qualified type, if we need to recurse through any
  // template parameter, this needs to be merged somehow with
  // GetPartialDesugaredType.

  // In case of myType* we need to strip the pointer first, fully qualifiy
  // and attach the pointer once again.
  if (isa<PointerType>(QT.getTypePtr())) {
    // Get the qualifiers.
    Qualifiers Quals = QT.getQualifiers();
    QT = GetFullyQualifiedType(QT->getPointeeType(), Ctx);
    QT = Ctx.getPointerType(QT);
    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
    return QT;
  }

  // In case of myType& we need to strip the pointer first, fully qualifiy
  // and attach the pointer once again.
  if (isa<ReferenceType>(QT.getTypePtr())) {
    // Get the qualifiers.
    bool IsLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
    Qualifiers Quals = QT.getQualifiers();
    QT = GetFullyQualifiedType(QT->getPointeeType(), Ctx);
    // Add the r- or l-value reference type back to the desugared one.
    if (IsLValueRefTy)
      QT = Ctx.getLValueReferenceType(QT);
    else
      QT = Ctx.getRValueReferenceType(QT);
    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
    return QT;
  }

  // Strip deduced types.
  if (const auto *AutoTy = dyn_cast<AutoType>(QT.getTypePtr())) {
    if (!AutoTy->getDeducedType().isNull())
      return GetFullyQualifiedType(
          AutoTy->getDeducedType().getDesugaredType(Ctx), Ctx);
  }

  // Remove the part of the type related to the type being a template
  // parameter (we won't report it as part of the 'type name' and it is
  // actually make the code below to be more complex (to handle those)
  while (isa<SubstTemplateTypeParmType>(QT.getTypePtr())) {
    // Get the qualifiers.
    Qualifiers Quals = QT.getQualifiers();

    QT = cast<SubstTemplateTypeParmType>(QT.getTypePtr())->desugar();

    // Add back the qualifiers.
    QT = Ctx.getQualifiedType(QT, Quals);
  }

  NestedNameSpecifier *Prefix = nullptr;
  Qualifiers PrefixQualifiers;
  if (const auto *EType = dyn_cast<ElaboratedType>(QT.getTypePtr())) {
    // Intentionally, we do not care about the other compononent of
    // the elaborated type (the keyword) as part of the partial
    // desugaring (and/or name normalization) is to remove it.
    Prefix = EType->getQualifier();
    if (Prefix) {
      const NamespaceDecl *NS = Prefix->getAsNamespace();
      if (Prefix != NestedNameSpecifier::GlobalSpecifier(Ctx) &&
          !(NS && NS->isAnonymousNamespace())) {
        PrefixQualifiers = QT.getLocalQualifiers();
        Prefix = GetFullyQualifiedNameSpecifier(Ctx, Prefix);
        QT = QualType(EType->getNamedType().getTypePtr(), 0);
      } else {
        Prefix = nullptr;
      }
    }
  } else {

    // Create a nested name specifier if needed (i.e. if the decl context
    // is not the global scope.
    Prefix = CreateNestedNameSpecifierForScopeOf(Ctx, QT.getTypePtr(),
                                                 true /*FullyQualified*/);

    // move the qualifiers on the outer type (avoid 'std::const string'!)
    if (Prefix) {
      PrefixQualifiers = QT.getLocalQualifiers();
      QT = QualType(QT.getTypePtr(), 0);
    }
  }

  // In case of template specializations iterate over the arguments and
  // fully qualify them as well.
  if (isa<const TemplateSpecializationType>(QT.getTypePtr())) {

    Qualifiers Qualifiers = QT.getLocalQualifiers();
    const Type *TypePtr = GetFullyQualifiedLocalType(Ctx, QT.getTypePtr());
    QT = Ctx.getQualifiedType(TypePtr, Qualifiers);

  } else if (isa<const RecordType>(QT.getTypePtr())) {
    // We are asked to fully qualify and we have a Record Type,
    // which can point to a template instantiation with no sugar in any of
    // its template argument, however we still need to fully qualify them.

    Qualifiers Qualifiers = QT.getLocalQualifiers();
    const Type *TypePtr = GetFullyQualifiedLocalType(Ctx, QT.getTypePtr());
    QT = Ctx.getQualifiedType(TypePtr, Qualifiers);
  }
  if (Prefix) {
    // We intentionally always use ElaboratedTypeKeyword::None, we never want
    // the keyword (humm ... what about anonymous types?)
    QT = Ctx.getElaboratedType(ElaboratedTypeKeyword::None, Prefix, QT);
    QT = Ctx.getQualifiedType(QT, PrefixQualifiers);
  }
  return QT;
}

std::string GetFullTypeName(ASTContext &Ctx, QualType QT) {
  QualType FQT = GetFullyQualifiedType(QT, Ctx);
  PrintingPolicy Policy(Ctx.getPrintingPolicy());
  Policy.SuppressScope = false;
  Policy.AnonymousTagLocations = false;
  return FQT.getAsString(Policy);
}
} // namespace clang
