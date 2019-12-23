//===- ASTImporter.cpp - Importing ASTs from other Contexts ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTImporter class which imports AST nodes from one
//  context into another context.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTImporterSharedState.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/LambdaCapture.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace clang {

  using llvm::make_error;
  using llvm::Error;
  using llvm::Expected;
  using ExpectedType = llvm::Expected<QualType>;
  using ExpectedStmt = llvm::Expected<Stmt *>;
  using ExpectedExpr = llvm::Expected<Expr *>;
  using ExpectedDecl = llvm::Expected<Decl *>;
  using ExpectedSLoc = llvm::Expected<SourceLocation>;
  using ExpectedName = llvm::Expected<DeclarationName>;

  std::string ImportError::toString() const {
    // FIXME: Improve error texts.
    switch (Error) {
    case NameConflict:
      return "NameConflict";
    case UnsupportedConstruct:
      return "UnsupportedConstruct";
    case Unknown:
      return "Unknown error";
    }
    llvm_unreachable("Invalid error code.");
    return "Invalid error code.";
  }

  void ImportError::log(raw_ostream &OS) const {
    OS << toString();
  }

  std::error_code ImportError::convertToErrorCode() const {
    llvm_unreachable("Function not implemented.");
  }

  char ImportError::ID;

  template <class T>
  SmallVector<Decl *, 2>
  getCanonicalForwardRedeclChain(Redeclarable<T>* D) {
    SmallVector<Decl *, 2> Redecls;
    for (auto *R : D->getFirstDecl()->redecls()) {
      if (R != D->getFirstDecl())
        Redecls.push_back(R);
    }
    Redecls.push_back(D->getFirstDecl());
    std::reverse(Redecls.begin(), Redecls.end());
    return Redecls;
  }

  SmallVector<Decl*, 2> getCanonicalForwardRedeclChain(Decl* D) {
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      return getCanonicalForwardRedeclChain<FunctionDecl>(FD);
    if (auto *VD = dyn_cast<VarDecl>(D))
      return getCanonicalForwardRedeclChain<VarDecl>(VD);
    if (auto *TD = dyn_cast<TagDecl>(D))
      return getCanonicalForwardRedeclChain<TagDecl>(TD);
    llvm_unreachable("Bad declaration kind");
  }

  void updateFlags(const Decl *From, Decl *To) {
    // Check if some flags or attrs are new in 'From' and copy into 'To'.
    // FIXME: Other flags or attrs?
    if (From->isUsed(false) && !To->isUsed(false))
      To->setIsUsed();
  }

  class ASTNodeImporter : public TypeVisitor<ASTNodeImporter, ExpectedType>,
                          public DeclVisitor<ASTNodeImporter, ExpectedDecl>,
                          public StmtVisitor<ASTNodeImporter, ExpectedStmt> {
    ASTImporter &Importer;

    // Use this instead of Importer.importInto .
    template <typename ImportT>
    LLVM_NODISCARD Error importInto(ImportT &To, const ImportT &From) {
      return Importer.importInto(To, From);
    }

    // Use this to import pointers of specific type.
    template <typename ImportT>
    LLVM_NODISCARD Error importInto(ImportT *&To, ImportT *From) {
      auto ToOrErr = Importer.Import(From);
      if (ToOrErr)
        To = cast_or_null<ImportT>(*ToOrErr);
      return ToOrErr.takeError();
    }

    // Call the import function of ASTImporter for a baseclass of type `T` and
    // cast the return value to `T`.
    template <typename T>
    Expected<T *> import(T *From) {
      auto ToOrErr = Importer.Import(From);
      if (!ToOrErr)
        return ToOrErr.takeError();
      return cast_or_null<T>(*ToOrErr);
    }

    template <typename T>
    Expected<T *> import(const T *From) {
      return import(const_cast<T *>(From));
    }

    // Call the import function of ASTImporter for type `T`.
    template <typename T>
    Expected<T> import(const T &From) {
      return Importer.Import(From);
    }

    // Import an Optional<T> by importing the contained T, if any.
    template<typename T>
    Expected<Optional<T>> import(Optional<T> From) {
      if (!From)
        return Optional<T>();
      return import(*From);
    }

    template <class T>
    Expected<std::tuple<T>>
    importSeq(const T &From) {
      Expected<T> ToOrErr = import(From);
      if (!ToOrErr)
        return ToOrErr.takeError();
      return std::make_tuple<T>(std::move(*ToOrErr));
    }

    // Import multiple objects with a single function call.
    // This should work for every type for which a variant of `import` exists.
    // The arguments are processed from left to right and import is stopped on
    // first error.
    template <class THead, class... TTail>
    Expected<std::tuple<THead, TTail...>>
    importSeq(const THead &FromHead, const TTail &...FromTail) {
      Expected<std::tuple<THead>> ToHeadOrErr = importSeq(FromHead);
      if (!ToHeadOrErr)
        return ToHeadOrErr.takeError();
      Expected<std::tuple<TTail...>> ToTailOrErr = importSeq(FromTail...);
      if (!ToTailOrErr)
        return ToTailOrErr.takeError();
      return std::tuple_cat(*ToHeadOrErr, *ToTailOrErr);
    }

// Wrapper for an overload set.
    template <typename ToDeclT> struct CallOverloadedCreateFun {
      template <typename... Args>
      auto operator()(Args &&... args)
          -> decltype(ToDeclT::Create(std::forward<Args>(args)...)) {
        return ToDeclT::Create(std::forward<Args>(args)...);
      }
    };

    // Always use these functions to create a Decl during import. There are
    // certain tasks which must be done after the Decl was created, e.g. we
    // must immediately register that as an imported Decl.  The parameter `ToD`
    // will be set to the newly created Decl or if had been imported before
    // then to the already imported Decl.  Returns a bool value set to true if
    // the `FromD` had been imported before.
    template <typename ToDeclT, typename FromDeclT, typename... Args>
    LLVM_NODISCARD bool GetImportedOrCreateDecl(ToDeclT *&ToD, FromDeclT *FromD,
                                                Args &&... args) {
      // There may be several overloads of ToDeclT::Create. We must make sure
      // to call the one which would be chosen by the arguments, thus we use a
      // wrapper for the overload set.
      CallOverloadedCreateFun<ToDeclT> OC;
      return GetImportedOrCreateSpecialDecl(ToD, OC, FromD,
                                            std::forward<Args>(args)...);
    }
    // Use this overload if a special Type is needed to be created.  E.g if we
    // want to create a `TypeAliasDecl` and assign that to a `TypedefNameDecl`
    // then:
    // TypedefNameDecl *ToTypedef;
    // GetImportedOrCreateDecl<TypeAliasDecl>(ToTypedef, FromD, ...);
    template <typename NewDeclT, typename ToDeclT, typename FromDeclT,
              typename... Args>
    LLVM_NODISCARD bool GetImportedOrCreateDecl(ToDeclT *&ToD, FromDeclT *FromD,
                                                Args &&... args) {
      CallOverloadedCreateFun<NewDeclT> OC;
      return GetImportedOrCreateSpecialDecl(ToD, OC, FromD,
                                            std::forward<Args>(args)...);
    }
    // Use this version if a special create function must be
    // used, e.g. CXXRecordDecl::CreateLambda .
    template <typename ToDeclT, typename CreateFunT, typename FromDeclT,
              typename... Args>
    LLVM_NODISCARD bool
    GetImportedOrCreateSpecialDecl(ToDeclT *&ToD, CreateFunT CreateFun,
                                   FromDeclT *FromD, Args &&... args) {
      if (Importer.getImportDeclErrorIfAny(FromD)) {
        ToD = nullptr;
        return true; // Already imported but with error.
      }
      ToD = cast_or_null<ToDeclT>(Importer.GetAlreadyImportedOrNull(FromD));
      if (ToD)
        return true; // Already imported.
      ToD = CreateFun(std::forward<Args>(args)...);
      // Keep track of imported Decls.
      Importer.RegisterImportedDecl(FromD, ToD);
      InitializeImportedDecl(FromD, ToD);
      return false; // A new Decl is created.
    }

    void InitializeImportedDecl(Decl *FromD, Decl *ToD) {
      ToD->IdentifierNamespace = FromD->IdentifierNamespace;
      if (FromD->hasAttrs())
        for (const Attr *FromAttr : FromD->getAttrs()) {
          // FIXME: Return of the error here is not possible until store of
          // import errors is implemented.
          auto ToAttrOrErr = import(FromAttr);
          if (ToAttrOrErr)
            ToD->addAttr(*ToAttrOrErr);
          else
            llvm::consumeError(ToAttrOrErr.takeError());
        }
      if (FromD->isUsed())
        ToD->setIsUsed();
      if (FromD->isImplicit())
        ToD->setImplicit();
    }

    // Check if we have found an existing definition.  Returns with that
    // definition if yes, otherwise returns null.
    Decl *FindAndMapDefinition(FunctionDecl *D, FunctionDecl *FoundFunction) {
      const FunctionDecl *Definition = nullptr;
      if (D->doesThisDeclarationHaveABody() &&
          FoundFunction->hasBody(Definition))
        return Importer.MapImported(D, const_cast<FunctionDecl *>(Definition));
      return nullptr;
    }

    void addDeclToContexts(Decl *FromD, Decl *ToD) {
      if (Importer.isMinimalImport()) {
        // In minimal import case the decl must be added even if it is not
        // contained in original context, for LLDB compatibility.
        // FIXME: Check if a better solution is possible.
        if (!FromD->getDescribedTemplate() &&
            FromD->getFriendObjectKind() == Decl::FOK_None)
          ToD->getLexicalDeclContext()->addDeclInternal(ToD);
        return;
      }

      DeclContext *FromDC = FromD->getDeclContext();
      DeclContext *FromLexicalDC = FromD->getLexicalDeclContext();
      DeclContext *ToDC = ToD->getDeclContext();
      DeclContext *ToLexicalDC = ToD->getLexicalDeclContext();

      bool Visible = false;
      if (FromDC->containsDeclAndLoad(FromD)) {
        ToDC->addDeclInternal(ToD);
        Visible = true;
      }
      if (ToDC != ToLexicalDC && FromLexicalDC->containsDeclAndLoad(FromD)) {
        ToLexicalDC->addDeclInternal(ToD);
        Visible = true;
      }

      // If the Decl was added to any context, it was made already visible.
      // Otherwise it is still possible that it should be visible.
      if (!Visible) {
        if (auto *FromNamed = dyn_cast<NamedDecl>(FromD)) {
          auto *ToNamed = cast<NamedDecl>(ToD);
          DeclContextLookupResult FromLookup =
              FromDC->lookup(FromNamed->getDeclName());
          for (NamedDecl *ND : FromLookup)
            if (ND == FromNamed) {
              ToDC->makeDeclVisibleInContext(ToNamed);
              break;
            }
        }
      }
    }

  public:
    explicit ASTNodeImporter(ASTImporter &Importer) : Importer(Importer) {}

    using TypeVisitor<ASTNodeImporter, ExpectedType>::Visit;
    using DeclVisitor<ASTNodeImporter, ExpectedDecl>::Visit;
    using StmtVisitor<ASTNodeImporter, ExpectedStmt>::Visit;

    // Importing types
    ExpectedType VisitType(const Type *T);
    ExpectedType VisitAtomicType(const AtomicType *T);
    ExpectedType VisitBuiltinType(const BuiltinType *T);
    ExpectedType VisitDecayedType(const DecayedType *T);
    ExpectedType VisitComplexType(const ComplexType *T);
    ExpectedType VisitPointerType(const PointerType *T);
    ExpectedType VisitBlockPointerType(const BlockPointerType *T);
    ExpectedType VisitLValueReferenceType(const LValueReferenceType *T);
    ExpectedType VisitRValueReferenceType(const RValueReferenceType *T);
    ExpectedType VisitMemberPointerType(const MemberPointerType *T);
    ExpectedType VisitConstantArrayType(const ConstantArrayType *T);
    ExpectedType VisitIncompleteArrayType(const IncompleteArrayType *T);
    ExpectedType VisitVariableArrayType(const VariableArrayType *T);
    ExpectedType VisitDependentSizedArrayType(const DependentSizedArrayType *T);
    // FIXME: DependentSizedExtVectorType
    ExpectedType VisitVectorType(const VectorType *T);
    ExpectedType VisitExtVectorType(const ExtVectorType *T);
    ExpectedType VisitFunctionNoProtoType(const FunctionNoProtoType *T);
    ExpectedType VisitFunctionProtoType(const FunctionProtoType *T);
    ExpectedType VisitUnresolvedUsingType(const UnresolvedUsingType *T);
    ExpectedType VisitParenType(const ParenType *T);
    ExpectedType VisitTypedefType(const TypedefType *T);
    ExpectedType VisitTypeOfExprType(const TypeOfExprType *T);
    // FIXME: DependentTypeOfExprType
    ExpectedType VisitTypeOfType(const TypeOfType *T);
    ExpectedType VisitDecltypeType(const DecltypeType *T);
    ExpectedType VisitUnaryTransformType(const UnaryTransformType *T);
    ExpectedType VisitAutoType(const AutoType *T);
    ExpectedType VisitInjectedClassNameType(const InjectedClassNameType *T);
    // FIXME: DependentDecltypeType
    ExpectedType VisitRecordType(const RecordType *T);
    ExpectedType VisitEnumType(const EnumType *T);
    ExpectedType VisitAttributedType(const AttributedType *T);
    ExpectedType VisitTemplateTypeParmType(const TemplateTypeParmType *T);
    ExpectedType VisitSubstTemplateTypeParmType(
        const SubstTemplateTypeParmType *T);
    ExpectedType VisitTemplateSpecializationType(
        const TemplateSpecializationType *T);
    ExpectedType VisitElaboratedType(const ElaboratedType *T);
    ExpectedType VisitDependentNameType(const DependentNameType *T);
    ExpectedType VisitPackExpansionType(const PackExpansionType *T);
    ExpectedType VisitDependentTemplateSpecializationType(
        const DependentTemplateSpecializationType *T);
    ExpectedType VisitObjCInterfaceType(const ObjCInterfaceType *T);
    ExpectedType VisitObjCObjectType(const ObjCObjectType *T);
    ExpectedType VisitObjCObjectPointerType(const ObjCObjectPointerType *T);

    // Importing declarations
    Error ImportDeclParts(
        NamedDecl *D, DeclContext *&DC, DeclContext *&LexicalDC,
        DeclarationName &Name, NamedDecl *&ToD, SourceLocation &Loc);
    Error ImportDefinitionIfNeeded(Decl *FromD, Decl *ToD = nullptr);
    Error ImportDeclarationNameLoc(
        const DeclarationNameInfo &From, DeclarationNameInfo &To);
    Error ImportDeclContext(DeclContext *FromDC, bool ForceImport = false);
    Error ImportDeclContext(
        Decl *From, DeclContext *&ToDC, DeclContext *&ToLexicalDC);
    Error ImportImplicitMethods(const CXXRecordDecl *From, CXXRecordDecl *To);

    Expected<CXXCastPath> ImportCastPath(CastExpr *E);

    using Designator = DesignatedInitExpr::Designator;

    /// What we should import from the definition.
    enum ImportDefinitionKind {
      /// Import the default subset of the definition, which might be
      /// nothing (if minimal import is set) or might be everything (if minimal
      /// import is not set).
      IDK_Default,
      /// Import everything.
      IDK_Everything,
      /// Import only the bare bones needed to establish a valid
      /// DeclContext.
      IDK_Basic
    };

    bool shouldForceImportDeclContext(ImportDefinitionKind IDK) {
      return IDK == IDK_Everything ||
             (IDK == IDK_Default && !Importer.isMinimalImport());
    }

    Error ImportInitializer(VarDecl *From, VarDecl *To);
    Error ImportDefinition(
        RecordDecl *From, RecordDecl *To,
        ImportDefinitionKind Kind = IDK_Default);
    Error ImportDefinition(
        EnumDecl *From, EnumDecl *To,
        ImportDefinitionKind Kind = IDK_Default);
    Error ImportDefinition(
        ObjCInterfaceDecl *From, ObjCInterfaceDecl *To,
        ImportDefinitionKind Kind = IDK_Default);
    Error ImportDefinition(
        ObjCProtocolDecl *From, ObjCProtocolDecl *To,
        ImportDefinitionKind Kind = IDK_Default);
    Error ImportTemplateArguments(
        const TemplateArgument *FromArgs, unsigned NumFromArgs,
        SmallVectorImpl<TemplateArgument> &ToArgs);
    Expected<TemplateArgument>
    ImportTemplateArgument(const TemplateArgument &From);

    template <typename InContainerTy>
    Error ImportTemplateArgumentListInfo(
        const InContainerTy &Container, TemplateArgumentListInfo &ToTAInfo);

    template<typename InContainerTy>
    Error ImportTemplateArgumentListInfo(
      SourceLocation FromLAngleLoc, SourceLocation FromRAngleLoc,
      const InContainerTy &Container, TemplateArgumentListInfo &Result);

    using TemplateArgsTy = SmallVector<TemplateArgument, 8>;
    using FunctionTemplateAndArgsTy =
        std::tuple<FunctionTemplateDecl *, TemplateArgsTy>;
    Expected<FunctionTemplateAndArgsTy>
    ImportFunctionTemplateWithTemplateArgsFromSpecialization(
        FunctionDecl *FromFD);
    Error ImportTemplateParameterLists(const DeclaratorDecl *FromD,
                                       DeclaratorDecl *ToD);

    Error ImportTemplateInformation(FunctionDecl *FromFD, FunctionDecl *ToFD);

    Error ImportFunctionDeclBody(FunctionDecl *FromFD, FunctionDecl *ToFD);

    Error ImportDefaultArgOfParmVarDecl(const ParmVarDecl *FromParam,
                                        ParmVarDecl *ToParam);

    template <typename T>
    bool hasSameVisibilityContext(T *Found, T *From);

    bool IsStructuralMatch(Decl *From, Decl *To, bool Complain);
    bool IsStructuralMatch(RecordDecl *FromRecord, RecordDecl *ToRecord,
                           bool Complain = true);
    bool IsStructuralMatch(VarDecl *FromVar, VarDecl *ToVar,
                           bool Complain = true);
    bool IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToRecord);
    bool IsStructuralMatch(EnumConstantDecl *FromEC, EnumConstantDecl *ToEC);
    bool IsStructuralMatch(FunctionTemplateDecl *From,
                           FunctionTemplateDecl *To);
    bool IsStructuralMatch(FunctionDecl *From, FunctionDecl *To);
    bool IsStructuralMatch(ClassTemplateDecl *From, ClassTemplateDecl *To);
    bool IsStructuralMatch(VarTemplateDecl *From, VarTemplateDecl *To);
    ExpectedDecl VisitDecl(Decl *D);
    ExpectedDecl VisitImportDecl(ImportDecl *D);
    ExpectedDecl VisitEmptyDecl(EmptyDecl *D);
    ExpectedDecl VisitAccessSpecDecl(AccessSpecDecl *D);
    ExpectedDecl VisitStaticAssertDecl(StaticAssertDecl *D);
    ExpectedDecl VisitTranslationUnitDecl(TranslationUnitDecl *D);
    ExpectedDecl VisitNamespaceDecl(NamespaceDecl *D);
    ExpectedDecl VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    ExpectedDecl VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias);
    ExpectedDecl VisitTypedefDecl(TypedefDecl *D);
    ExpectedDecl VisitTypeAliasDecl(TypeAliasDecl *D);
    ExpectedDecl VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    ExpectedDecl VisitLabelDecl(LabelDecl *D);
    ExpectedDecl VisitEnumDecl(EnumDecl *D);
    ExpectedDecl VisitRecordDecl(RecordDecl *D);
    ExpectedDecl VisitEnumConstantDecl(EnumConstantDecl *D);
    ExpectedDecl VisitFunctionDecl(FunctionDecl *D);
    ExpectedDecl VisitCXXMethodDecl(CXXMethodDecl *D);
    ExpectedDecl VisitCXXConstructorDecl(CXXConstructorDecl *D);
    ExpectedDecl VisitCXXDestructorDecl(CXXDestructorDecl *D);
    ExpectedDecl VisitCXXConversionDecl(CXXConversionDecl *D);
    ExpectedDecl VisitFieldDecl(FieldDecl *D);
    ExpectedDecl VisitIndirectFieldDecl(IndirectFieldDecl *D);
    ExpectedDecl VisitFriendDecl(FriendDecl *D);
    ExpectedDecl VisitObjCIvarDecl(ObjCIvarDecl *D);
    ExpectedDecl VisitVarDecl(VarDecl *D);
    ExpectedDecl VisitImplicitParamDecl(ImplicitParamDecl *D);
    ExpectedDecl VisitParmVarDecl(ParmVarDecl *D);
    ExpectedDecl VisitObjCMethodDecl(ObjCMethodDecl *D);
    ExpectedDecl VisitObjCTypeParamDecl(ObjCTypeParamDecl *D);
    ExpectedDecl VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    ExpectedDecl VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    ExpectedDecl VisitLinkageSpecDecl(LinkageSpecDecl *D);
    ExpectedDecl VisitUsingDecl(UsingDecl *D);
    ExpectedDecl VisitUsingShadowDecl(UsingShadowDecl *D);
    ExpectedDecl VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    ExpectedDecl VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    ExpectedDecl VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    ExpectedDecl VisitBuiltinTemplateDecl(BuiltinTemplateDecl *D);
    ExpectedDecl
    VisitLifetimeExtendedTemporaryDecl(LifetimeExtendedTemporaryDecl *D);

    Expected<ObjCTypeParamList *>
    ImportObjCTypeParamList(ObjCTypeParamList *list);

    ExpectedDecl VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    ExpectedDecl VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    ExpectedDecl VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    ExpectedDecl VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    ExpectedDecl VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    ExpectedDecl VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    ExpectedDecl VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    ExpectedDecl VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    ExpectedDecl VisitClassTemplateDecl(ClassTemplateDecl *D);
    ExpectedDecl VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    ExpectedDecl VisitVarTemplateDecl(VarTemplateDecl *D);
    ExpectedDecl VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl *D);
    ExpectedDecl VisitFunctionTemplateDecl(FunctionTemplateDecl *D);

    // Importing statements
    ExpectedStmt VisitStmt(Stmt *S);
    ExpectedStmt VisitGCCAsmStmt(GCCAsmStmt *S);
    ExpectedStmt VisitDeclStmt(DeclStmt *S);
    ExpectedStmt VisitNullStmt(NullStmt *S);
    ExpectedStmt VisitCompoundStmt(CompoundStmt *S);
    ExpectedStmt VisitCaseStmt(CaseStmt *S);
    ExpectedStmt VisitDefaultStmt(DefaultStmt *S);
    ExpectedStmt VisitLabelStmt(LabelStmt *S);
    ExpectedStmt VisitAttributedStmt(AttributedStmt *S);
    ExpectedStmt VisitIfStmt(IfStmt *S);
    ExpectedStmt VisitSwitchStmt(SwitchStmt *S);
    ExpectedStmt VisitWhileStmt(WhileStmt *S);
    ExpectedStmt VisitDoStmt(DoStmt *S);
    ExpectedStmt VisitForStmt(ForStmt *S);
    ExpectedStmt VisitGotoStmt(GotoStmt *S);
    ExpectedStmt VisitIndirectGotoStmt(IndirectGotoStmt *S);
    ExpectedStmt VisitContinueStmt(ContinueStmt *S);
    ExpectedStmt VisitBreakStmt(BreakStmt *S);
    ExpectedStmt VisitReturnStmt(ReturnStmt *S);
    // FIXME: MSAsmStmt
    // FIXME: SEHExceptStmt
    // FIXME: SEHFinallyStmt
    // FIXME: SEHTryStmt
    // FIXME: SEHLeaveStmt
    // FIXME: CapturedStmt
    ExpectedStmt VisitCXXCatchStmt(CXXCatchStmt *S);
    ExpectedStmt VisitCXXTryStmt(CXXTryStmt *S);
    ExpectedStmt VisitCXXForRangeStmt(CXXForRangeStmt *S);
    // FIXME: MSDependentExistsStmt
    ExpectedStmt VisitObjCForCollectionStmt(ObjCForCollectionStmt *S);
    ExpectedStmt VisitObjCAtCatchStmt(ObjCAtCatchStmt *S);
    ExpectedStmt VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S);
    ExpectedStmt VisitObjCAtTryStmt(ObjCAtTryStmt *S);
    ExpectedStmt VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *S);
    ExpectedStmt VisitObjCAtThrowStmt(ObjCAtThrowStmt *S);
    ExpectedStmt VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *S);

    // Importing expressions
    ExpectedStmt VisitExpr(Expr *E);
    ExpectedStmt VisitVAArgExpr(VAArgExpr *E);
    ExpectedStmt VisitChooseExpr(ChooseExpr *E);
    ExpectedStmt VisitGNUNullExpr(GNUNullExpr *E);
    ExpectedStmt VisitPredefinedExpr(PredefinedExpr *E);
    ExpectedStmt VisitDeclRefExpr(DeclRefExpr *E);
    ExpectedStmt VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    ExpectedStmt VisitDesignatedInitExpr(DesignatedInitExpr *E);
    ExpectedStmt VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E);
    ExpectedStmt VisitIntegerLiteral(IntegerLiteral *E);
    ExpectedStmt VisitFloatingLiteral(FloatingLiteral *E);
    ExpectedStmt VisitImaginaryLiteral(ImaginaryLiteral *E);
    ExpectedStmt VisitCharacterLiteral(CharacterLiteral *E);
    ExpectedStmt VisitStringLiteral(StringLiteral *E);
    ExpectedStmt VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    ExpectedStmt VisitAtomicExpr(AtomicExpr *E);
    ExpectedStmt VisitAddrLabelExpr(AddrLabelExpr *E);
    ExpectedStmt VisitConstantExpr(ConstantExpr *E);
    ExpectedStmt VisitParenExpr(ParenExpr *E);
    ExpectedStmt VisitParenListExpr(ParenListExpr *E);
    ExpectedStmt VisitStmtExpr(StmtExpr *E);
    ExpectedStmt VisitUnaryOperator(UnaryOperator *E);
    ExpectedStmt VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
    ExpectedStmt VisitBinaryOperator(BinaryOperator *E);
    ExpectedStmt VisitConditionalOperator(ConditionalOperator *E);
    ExpectedStmt VisitBinaryConditionalOperator(BinaryConditionalOperator *E);
    ExpectedStmt VisitOpaqueValueExpr(OpaqueValueExpr *E);
    ExpectedStmt VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E);
    ExpectedStmt VisitExpressionTraitExpr(ExpressionTraitExpr *E);
    ExpectedStmt VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    ExpectedStmt VisitCompoundAssignOperator(CompoundAssignOperator *E);
    ExpectedStmt VisitImplicitCastExpr(ImplicitCastExpr *E);
    ExpectedStmt VisitExplicitCastExpr(ExplicitCastExpr *E);
    ExpectedStmt VisitOffsetOfExpr(OffsetOfExpr *OE);
    ExpectedStmt VisitCXXThrowExpr(CXXThrowExpr *E);
    ExpectedStmt VisitCXXNoexceptExpr(CXXNoexceptExpr *E);
    ExpectedStmt VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E);
    ExpectedStmt VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
    ExpectedStmt VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);
    ExpectedStmt VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
    ExpectedStmt VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E);
    ExpectedStmt VisitPackExpansionExpr(PackExpansionExpr *E);
    ExpectedStmt VisitSizeOfPackExpr(SizeOfPackExpr *E);
    ExpectedStmt VisitCXXNewExpr(CXXNewExpr *E);
    ExpectedStmt VisitCXXDeleteExpr(CXXDeleteExpr *E);
    ExpectedStmt VisitCXXConstructExpr(CXXConstructExpr *E);
    ExpectedStmt VisitCXXMemberCallExpr(CXXMemberCallExpr *E);
    ExpectedStmt VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E);
    ExpectedStmt VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E);
    ExpectedStmt VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E);
    ExpectedStmt VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E);
    ExpectedStmt VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E);
    ExpectedStmt VisitExprWithCleanups(ExprWithCleanups *E);
    ExpectedStmt VisitCXXThisExpr(CXXThisExpr *E);
    ExpectedStmt VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E);
    ExpectedStmt VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E);
    ExpectedStmt VisitMemberExpr(MemberExpr *E);
    ExpectedStmt VisitCallExpr(CallExpr *E);
    ExpectedStmt VisitLambdaExpr(LambdaExpr *LE);
    ExpectedStmt VisitInitListExpr(InitListExpr *E);
    ExpectedStmt VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E);
    ExpectedStmt VisitCXXInheritedCtorInitExpr(CXXInheritedCtorInitExpr *E);
    ExpectedStmt VisitArrayInitLoopExpr(ArrayInitLoopExpr *E);
    ExpectedStmt VisitArrayInitIndexExpr(ArrayInitIndexExpr *E);
    ExpectedStmt VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E);
    ExpectedStmt VisitCXXNamedCastExpr(CXXNamedCastExpr *E);
    ExpectedStmt VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E);
    ExpectedStmt VisitTypeTraitExpr(TypeTraitExpr *E);
    ExpectedStmt VisitCXXTypeidExpr(CXXTypeidExpr *E);

    template<typename IIter, typename OIter>
    Error ImportArrayChecked(IIter Ibegin, IIter Iend, OIter Obegin) {
      using ItemT = typename std::remove_reference<decltype(*Obegin)>::type;
      for (; Ibegin != Iend; ++Ibegin, ++Obegin) {
        Expected<ItemT> ToOrErr = import(*Ibegin);
        if (!ToOrErr)
          return ToOrErr.takeError();
        *Obegin = *ToOrErr;
      }
      return Error::success();
    }

    // Import every item from a container structure into an output container.
    // If error occurs, stops at first error and returns the error.
    // The output container should have space for all needed elements (it is not
    // expanded, new items are put into from the beginning).
    template<typename InContainerTy, typename OutContainerTy>
    Error ImportContainerChecked(
        const InContainerTy &InContainer, OutContainerTy &OutContainer) {
      return ImportArrayChecked(
          InContainer.begin(), InContainer.end(), OutContainer.begin());
    }

    template<typename InContainerTy, typename OIter>
    Error ImportArrayChecked(const InContainerTy &InContainer, OIter Obegin) {
      return ImportArrayChecked(InContainer.begin(), InContainer.end(), Obegin);
    }

    Error ImportOverriddenMethods(CXXMethodDecl *ToMethod,
                                  CXXMethodDecl *FromMethod);

    Expected<FunctionDecl *> FindFunctionTemplateSpecialization(
        FunctionDecl *FromFD);

    // Returns true if the given function has a placeholder return type and
    // that type is declared inside the body of the function.
    // E.g. auto f() { struct X{}; return X(); }
    bool hasAutoReturnTypeDeclaredInside(FunctionDecl *D);
  };

template <typename InContainerTy>
Error ASTNodeImporter::ImportTemplateArgumentListInfo(
    SourceLocation FromLAngleLoc, SourceLocation FromRAngleLoc,
    const InContainerTy &Container, TemplateArgumentListInfo &Result) {
  auto ToLAngleLocOrErr = import(FromLAngleLoc);
  if (!ToLAngleLocOrErr)
    return ToLAngleLocOrErr.takeError();
  auto ToRAngleLocOrErr = import(FromRAngleLoc);
  if (!ToRAngleLocOrErr)
    return ToRAngleLocOrErr.takeError();

  TemplateArgumentListInfo ToTAInfo(*ToLAngleLocOrErr, *ToRAngleLocOrErr);
  if (auto Err = ImportTemplateArgumentListInfo(Container, ToTAInfo))
    return Err;
  Result = ToTAInfo;
  return Error::success();
}

template <>
Error ASTNodeImporter::ImportTemplateArgumentListInfo<TemplateArgumentListInfo>(
    const TemplateArgumentListInfo &From, TemplateArgumentListInfo &Result) {
  return ImportTemplateArgumentListInfo(
      From.getLAngleLoc(), From.getRAngleLoc(), From.arguments(), Result);
}

template <>
Error ASTNodeImporter::ImportTemplateArgumentListInfo<
    ASTTemplateArgumentListInfo>(
        const ASTTemplateArgumentListInfo &From,
        TemplateArgumentListInfo &Result) {
  return ImportTemplateArgumentListInfo(
      From.LAngleLoc, From.RAngleLoc, From.arguments(), Result);
}

Expected<ASTNodeImporter::FunctionTemplateAndArgsTy>
ASTNodeImporter::ImportFunctionTemplateWithTemplateArgsFromSpecialization(
    FunctionDecl *FromFD) {
  assert(FromFD->getTemplatedKind() ==
      FunctionDecl::TK_FunctionTemplateSpecialization);

  FunctionTemplateAndArgsTy Result;

  auto *FTSInfo = FromFD->getTemplateSpecializationInfo();
  if (Error Err = importInto(std::get<0>(Result), FTSInfo->getTemplate()))
    return std::move(Err);

  // Import template arguments.
  auto TemplArgs = FTSInfo->TemplateArguments->asArray();
  if (Error Err = ImportTemplateArguments(TemplArgs.data(), TemplArgs.size(),
      std::get<1>(Result)))
    return std::move(Err);

  return Result;
}

template <>
Expected<TemplateParameterList *>
ASTNodeImporter::import(TemplateParameterList *From) {
  SmallVector<NamedDecl *, 4> To(From->size());
  if (Error Err = ImportContainerChecked(*From, To))
    return std::move(Err);

  ExpectedExpr ToRequiresClause = import(From->getRequiresClause());
  if (!ToRequiresClause)
    return ToRequiresClause.takeError();

  auto ToTemplateLocOrErr = import(From->getTemplateLoc());
  if (!ToTemplateLocOrErr)
    return ToTemplateLocOrErr.takeError();
  auto ToLAngleLocOrErr = import(From->getLAngleLoc());
  if (!ToLAngleLocOrErr)
    return ToLAngleLocOrErr.takeError();
  auto ToRAngleLocOrErr = import(From->getRAngleLoc());
  if (!ToRAngleLocOrErr)
    return ToRAngleLocOrErr.takeError();

  return TemplateParameterList::Create(
      Importer.getToContext(),
      *ToTemplateLocOrErr,
      *ToLAngleLocOrErr,
      To,
      *ToRAngleLocOrErr,
      *ToRequiresClause);
}

template <>
Expected<TemplateArgument>
ASTNodeImporter::import(const TemplateArgument &From) {
  switch (From.getKind()) {
  case TemplateArgument::Null:
    return TemplateArgument();

  case TemplateArgument::Type: {
    ExpectedType ToTypeOrErr = import(From.getAsType());
    if (!ToTypeOrErr)
      return ToTypeOrErr.takeError();
    return TemplateArgument(*ToTypeOrErr);
  }

  case TemplateArgument::Integral: {
    ExpectedType ToTypeOrErr = import(From.getIntegralType());
    if (!ToTypeOrErr)
      return ToTypeOrErr.takeError();
    return TemplateArgument(From, *ToTypeOrErr);
  }

  case TemplateArgument::Declaration: {
    Expected<ValueDecl *> ToOrErr = import(From.getAsDecl());
    if (!ToOrErr)
      return ToOrErr.takeError();
    ExpectedType ToTypeOrErr = import(From.getParamTypeForDecl());
    if (!ToTypeOrErr)
      return ToTypeOrErr.takeError();
    return TemplateArgument(*ToOrErr, *ToTypeOrErr);
  }

  case TemplateArgument::NullPtr: {
    ExpectedType ToTypeOrErr = import(From.getNullPtrType());
    if (!ToTypeOrErr)
      return ToTypeOrErr.takeError();
    return TemplateArgument(*ToTypeOrErr, /*isNullPtr*/true);
  }

  case TemplateArgument::Template: {
    Expected<TemplateName> ToTemplateOrErr = import(From.getAsTemplate());
    if (!ToTemplateOrErr)
      return ToTemplateOrErr.takeError();

    return TemplateArgument(*ToTemplateOrErr);
  }

  case TemplateArgument::TemplateExpansion: {
    Expected<TemplateName> ToTemplateOrErr =
        import(From.getAsTemplateOrTemplatePattern());
    if (!ToTemplateOrErr)
      return ToTemplateOrErr.takeError();

    return TemplateArgument(
        *ToTemplateOrErr, From.getNumTemplateExpansions());
  }

  case TemplateArgument::Expression:
    if (ExpectedExpr ToExpr = import(From.getAsExpr()))
      return TemplateArgument(*ToExpr);
    else
      return ToExpr.takeError();

  case TemplateArgument::Pack: {
    SmallVector<TemplateArgument, 2> ToPack;
    ToPack.reserve(From.pack_size());
    if (Error Err = ImportTemplateArguments(
        From.pack_begin(), From.pack_size(), ToPack))
      return std::move(Err);

    return TemplateArgument(
        llvm::makeArrayRef(ToPack).copy(Importer.getToContext()));
  }
  }

  llvm_unreachable("Invalid template argument kind");
}

template <>
Expected<TemplateArgumentLoc>
ASTNodeImporter::import(const TemplateArgumentLoc &TALoc) {
  Expected<TemplateArgument> ArgOrErr = import(TALoc.getArgument());
  if (!ArgOrErr)
    return ArgOrErr.takeError();
  TemplateArgument Arg = *ArgOrErr;

  TemplateArgumentLocInfo FromInfo = TALoc.getLocInfo();

  TemplateArgumentLocInfo ToInfo;
  if (Arg.getKind() == TemplateArgument::Expression) {
    ExpectedExpr E = import(FromInfo.getAsExpr());
    if (!E)
      return E.takeError();
    ToInfo = TemplateArgumentLocInfo(*E);
  } else if (Arg.getKind() == TemplateArgument::Type) {
    if (auto TSIOrErr = import(FromInfo.getAsTypeSourceInfo()))
      ToInfo = TemplateArgumentLocInfo(*TSIOrErr);
    else
      return TSIOrErr.takeError();
  } else {
    auto ToTemplateQualifierLocOrErr =
        import(FromInfo.getTemplateQualifierLoc());
    if (!ToTemplateQualifierLocOrErr)
      return ToTemplateQualifierLocOrErr.takeError();
    auto ToTemplateNameLocOrErr = import(FromInfo.getTemplateNameLoc());
    if (!ToTemplateNameLocOrErr)
      return ToTemplateNameLocOrErr.takeError();
    auto ToTemplateEllipsisLocOrErr =
        import(FromInfo.getTemplateEllipsisLoc());
    if (!ToTemplateEllipsisLocOrErr)
      return ToTemplateEllipsisLocOrErr.takeError();

    ToInfo = TemplateArgumentLocInfo(
          *ToTemplateQualifierLocOrErr,
          *ToTemplateNameLocOrErr,
          *ToTemplateEllipsisLocOrErr);
  }

  return TemplateArgumentLoc(Arg, ToInfo);
}

template <>
Expected<DeclGroupRef> ASTNodeImporter::import(const DeclGroupRef &DG) {
  if (DG.isNull())
    return DeclGroupRef::Create(Importer.getToContext(), nullptr, 0);
  size_t NumDecls = DG.end() - DG.begin();
  SmallVector<Decl *, 1> ToDecls;
  ToDecls.reserve(NumDecls);
  for (Decl *FromD : DG) {
    if (auto ToDOrErr = import(FromD))
      ToDecls.push_back(*ToDOrErr);
    else
      return ToDOrErr.takeError();
  }
  return DeclGroupRef::Create(Importer.getToContext(),
                              ToDecls.begin(),
                              NumDecls);
}

template <>
Expected<ASTNodeImporter::Designator>
ASTNodeImporter::import(const Designator &D) {
  if (D.isFieldDesignator()) {
    IdentifierInfo *ToFieldName = Importer.Import(D.getFieldName());

    ExpectedSLoc ToDotLocOrErr = import(D.getDotLoc());
    if (!ToDotLocOrErr)
      return ToDotLocOrErr.takeError();

    ExpectedSLoc ToFieldLocOrErr = import(D.getFieldLoc());
    if (!ToFieldLocOrErr)
      return ToFieldLocOrErr.takeError();

    return Designator(ToFieldName, *ToDotLocOrErr, *ToFieldLocOrErr);
  }

  ExpectedSLoc ToLBracketLocOrErr = import(D.getLBracketLoc());
  if (!ToLBracketLocOrErr)
    return ToLBracketLocOrErr.takeError();

  ExpectedSLoc ToRBracketLocOrErr = import(D.getRBracketLoc());
  if (!ToRBracketLocOrErr)
    return ToRBracketLocOrErr.takeError();

  if (D.isArrayDesignator())
    return Designator(D.getFirstExprIndex(),
                      *ToLBracketLocOrErr, *ToRBracketLocOrErr);

  ExpectedSLoc ToEllipsisLocOrErr = import(D.getEllipsisLoc());
  if (!ToEllipsisLocOrErr)
    return ToEllipsisLocOrErr.takeError();

  assert(D.isArrayRangeDesignator());
  return Designator(
      D.getFirstExprIndex(), *ToLBracketLocOrErr, *ToEllipsisLocOrErr,
      *ToRBracketLocOrErr);
}

template <>
Expected<LambdaCapture> ASTNodeImporter::import(const LambdaCapture &From) {
  VarDecl *Var = nullptr;
  if (From.capturesVariable()) {
    if (auto VarOrErr = import(From.getCapturedVar()))
      Var = *VarOrErr;
    else
      return VarOrErr.takeError();
  }

  auto LocationOrErr = import(From.getLocation());
  if (!LocationOrErr)
    return LocationOrErr.takeError();

  SourceLocation EllipsisLoc;
  if (From.isPackExpansion())
    if (Error Err = importInto(EllipsisLoc, From.getEllipsisLoc()))
      return std::move(Err);

  return LambdaCapture(
      *LocationOrErr, From.isImplicit(), From.getCaptureKind(), Var,
      EllipsisLoc);
}

template <typename T>
bool ASTNodeImporter::hasSameVisibilityContext(T *Found, T *From) {
  if (From->hasExternalFormalLinkage())
    return Found->hasExternalFormalLinkage();
  if (Importer.GetFromTU(Found) != From->getTranslationUnitDecl())
    return false;
  if (From->isInAnonymousNamespace())
    return Found->isInAnonymousNamespace();
  else
    return !Found->isInAnonymousNamespace() &&
           !Found->hasExternalFormalLinkage();
}

template <>
bool ASTNodeImporter::hasSameVisibilityContext(TypedefNameDecl *Found,
                                               TypedefNameDecl *From) {
  if (From->isInAnonymousNamespace() && Found->isInAnonymousNamespace())
    return Importer.GetFromTU(Found) == From->getTranslationUnitDecl();
  return From->isInAnonymousNamespace() == Found->isInAnonymousNamespace();
}

} // namespace clang

//----------------------------------------------------------------------------
// Import Types
//----------------------------------------------------------------------------

using namespace clang;

ExpectedType ASTNodeImporter::VisitType(const Type *T) {
  Importer.FromDiag(SourceLocation(), diag::err_unsupported_ast_node)
    << T->getTypeClassName();
  return make_error<ImportError>(ImportError::UnsupportedConstruct);
}

ExpectedType ASTNodeImporter::VisitAtomicType(const AtomicType *T){
  ExpectedType UnderlyingTypeOrErr = import(T->getValueType());
  if (!UnderlyingTypeOrErr)
    return UnderlyingTypeOrErr.takeError();

  return Importer.getToContext().getAtomicType(*UnderlyingTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitBuiltinType(const BuiltinType *T) {
  switch (T->getKind()) {
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) \
  case BuiltinType::Id: \
    return Importer.getToContext().SingletonId;
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) \
  case BuiltinType::Id: \
    return Importer.getToContext().Id##Ty;
#include "clang/Basic/OpenCLExtensionTypes.def"
#define SVE_TYPE(Name, Id, SingletonId) \
  case BuiltinType::Id: \
    return Importer.getToContext().SingletonId;
#include "clang/Basic/AArch64SVEACLETypes.def"
#define SHARED_SINGLETON_TYPE(Expansion)
#define BUILTIN_TYPE(Id, SingletonId) \
  case BuiltinType::Id: return Importer.getToContext().SingletonId;
#include "clang/AST/BuiltinTypes.def"

  // FIXME: for Char16, Char32, and NullPtr, make sure that the "to"
  // context supports C++.

  // FIXME: for ObjCId, ObjCClass, and ObjCSel, make sure that the "to"
  // context supports ObjC.

  case BuiltinType::Char_U:
    // The context we're importing from has an unsigned 'char'. If we're
    // importing into a context with a signed 'char', translate to
    // 'unsigned char' instead.
    if (Importer.getToContext().getLangOpts().CharIsSigned)
      return Importer.getToContext().UnsignedCharTy;

    return Importer.getToContext().CharTy;

  case BuiltinType::Char_S:
    // The context we're importing from has an unsigned 'char'. If we're
    // importing into a context with a signed 'char', translate to
    // 'unsigned char' instead.
    if (!Importer.getToContext().getLangOpts().CharIsSigned)
      return Importer.getToContext().SignedCharTy;

    return Importer.getToContext().CharTy;

  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
    // FIXME: If not in C++, shall we translate to the C equivalent of
    // wchar_t?
    return Importer.getToContext().WCharTy;
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

ExpectedType ASTNodeImporter::VisitDecayedType(const DecayedType *T) {
  ExpectedType ToOriginalTypeOrErr = import(T->getOriginalType());
  if (!ToOriginalTypeOrErr)
    return ToOriginalTypeOrErr.takeError();

  return Importer.getToContext().getDecayedType(*ToOriginalTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitComplexType(const ComplexType *T) {
  ExpectedType ToElementTypeOrErr = import(T->getElementType());
  if (!ToElementTypeOrErr)
    return ToElementTypeOrErr.takeError();

  return Importer.getToContext().getComplexType(*ToElementTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitPointerType(const PointerType *T) {
  ExpectedType ToPointeeTypeOrErr = import(T->getPointeeType());
  if (!ToPointeeTypeOrErr)
    return ToPointeeTypeOrErr.takeError();

  return Importer.getToContext().getPointerType(*ToPointeeTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitBlockPointerType(const BlockPointerType *T) {
  // FIXME: Check for blocks support in "to" context.
  ExpectedType ToPointeeTypeOrErr = import(T->getPointeeType());
  if (!ToPointeeTypeOrErr)
    return ToPointeeTypeOrErr.takeError();

  return Importer.getToContext().getBlockPointerType(*ToPointeeTypeOrErr);
}

ExpectedType
ASTNodeImporter::VisitLValueReferenceType(const LValueReferenceType *T) {
  // FIXME: Check for C++ support in "to" context.
  ExpectedType ToPointeeTypeOrErr = import(T->getPointeeTypeAsWritten());
  if (!ToPointeeTypeOrErr)
    return ToPointeeTypeOrErr.takeError();

  return Importer.getToContext().getLValueReferenceType(*ToPointeeTypeOrErr);
}

ExpectedType
ASTNodeImporter::VisitRValueReferenceType(const RValueReferenceType *T) {
  // FIXME: Check for C++0x support in "to" context.
  ExpectedType ToPointeeTypeOrErr = import(T->getPointeeTypeAsWritten());
  if (!ToPointeeTypeOrErr)
    return ToPointeeTypeOrErr.takeError();

  return Importer.getToContext().getRValueReferenceType(*ToPointeeTypeOrErr);
}

ExpectedType
ASTNodeImporter::VisitMemberPointerType(const MemberPointerType *T) {
  // FIXME: Check for C++ support in "to" context.
  ExpectedType ToPointeeTypeOrErr = import(T->getPointeeType());
  if (!ToPointeeTypeOrErr)
    return ToPointeeTypeOrErr.takeError();

  ExpectedType ClassTypeOrErr = import(QualType(T->getClass(), 0));
  if (!ClassTypeOrErr)
    return ClassTypeOrErr.takeError();

  return Importer.getToContext().getMemberPointerType(
      *ToPointeeTypeOrErr, (*ClassTypeOrErr).getTypePtr());
}

ExpectedType
ASTNodeImporter::VisitConstantArrayType(const ConstantArrayType *T) {
  QualType ToElementType;
  const Expr *ToSizeExpr;
  if (auto Imp = importSeq(T->getElementType(), T->getSizeExpr()))
    std::tie(ToElementType, ToSizeExpr) = *Imp;
  else
    return Imp.takeError();

  return Importer.getToContext().getConstantArrayType(
      ToElementType, T->getSize(), ToSizeExpr, T->getSizeModifier(),
      T->getIndexTypeCVRQualifiers());
}

ExpectedType
ASTNodeImporter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  ExpectedType ToElementTypeOrErr = import(T->getElementType());
  if (!ToElementTypeOrErr)
    return ToElementTypeOrErr.takeError();

  return Importer.getToContext().getIncompleteArrayType(*ToElementTypeOrErr,
                                                        T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers());
}

ExpectedType
ASTNodeImporter::VisitVariableArrayType(const VariableArrayType *T) {
  QualType ToElementType;
  Expr *ToSizeExpr;
  SourceRange ToBracketsRange;
  if (auto Imp = importSeq(
      T->getElementType(), T->getSizeExpr(), T->getBracketsRange()))
    std::tie(ToElementType, ToSizeExpr, ToBracketsRange) = *Imp;
  else
    return Imp.takeError();

  return Importer.getToContext().getVariableArrayType(
      ToElementType, ToSizeExpr, T->getSizeModifier(),
      T->getIndexTypeCVRQualifiers(), ToBracketsRange);
}

ExpectedType ASTNodeImporter::VisitDependentSizedArrayType(
    const DependentSizedArrayType *T) {
  QualType ToElementType;
  Expr *ToSizeExpr;
  SourceRange ToBracketsRange;
  if (auto Imp = importSeq(
      T->getElementType(), T->getSizeExpr(), T->getBracketsRange()))
    std::tie(ToElementType, ToSizeExpr, ToBracketsRange) = *Imp;
  else
    return Imp.takeError();
  // SizeExpr may be null if size is not specified directly.
  // For example, 'int a[]'.

  return Importer.getToContext().getDependentSizedArrayType(
      ToElementType, ToSizeExpr, T->getSizeModifier(),
      T->getIndexTypeCVRQualifiers(), ToBracketsRange);
}

ExpectedType ASTNodeImporter::VisitVectorType(const VectorType *T) {
  ExpectedType ToElementTypeOrErr = import(T->getElementType());
  if (!ToElementTypeOrErr)
    return ToElementTypeOrErr.takeError();

  return Importer.getToContext().getVectorType(*ToElementTypeOrErr,
                                               T->getNumElements(),
                                               T->getVectorKind());
}

ExpectedType ASTNodeImporter::VisitExtVectorType(const ExtVectorType *T) {
  ExpectedType ToElementTypeOrErr = import(T->getElementType());
  if (!ToElementTypeOrErr)
    return ToElementTypeOrErr.takeError();

  return Importer.getToContext().getExtVectorType(*ToElementTypeOrErr,
                                                  T->getNumElements());
}

ExpectedType
ASTNodeImporter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  // FIXME: What happens if we're importing a function without a prototype
  // into C++? Should we make it variadic?
  ExpectedType ToReturnTypeOrErr = import(T->getReturnType());
  if (!ToReturnTypeOrErr)
    return ToReturnTypeOrErr.takeError();

  return Importer.getToContext().getFunctionNoProtoType(*ToReturnTypeOrErr,
                                                        T->getExtInfo());
}

ExpectedType
ASTNodeImporter::VisitFunctionProtoType(const FunctionProtoType *T) {
  ExpectedType ToReturnTypeOrErr = import(T->getReturnType());
  if (!ToReturnTypeOrErr)
    return ToReturnTypeOrErr.takeError();

  // Import argument types
  SmallVector<QualType, 4> ArgTypes;
  for (const auto &A : T->param_types()) {
    ExpectedType TyOrErr = import(A);
    if (!TyOrErr)
      return TyOrErr.takeError();
    ArgTypes.push_back(*TyOrErr);
  }

  // Import exception types
  SmallVector<QualType, 4> ExceptionTypes;
  for (const auto &E : T->exceptions()) {
    ExpectedType TyOrErr = import(E);
    if (!TyOrErr)
      return TyOrErr.takeError();
    ExceptionTypes.push_back(*TyOrErr);
  }

  FunctionProtoType::ExtProtoInfo FromEPI = T->getExtProtoInfo();
  FunctionProtoType::ExtProtoInfo ToEPI;

  auto Imp = importSeq(
      FromEPI.ExceptionSpec.NoexceptExpr,
      FromEPI.ExceptionSpec.SourceDecl,
      FromEPI.ExceptionSpec.SourceTemplate);
  if (!Imp)
    return Imp.takeError();

  ToEPI.ExtInfo = FromEPI.ExtInfo;
  ToEPI.Variadic = FromEPI.Variadic;
  ToEPI.HasTrailingReturn = FromEPI.HasTrailingReturn;
  ToEPI.TypeQuals = FromEPI.TypeQuals;
  ToEPI.RefQualifier = FromEPI.RefQualifier;
  ToEPI.ExceptionSpec.Type = FromEPI.ExceptionSpec.Type;
  ToEPI.ExceptionSpec.Exceptions = ExceptionTypes;
  std::tie(
      ToEPI.ExceptionSpec.NoexceptExpr,
      ToEPI.ExceptionSpec.SourceDecl,
      ToEPI.ExceptionSpec.SourceTemplate) = *Imp;

  return Importer.getToContext().getFunctionType(
      *ToReturnTypeOrErr, ArgTypes, ToEPI);
}

ExpectedType ASTNodeImporter::VisitUnresolvedUsingType(
    const UnresolvedUsingType *T) {
  UnresolvedUsingTypenameDecl *ToD;
  Decl *ToPrevD;
  if (auto Imp = importSeq(T->getDecl(), T->getDecl()->getPreviousDecl()))
    std::tie(ToD, ToPrevD) = *Imp;
  else
    return Imp.takeError();

  return Importer.getToContext().getTypeDeclType(
      ToD, cast_or_null<TypeDecl>(ToPrevD));
}

ExpectedType ASTNodeImporter::VisitParenType(const ParenType *T) {
  ExpectedType ToInnerTypeOrErr = import(T->getInnerType());
  if (!ToInnerTypeOrErr)
    return ToInnerTypeOrErr.takeError();

  return Importer.getToContext().getParenType(*ToInnerTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitTypedefType(const TypedefType *T) {
  Expected<TypedefNameDecl *> ToDeclOrErr = import(T->getDecl());
  if (!ToDeclOrErr)
    return ToDeclOrErr.takeError();

  return Importer.getToContext().getTypeDeclType(*ToDeclOrErr);
}

ExpectedType ASTNodeImporter::VisitTypeOfExprType(const TypeOfExprType *T) {
  ExpectedExpr ToExprOrErr = import(T->getUnderlyingExpr());
  if (!ToExprOrErr)
    return ToExprOrErr.takeError();

  return Importer.getToContext().getTypeOfExprType(*ToExprOrErr);
}

ExpectedType ASTNodeImporter::VisitTypeOfType(const TypeOfType *T) {
  ExpectedType ToUnderlyingTypeOrErr = import(T->getUnderlyingType());
  if (!ToUnderlyingTypeOrErr)
    return ToUnderlyingTypeOrErr.takeError();

  return Importer.getToContext().getTypeOfType(*ToUnderlyingTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitDecltypeType(const DecltypeType *T) {
  // FIXME: Make sure that the "to" context supports C++0x!
  ExpectedExpr ToExprOrErr = import(T->getUnderlyingExpr());
  if (!ToExprOrErr)
    return ToExprOrErr.takeError();

  ExpectedType ToUnderlyingTypeOrErr = import(T->getUnderlyingType());
  if (!ToUnderlyingTypeOrErr)
    return ToUnderlyingTypeOrErr.takeError();

  return Importer.getToContext().getDecltypeType(
      *ToExprOrErr, *ToUnderlyingTypeOrErr);
}

ExpectedType
ASTNodeImporter::VisitUnaryTransformType(const UnaryTransformType *T) {
  ExpectedType ToBaseTypeOrErr = import(T->getBaseType());
  if (!ToBaseTypeOrErr)
    return ToBaseTypeOrErr.takeError();

  ExpectedType ToUnderlyingTypeOrErr = import(T->getUnderlyingType());
  if (!ToUnderlyingTypeOrErr)
    return ToUnderlyingTypeOrErr.takeError();

  return Importer.getToContext().getUnaryTransformType(
      *ToBaseTypeOrErr, *ToUnderlyingTypeOrErr, T->getUTTKind());
}

ExpectedType ASTNodeImporter::VisitAutoType(const AutoType *T) {
  // FIXME: Make sure that the "to" context supports C++11!
  ExpectedType ToDeducedTypeOrErr = import(T->getDeducedType());
  if (!ToDeducedTypeOrErr)
    return ToDeducedTypeOrErr.takeError();

  return Importer.getToContext().getAutoType(*ToDeducedTypeOrErr,
                                             T->getKeyword(),
                                             /*IsDependent*/false);
}

ExpectedType ASTNodeImporter::VisitInjectedClassNameType(
    const InjectedClassNameType *T) {
  Expected<CXXRecordDecl *> ToDeclOrErr = import(T->getDecl());
  if (!ToDeclOrErr)
    return ToDeclOrErr.takeError();

  ExpectedType ToInjTypeOrErr = import(T->getInjectedSpecializationType());
  if (!ToInjTypeOrErr)
    return ToInjTypeOrErr.takeError();

  // FIXME: ASTContext::getInjectedClassNameType is not suitable for AST reading
  // See comments in InjectedClassNameType definition for details
  // return Importer.getToContext().getInjectedClassNameType(D, InjType);
  enum {
    TypeAlignmentInBits = 4,
    TypeAlignment = 1 << TypeAlignmentInBits
  };

  return QualType(new (Importer.getToContext(), TypeAlignment)
                  InjectedClassNameType(*ToDeclOrErr, *ToInjTypeOrErr), 0);
}

ExpectedType ASTNodeImporter::VisitRecordType(const RecordType *T) {
  Expected<RecordDecl *> ToDeclOrErr = import(T->getDecl());
  if (!ToDeclOrErr)
    return ToDeclOrErr.takeError();

  return Importer.getToContext().getTagDeclType(*ToDeclOrErr);
}

ExpectedType ASTNodeImporter::VisitEnumType(const EnumType *T) {
  Expected<EnumDecl *> ToDeclOrErr = import(T->getDecl());
  if (!ToDeclOrErr)
    return ToDeclOrErr.takeError();

  return Importer.getToContext().getTagDeclType(*ToDeclOrErr);
}

ExpectedType ASTNodeImporter::VisitAttributedType(const AttributedType *T) {
  ExpectedType ToModifiedTypeOrErr = import(T->getModifiedType());
  if (!ToModifiedTypeOrErr)
    return ToModifiedTypeOrErr.takeError();
  ExpectedType ToEquivalentTypeOrErr = import(T->getEquivalentType());
  if (!ToEquivalentTypeOrErr)
    return ToEquivalentTypeOrErr.takeError();

  return Importer.getToContext().getAttributedType(T->getAttrKind(),
      *ToModifiedTypeOrErr, *ToEquivalentTypeOrErr);
}

ExpectedType ASTNodeImporter::VisitTemplateTypeParmType(
    const TemplateTypeParmType *T) {
  Expected<TemplateTypeParmDecl *> ToDeclOrErr = import(T->getDecl());
  if (!ToDeclOrErr)
    return ToDeclOrErr.takeError();

  return Importer.getToContext().getTemplateTypeParmType(
      T->getDepth(), T->getIndex(), T->isParameterPack(), *ToDeclOrErr);
}

ExpectedType ASTNodeImporter::VisitSubstTemplateTypeParmType(
    const SubstTemplateTypeParmType *T) {
  ExpectedType ReplacedOrErr = import(QualType(T->getReplacedParameter(), 0));
  if (!ReplacedOrErr)
    return ReplacedOrErr.takeError();
  const TemplateTypeParmType *Replaced =
      cast<TemplateTypeParmType>((*ReplacedOrErr).getTypePtr());

  ExpectedType ToReplacementTypeOrErr = import(T->getReplacementType());
  if (!ToReplacementTypeOrErr)
    return ToReplacementTypeOrErr.takeError();

  return Importer.getToContext().getSubstTemplateTypeParmType(
        Replaced, (*ToReplacementTypeOrErr).getCanonicalType());
}

ExpectedType ASTNodeImporter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  auto ToTemplateOrErr = import(T->getTemplateName());
  if (!ToTemplateOrErr)
    return ToTemplateOrErr.takeError();

  SmallVector<TemplateArgument, 2> ToTemplateArgs;
  if (Error Err = ImportTemplateArguments(
      T->getArgs(), T->getNumArgs(), ToTemplateArgs))
    return std::move(Err);

  QualType ToCanonType;
  if (!QualType(T, 0).isCanonical()) {
    QualType FromCanonType
      = Importer.getFromContext().getCanonicalType(QualType(T, 0));
    if (ExpectedType TyOrErr = import(FromCanonType))
      ToCanonType = *TyOrErr;
    else
      return TyOrErr.takeError();
  }
  return Importer.getToContext().getTemplateSpecializationType(*ToTemplateOrErr,
                                                               ToTemplateArgs,
                                                               ToCanonType);
}

ExpectedType ASTNodeImporter::VisitElaboratedType(const ElaboratedType *T) {
  // Note: the qualifier in an ElaboratedType is optional.
  auto ToQualifierOrErr = import(T->getQualifier());
  if (!ToQualifierOrErr)
    return ToQualifierOrErr.takeError();

  ExpectedType ToNamedTypeOrErr = import(T->getNamedType());
  if (!ToNamedTypeOrErr)
    return ToNamedTypeOrErr.takeError();

  Expected<TagDecl *> ToOwnedTagDeclOrErr = import(T->getOwnedTagDecl());
  if (!ToOwnedTagDeclOrErr)
    return ToOwnedTagDeclOrErr.takeError();

  return Importer.getToContext().getElaboratedType(T->getKeyword(),
                                                   *ToQualifierOrErr,
                                                   *ToNamedTypeOrErr,
                                                   *ToOwnedTagDeclOrErr);
}

ExpectedType
ASTNodeImporter::VisitPackExpansionType(const PackExpansionType *T) {
  ExpectedType ToPatternOrErr = import(T->getPattern());
  if (!ToPatternOrErr)
    return ToPatternOrErr.takeError();

  return Importer.getToContext().getPackExpansionType(*ToPatternOrErr,
                                                      T->getNumExpansions());
}

ExpectedType ASTNodeImporter::VisitDependentTemplateSpecializationType(
    const DependentTemplateSpecializationType *T) {
  auto ToQualifierOrErr = import(T->getQualifier());
  if (!ToQualifierOrErr)
    return ToQualifierOrErr.takeError();

  IdentifierInfo *ToName = Importer.Import(T->getIdentifier());

  SmallVector<TemplateArgument, 2> ToPack;
  ToPack.reserve(T->getNumArgs());
  if (Error Err = ImportTemplateArguments(
      T->getArgs(), T->getNumArgs(), ToPack))
    return std::move(Err);

  return Importer.getToContext().getDependentTemplateSpecializationType(
      T->getKeyword(), *ToQualifierOrErr, ToName, ToPack);
}

ExpectedType
ASTNodeImporter::VisitDependentNameType(const DependentNameType *T) {
  auto ToQualifierOrErr = import(T->getQualifier());
  if (!ToQualifierOrErr)
    return ToQualifierOrErr.takeError();

  IdentifierInfo *Name = Importer.Import(T->getIdentifier());

  QualType Canon;
  if (T != T->getCanonicalTypeInternal().getTypePtr()) {
    if (ExpectedType TyOrErr = import(T->getCanonicalTypeInternal()))
      Canon = (*TyOrErr).getCanonicalType();
    else
      return TyOrErr.takeError();
  }

  return Importer.getToContext().getDependentNameType(T->getKeyword(),
                                                      *ToQualifierOrErr,
                                                      Name, Canon);
}

ExpectedType
ASTNodeImporter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  Expected<ObjCInterfaceDecl *> ToDeclOrErr = import(T->getDecl());
  if (!ToDeclOrErr)
    return ToDeclOrErr.takeError();

  return Importer.getToContext().getObjCInterfaceType(*ToDeclOrErr);
}

ExpectedType ASTNodeImporter::VisitObjCObjectType(const ObjCObjectType *T) {
  ExpectedType ToBaseTypeOrErr = import(T->getBaseType());
  if (!ToBaseTypeOrErr)
    return ToBaseTypeOrErr.takeError();

  SmallVector<QualType, 4> TypeArgs;
  for (auto TypeArg : T->getTypeArgsAsWritten()) {
    if (ExpectedType TyOrErr = import(TypeArg))
      TypeArgs.push_back(*TyOrErr);
    else
      return TyOrErr.takeError();
  }

  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  for (auto *P : T->quals()) {
    if (Expected<ObjCProtocolDecl *> ProtocolOrErr = import(P))
      Protocols.push_back(*ProtocolOrErr);
    else
      return ProtocolOrErr.takeError();

  }

  return Importer.getToContext().getObjCObjectType(*ToBaseTypeOrErr, TypeArgs,
                                                   Protocols,
                                                   T->isKindOfTypeAsWritten());
}

ExpectedType
ASTNodeImporter::VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
  ExpectedType ToPointeeTypeOrErr = import(T->getPointeeType());
  if (!ToPointeeTypeOrErr)
    return ToPointeeTypeOrErr.takeError();

  return Importer.getToContext().getObjCObjectPointerType(*ToPointeeTypeOrErr);
}

//----------------------------------------------------------------------------
// Import Declarations
//----------------------------------------------------------------------------
Error ASTNodeImporter::ImportDeclParts(
    NamedDecl *D, DeclContext *&DC, DeclContext *&LexicalDC,
    DeclarationName &Name, NamedDecl *&ToD, SourceLocation &Loc) {
  // Check if RecordDecl is in FunctionDecl parameters to avoid infinite loop.
  // example: int struct_in_proto(struct data_t{int a;int b;} *d);
  // FIXME: We could support these constructs by importing a different type of
  // this parameter and by importing the original type of the parameter only
  // after the FunctionDecl is created. See
  // VisitFunctionDecl::UsedDifferentProtoType.
  DeclContext *OrigDC = D->getDeclContext();
  FunctionDecl *FunDecl;
  if (isa<RecordDecl>(D) && (FunDecl = dyn_cast<FunctionDecl>(OrigDC)) &&
      FunDecl->hasBody()) {
    auto getLeafPointeeType = [](const Type *T) {
      while (T->isPointerType() || T->isArrayType()) {
        T = T->getPointeeOrArrayElementType();
      }
      return T;
    };
    for (const ParmVarDecl *P : FunDecl->parameters()) {
      const Type *LeafT =
          getLeafPointeeType(P->getType().getCanonicalType().getTypePtr());
      auto *RT = dyn_cast<RecordType>(LeafT);
      if (RT && RT->getDecl() == D) {
        Importer.FromDiag(D->getLocation(), diag::err_unsupported_ast_node)
            << D->getDeclKindName();
        return make_error<ImportError>(ImportError::UnsupportedConstruct);
      }
    }
  }

  // Import the context of this declaration.
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return Err;

  // Import the name of this declaration.
  if (Error Err = importInto(Name, D->getDeclName()))
    return Err;

  // Import the location of this declaration.
  if (Error Err = importInto(Loc, D->getLocation()))
    return Err;

  ToD = cast_or_null<NamedDecl>(Importer.GetAlreadyImportedOrNull(D));
  if (ToD)
    if (Error Err = ASTNodeImporter(*this).ImportDefinitionIfNeeded(D, ToD))
      return Err;

  return Error::success();
}

Error ASTNodeImporter::ImportDefinitionIfNeeded(Decl *FromD, Decl *ToD) {
  if (!FromD)
    return Error::success();

  if (!ToD)
    if (Error Err = importInto(ToD, FromD))
      return Err;

  if (RecordDecl *FromRecord = dyn_cast<RecordDecl>(FromD)) {
    if (RecordDecl *ToRecord = cast<RecordDecl>(ToD)) {
      if (FromRecord->getDefinition() && FromRecord->isCompleteDefinition() &&
          !ToRecord->getDefinition()) {
        if (Error Err = ImportDefinition(FromRecord, ToRecord))
          return Err;
      }
    }
    return Error::success();
  }

  if (EnumDecl *FromEnum = dyn_cast<EnumDecl>(FromD)) {
    if (EnumDecl *ToEnum = cast<EnumDecl>(ToD)) {
      if (FromEnum->getDefinition() && !ToEnum->getDefinition()) {
        if (Error Err = ImportDefinition(FromEnum, ToEnum))
          return Err;
      }
    }
    return Error::success();
  }

  return Error::success();
}

Error
ASTNodeImporter::ImportDeclarationNameLoc(
    const DeclarationNameInfo &From, DeclarationNameInfo& To) {
  // NOTE: To.Name and To.Loc are already imported.
  // We only have to import To.LocInfo.
  switch (To.getName().getNameKind()) {
  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXUsingDirective:
  case DeclarationName::CXXDeductionGuideName:
    return Error::success();

  case DeclarationName::CXXOperatorName: {
    if (auto ToRangeOrErr = import(From.getCXXOperatorNameRange()))
      To.setCXXOperatorNameRange(*ToRangeOrErr);
    else
      return ToRangeOrErr.takeError();
    return Error::success();
  }
  case DeclarationName::CXXLiteralOperatorName: {
    if (ExpectedSLoc LocOrErr = import(From.getCXXLiteralOperatorNameLoc()))
      To.setCXXLiteralOperatorNameLoc(*LocOrErr);
    else
      return LocOrErr.takeError();
    return Error::success();
  }
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName: {
    if (auto ToTInfoOrErr = import(From.getNamedTypeInfo()))
      To.setNamedTypeInfo(*ToTInfoOrErr);
    else
      return ToTInfoOrErr.takeError();
    return Error::success();
  }
  }
  llvm_unreachable("Unknown name kind.");
}

Error
ASTNodeImporter::ImportDeclContext(DeclContext *FromDC, bool ForceImport) {
  if (Importer.isMinimalImport() && !ForceImport) {
    auto ToDCOrErr = Importer.ImportContext(FromDC);
    return ToDCOrErr.takeError();
  }

  // We use strict error handling in case of records and enums, but not
  // with e.g. namespaces.
  //
  // FIXME Clients of the ASTImporter should be able to choose an
  // appropriate error handling strategy for their needs.  For instance,
  // they may not want to mark an entire namespace as erroneous merely
  // because there is an ODR error with two typedefs.  As another example,
  // the client may allow EnumConstantDecls with same names but with
  // different values in two distinct translation units.
  bool AccumulateChildErrors = isa<TagDecl>(FromDC);

  Error ChildErrors = Error::success();
  for (auto *From : FromDC->decls()) {
    ExpectedDecl ImportedOrErr = import(From);

    // If we are in the process of ImportDefinition(...) for a RecordDecl we
    // want to make sure that we are also completing each FieldDecl. There
    // are currently cases where this does not happen and this is correctness
    // fix since operations such as code generation will expect this to be so.
    if (ImportedOrErr) {
      FieldDecl *FieldFrom = dyn_cast_or_null<FieldDecl>(From);
      Decl *ImportedDecl = (Decl*)*ImportedOrErr;
      FieldDecl *FieldTo = dyn_cast_or_null<FieldDecl>(ImportedDecl);
      if (FieldFrom && FieldTo) {
        const RecordType *RecordFrom = FieldFrom->getType()->getAs<RecordType>();
        const RecordType *RecordTo = FieldTo->getType()->getAs<RecordType>();
        if (RecordFrom && RecordTo) {
          RecordDecl *FromRecordDecl = RecordFrom->getDecl();
          RecordDecl *ToRecordDecl = RecordTo->getDecl();

          if (FromRecordDecl->isCompleteDefinition() &&
              !ToRecordDecl->isCompleteDefinition()) {
            Error Err = ImportDefinition(FromRecordDecl, ToRecordDecl);

            if (Err && AccumulateChildErrors)
              ChildErrors =  joinErrors(std::move(ChildErrors), std::move(Err));
            else
              consumeError(std::move(Err));
          }
        }
      }
    } else {
      if (AccumulateChildErrors)
        ChildErrors =
            joinErrors(std::move(ChildErrors), ImportedOrErr.takeError());
      else
        consumeError(ImportedOrErr.takeError());
    }
  }

  // We reorder declarations in RecordDecls because they may have another order
  // in the "to" context than they have in the "from" context. This may happen
  // e.g when we import a class like this:
  //    struct declToImport {
  //        int a = c + b;
  //        int b = 1;
  //        int c = 2;
  //    };
  // During the import of `a` we import first the dependencies in sequence,
  // thus the order would be `c`, `b`, `a`. We will get the normal order by
  // first removing the already imported members and then adding them in the
  // order as they apper in the "from" context.
  //
  // Keeping field order is vital because it determines structure layout.
  //
  // Here and below, we cannot call field_begin() method and its callers on
  // ToDC if it has an external storage. Calling field_begin() will
  // automatically load all the fields by calling
  // LoadFieldsFromExternalStorage(). LoadFieldsFromExternalStorage() would
  // call ASTImporter::Import(). This is because the ExternalASTSource
  // interface in LLDB is implemented by the means of the ASTImporter. However,
  // calling an import at this point would result in an uncontrolled import, we
  // must avoid that.
  const auto *FromRD = dyn_cast<RecordDecl>(FromDC);
  if (!FromRD)
    return ChildErrors;

  auto ToDCOrErr = Importer.ImportContext(FromDC);
  if (!ToDCOrErr) {
    consumeError(std::move(ChildErrors));
    return ToDCOrErr.takeError();
  }

  DeclContext *ToDC = *ToDCOrErr;
  // Remove all declarations, which may be in wrong order in the
  // lexical DeclContext and then add them in the proper order.
  for (auto *D : FromRD->decls()) {
    if (isa<FieldDecl>(D) || isa<IndirectFieldDecl>(D) || isa<FriendDecl>(D)) {
      assert(D && "DC contains a null decl");
      Decl *ToD = Importer.GetAlreadyImportedOrNull(D);
      // Remove only the decls which we successfully imported.
      if (ToD) {
        assert(ToDC == ToD->getLexicalDeclContext() && ToDC->containsDecl(ToD));
        // Remove the decl from its wrong place in the linked list.
        ToDC->removeDecl(ToD);
        // Add the decl to the end of the linked list.
        // This time it will be at the proper place because the enclosing for
        // loop iterates in the original (good) order of the decls.
        ToDC->addDeclInternal(ToD);
      }
    }
  }

  return ChildErrors;
}

Error ASTNodeImporter::ImportDeclContext(
    Decl *FromD, DeclContext *&ToDC, DeclContext *&ToLexicalDC) {
  auto ToDCOrErr = Importer.ImportContext(FromD->getDeclContext());
  if (!ToDCOrErr)
    return ToDCOrErr.takeError();
  ToDC = *ToDCOrErr;

  if (FromD->getDeclContext() != FromD->getLexicalDeclContext()) {
    auto ToLexicalDCOrErr = Importer.ImportContext(
        FromD->getLexicalDeclContext());
    if (!ToLexicalDCOrErr)
      return ToLexicalDCOrErr.takeError();
    ToLexicalDC = *ToLexicalDCOrErr;
  } else
    ToLexicalDC = ToDC;

  return Error::success();
}

Error ASTNodeImporter::ImportImplicitMethods(
    const CXXRecordDecl *From, CXXRecordDecl *To) {
  assert(From->isCompleteDefinition() && To->getDefinition() == To &&
      "Import implicit methods to or from non-definition");

  for (CXXMethodDecl *FromM : From->methods())
    if (FromM->isImplicit()) {
      Expected<CXXMethodDecl *> ToMOrErr = import(FromM);
      if (!ToMOrErr)
        return ToMOrErr.takeError();
    }

  return Error::success();
}

static Error setTypedefNameForAnonDecl(TagDecl *From, TagDecl *To,
                                       ASTImporter &Importer) {
  if (TypedefNameDecl *FromTypedef = From->getTypedefNameForAnonDecl()) {
    if (ExpectedDecl ToTypedefOrErr = Importer.Import(FromTypedef))
      To->setTypedefNameForAnonDecl(cast<TypedefNameDecl>(*ToTypedefOrErr));
    else
      return ToTypedefOrErr.takeError();
  }
  return Error::success();
}

Error ASTNodeImporter::ImportDefinition(
    RecordDecl *From, RecordDecl *To, ImportDefinitionKind Kind) {
  auto DefinitionCompleter = [To]() {
    // There are cases in LLDB when we first import a class without its
    // members. The class will have DefinitionData, but no members. Then,
    // importDefinition is called from LLDB, which tries to get the members, so
    // when we get here, the class already has the DefinitionData set, so we
    // must unset the CompleteDefinition here to be able to complete again the
    // definition.
    To->setCompleteDefinition(false);
    To->completeDefinition();
  };

  if (To->getDefinition() || To->isBeingDefined()) {
    if (Kind == IDK_Everything ||
        // In case of lambdas, the class already has a definition ptr set, but
        // the contained decls are not imported yet. Also, isBeingDefined was
        // set in CXXRecordDecl::CreateLambda.  We must import the contained
        // decls here and finish the definition.
        (To->isLambda() && shouldForceImportDeclContext(Kind))) {
      Error Result = ImportDeclContext(From, /*ForceImport=*/true);
      // Finish the definition of the lambda, set isBeingDefined to false.
      if (To->isLambda())
        DefinitionCompleter();
      return Result;
    }

    return Error::success();
  }

  To->startDefinition();
  // Complete the definition even if error is returned.
  // The RecordDecl may be already part of the AST so it is better to
  // have it in complete state even if something is wrong with it.
  auto DefinitionCompleterScopeExit =
      llvm::make_scope_exit(DefinitionCompleter);

  if (Error Err = setTypedefNameForAnonDecl(From, To, Importer))
    return Err;

  // Add base classes.
  auto *ToCXX = dyn_cast<CXXRecordDecl>(To);
  auto *FromCXX = dyn_cast<CXXRecordDecl>(From);
  if (ToCXX && FromCXX && ToCXX->dataPtr() && FromCXX->dataPtr()) {

    struct CXXRecordDecl::DefinitionData &ToData = ToCXX->data();
    struct CXXRecordDecl::DefinitionData &FromData = FromCXX->data();

    #define FIELD(Name, Width, Merge) \
    ToData.Name = FromData.Name;
    #include "clang/AST/CXXRecordDeclDefinitionBits.def"

    // Copy over the data stored in RecordDeclBits
    ToCXX->setArgPassingRestrictions(FromCXX->getArgPassingRestrictions());

    SmallVector<CXXBaseSpecifier *, 4> Bases;
    for (const auto &Base1 : FromCXX->bases()) {
      ExpectedType TyOrErr = import(Base1.getType());
      if (!TyOrErr)
        return TyOrErr.takeError();

      SourceLocation EllipsisLoc;
      if (Base1.isPackExpansion()) {
        if (ExpectedSLoc LocOrErr = import(Base1.getEllipsisLoc()))
          EllipsisLoc = *LocOrErr;
        else
          return LocOrErr.takeError();
      }

      // Ensure that we have a definition for the base.
      if (Error Err =
          ImportDefinitionIfNeeded(Base1.getType()->getAsCXXRecordDecl()))
        return Err;

      auto RangeOrErr = import(Base1.getSourceRange());
      if (!RangeOrErr)
        return RangeOrErr.takeError();

      auto TSIOrErr = import(Base1.getTypeSourceInfo());
      if (!TSIOrErr)
        return TSIOrErr.takeError();

      Bases.push_back(
          new (Importer.getToContext()) CXXBaseSpecifier(
              *RangeOrErr,
              Base1.isVirtual(),
              Base1.isBaseOfClass(),
              Base1.getAccessSpecifierAsWritten(),
              *TSIOrErr,
              EllipsisLoc));
    }
    if (!Bases.empty())
      ToCXX->setBases(Bases.data(), Bases.size());
  }

  if (shouldForceImportDeclContext(Kind))
    if (Error Err = ImportDeclContext(From, /*ForceImport=*/true))
      return Err;

  return Error::success();
}

Error ASTNodeImporter::ImportInitializer(VarDecl *From, VarDecl *To) {
  if (To->getAnyInitializer())
    return Error::success();

  Expr *FromInit = From->getInit();
  if (!FromInit)
    return Error::success();

  ExpectedExpr ToInitOrErr = import(FromInit);
  if (!ToInitOrErr)
    return ToInitOrErr.takeError();

  To->setInit(*ToInitOrErr);
  if (From->isInitKnownICE()) {
    EvaluatedStmt *Eval = To->ensureEvaluatedStmt();
    Eval->CheckedICE = true;
    Eval->IsICE = From->isInitICE();
  }

  // FIXME: Other bits to merge?
  return Error::success();
}

Error ASTNodeImporter::ImportDefinition(
    EnumDecl *From, EnumDecl *To, ImportDefinitionKind Kind) {
  if (To->getDefinition() || To->isBeingDefined()) {
    if (Kind == IDK_Everything)
      return ImportDeclContext(From, /*ForceImport=*/true);
    return Error::success();
  }

  To->startDefinition();

  if (Error Err = setTypedefNameForAnonDecl(From, To, Importer))
    return Err;

  ExpectedType ToTypeOrErr =
      import(Importer.getFromContext().getTypeDeclType(From));
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedType ToPromotionTypeOrErr = import(From->getPromotionType());
  if (!ToPromotionTypeOrErr)
    return ToPromotionTypeOrErr.takeError();

  if (shouldForceImportDeclContext(Kind))
    if (Error Err = ImportDeclContext(From, /*ForceImport=*/true))
      return Err;

  // FIXME: we might need to merge the number of positive or negative bits
  // if the enumerator lists don't match.
  To->completeDefinition(*ToTypeOrErr, *ToPromotionTypeOrErr,
                         From->getNumPositiveBits(),
                         From->getNumNegativeBits());
  return Error::success();
}

Error ASTNodeImporter::ImportTemplateArguments(
    const TemplateArgument *FromArgs, unsigned NumFromArgs,
    SmallVectorImpl<TemplateArgument> &ToArgs) {
  for (unsigned I = 0; I != NumFromArgs; ++I) {
    if (auto ToOrErr = import(FromArgs[I]))
      ToArgs.push_back(*ToOrErr);
    else
      return ToOrErr.takeError();
  }

  return Error::success();
}

// FIXME: Do not forget to remove this and use only 'import'.
Expected<TemplateArgument>
ASTNodeImporter::ImportTemplateArgument(const TemplateArgument &From) {
  return import(From);
}

template <typename InContainerTy>
Error ASTNodeImporter::ImportTemplateArgumentListInfo(
    const InContainerTy &Container, TemplateArgumentListInfo &ToTAInfo) {
  for (const auto &FromLoc : Container) {
    if (auto ToLocOrErr = import(FromLoc))
      ToTAInfo.addArgument(*ToLocOrErr);
    else
      return ToLocOrErr.takeError();
  }
  return Error::success();
}

static StructuralEquivalenceKind
getStructuralEquivalenceKind(const ASTImporter &Importer) {
  return Importer.isMinimalImport() ? StructuralEquivalenceKind::Minimal
                                    : StructuralEquivalenceKind::Default;
}

bool ASTNodeImporter::IsStructuralMatch(Decl *From, Decl *To, bool Complain) {
  StructuralEquivalenceContext Ctx(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), getStructuralEquivalenceKind(Importer),
      false, Complain);
  return Ctx.IsEquivalent(From, To);
}

bool ASTNodeImporter::IsStructuralMatch(RecordDecl *FromRecord,
                                        RecordDecl *ToRecord, bool Complain) {
  // Eliminate a potential failure point where we attempt to re-import
  // something we're trying to import while completing ToRecord.
  Decl *ToOrigin = Importer.GetOriginalDecl(ToRecord);
  if (ToOrigin) {
    auto *ToOriginRecord = dyn_cast<RecordDecl>(ToOrigin);
    if (ToOriginRecord)
      ToRecord = ToOriginRecord;
  }

  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   ToRecord->getASTContext(),
                                   Importer.getNonEquivalentDecls(),
                                   getStructuralEquivalenceKind(Importer),
                                   false, Complain);
  return Ctx.IsEquivalent(FromRecord, ToRecord);
}

bool ASTNodeImporter::IsStructuralMatch(VarDecl *FromVar, VarDecl *ToVar,
                                        bool Complain) {
  StructuralEquivalenceContext Ctx(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), getStructuralEquivalenceKind(Importer),
      false, Complain);
  return Ctx.IsEquivalent(FromVar, ToVar);
}

bool ASTNodeImporter::IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToEnum) {
  // Eliminate a potential failure point where we attempt to re-import
  // something we're trying to import while completing ToEnum.
  if (Decl *ToOrigin = Importer.GetOriginalDecl(ToEnum))
    if (auto *ToOriginEnum = dyn_cast<EnumDecl>(ToOrigin))
        ToEnum = ToOriginEnum;

  StructuralEquivalenceContext Ctx(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), getStructuralEquivalenceKind(Importer));
  return Ctx.IsEquivalent(FromEnum, ToEnum);
}

bool ASTNodeImporter::IsStructuralMatch(FunctionTemplateDecl *From,
                                        FunctionTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), getStructuralEquivalenceKind(Importer),
      false, false);
  return Ctx.IsEquivalent(From, To);
}

bool ASTNodeImporter::IsStructuralMatch(FunctionDecl *From, FunctionDecl *To) {
  StructuralEquivalenceContext Ctx(
      Importer.getFromContext(), Importer.getToContext(),
      Importer.getNonEquivalentDecls(), getStructuralEquivalenceKind(Importer),
      false, false);
  return Ctx.IsEquivalent(From, To);
}

bool ASTNodeImporter::IsStructuralMatch(EnumConstantDecl *FromEC,
                                        EnumConstantDecl *ToEC) {
  const llvm::APSInt &FromVal = FromEC->getInitVal();
  const llvm::APSInt &ToVal = ToEC->getInitVal();

  return FromVal.isSigned() == ToVal.isSigned() &&
         FromVal.getBitWidth() == ToVal.getBitWidth() &&
         FromVal == ToVal;
}

bool ASTNodeImporter::IsStructuralMatch(ClassTemplateDecl *From,
                                        ClassTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls(),
                                   getStructuralEquivalenceKind(Importer));
  return Ctx.IsEquivalent(From, To);
}

bool ASTNodeImporter::IsStructuralMatch(VarTemplateDecl *From,
                                        VarTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls(),
                                   getStructuralEquivalenceKind(Importer));
  return Ctx.IsEquivalent(From, To);
}

ExpectedDecl ASTNodeImporter::VisitDecl(Decl *D) {
  Importer.FromDiag(D->getLocation(), diag::err_unsupported_ast_node)
    << D->getDeclKindName();
  return make_error<ImportError>(ImportError::UnsupportedConstruct);
}

ExpectedDecl ASTNodeImporter::VisitImportDecl(ImportDecl *D) {
  Importer.FromDiag(D->getLocation(), diag::err_unsupported_ast_node)
      << D->getDeclKindName();
  return make_error<ImportError>(ImportError::UnsupportedConstruct);
}

ExpectedDecl ASTNodeImporter::VisitEmptyDecl(EmptyDecl *D) {
  // Import the context of this declaration.
  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  // Import the location of this declaration.
  ExpectedSLoc LocOrErr = import(D->getLocation());
  if (!LocOrErr)
    return LocOrErr.takeError();

  EmptyDecl *ToD;
  if (GetImportedOrCreateDecl(ToD, D, Importer.getToContext(), DC, *LocOrErr))
    return ToD;

  ToD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToD);
  return ToD;
}

ExpectedDecl ASTNodeImporter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  TranslationUnitDecl *ToD =
    Importer.getToContext().getTranslationUnitDecl();

  Importer.MapImported(D, ToD);

  return ToD;
}

ExpectedDecl ASTNodeImporter::VisitAccessSpecDecl(AccessSpecDecl *D) {
  ExpectedSLoc LocOrErr = import(D->getLocation());
  if (!LocOrErr)
    return LocOrErr.takeError();
  auto ColonLocOrErr = import(D->getColonLoc());
  if (!ColonLocOrErr)
    return ColonLocOrErr.takeError();

  // Import the context of this declaration.
  auto DCOrErr = Importer.ImportContext(D->getDeclContext());
  if (!DCOrErr)
    return DCOrErr.takeError();
  DeclContext *DC = *DCOrErr;

  AccessSpecDecl *ToD;
  if (GetImportedOrCreateDecl(ToD, D, Importer.getToContext(), D->getAccess(),
                              DC, *LocOrErr, *ColonLocOrErr))
    return ToD;

  // Lexical DeclContext and Semantic DeclContext
  // is always the same for the accessSpec.
  ToD->setLexicalDeclContext(DC);
  DC->addDeclInternal(ToD);

  return ToD;
}

ExpectedDecl ASTNodeImporter::VisitStaticAssertDecl(StaticAssertDecl *D) {
  auto DCOrErr = Importer.ImportContext(D->getDeclContext());
  if (!DCOrErr)
    return DCOrErr.takeError();
  DeclContext *DC = *DCOrErr;
  DeclContext *LexicalDC = DC;

  SourceLocation ToLocation, ToRParenLoc;
  Expr *ToAssertExpr;
  StringLiteral *ToMessage;
  if (auto Imp = importSeq(
      D->getLocation(), D->getAssertExpr(), D->getMessage(), D->getRParenLoc()))
    std::tie(ToLocation, ToAssertExpr, ToMessage, ToRParenLoc) = *Imp;
  else
    return Imp.takeError();

  StaticAssertDecl *ToD;
  if (GetImportedOrCreateDecl(
      ToD, D, Importer.getToContext(), DC, ToLocation, ToAssertExpr, ToMessage,
      ToRParenLoc, D->isFailed()))
    return ToD;

  ToD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToD);
  return ToD;
}

ExpectedDecl ASTNodeImporter::VisitNamespaceDecl(NamespaceDecl *D) {
  // Import the major distinguishing characteristics of this namespace.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  NamespaceDecl *MergeWithNamespace = nullptr;
  if (!Name) {
    // This is an anonymous namespace. Adopt an existing anonymous
    // namespace if we can.
    // FIXME: Not testable.
    if (auto *TU = dyn_cast<TranslationUnitDecl>(DC))
      MergeWithNamespace = TU->getAnonymousNamespace();
    else
      MergeWithNamespace = cast<NamespaceDecl>(DC)->getAnonymousNamespace();
  } else {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(Decl::IDNS_Namespace))
        continue;

      if (auto *FoundNS = dyn_cast<NamespaceDecl>(FoundDecl)) {
        MergeWithNamespace = FoundNS;
        ConflictingDecls.clear();
        break;
      }

      ConflictingDecls.push_back(FoundDecl);
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, Decl::IDNS_Namespace, ConflictingDecls.data(),
          ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  ExpectedSLoc BeginLocOrErr = import(D->getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();
  ExpectedSLoc RBraceLocOrErr = import(D->getRBraceLoc());
  if (!RBraceLocOrErr)
    return RBraceLocOrErr.takeError();

  // Create the "to" namespace, if needed.
  NamespaceDecl *ToNamespace = MergeWithNamespace;
  if (!ToNamespace) {
    if (GetImportedOrCreateDecl(
            ToNamespace, D, Importer.getToContext(), DC, D->isInline(),
            *BeginLocOrErr, Loc, Name.getAsIdentifierInfo(),
            /*PrevDecl=*/nullptr))
      return ToNamespace;
    ToNamespace->setRBraceLoc(*RBraceLocOrErr);
    ToNamespace->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToNamespace);

    // If this is an anonymous namespace, register it as the anonymous
    // namespace within its context.
    if (!Name) {
      if (auto *TU = dyn_cast<TranslationUnitDecl>(DC))
        TU->setAnonymousNamespace(ToNamespace);
      else
        cast<NamespaceDecl>(DC)->setAnonymousNamespace(ToNamespace);
    }
  }
  Importer.MapImported(D, ToNamespace);

  if (Error Err = ImportDeclContext(D))
    return std::move(Err);

  return ToNamespace;
}

ExpectedDecl ASTNodeImporter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  // Import the major distinguishing characteristics of this namespace.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *LookupD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, LookupD, Loc))
    return std::move(Err);
  if (LookupD)
    return LookupD;

  // NOTE: No conflict resolution is done for namespace aliases now.

  SourceLocation ToNamespaceLoc, ToAliasLoc, ToTargetNameLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  NamespaceDecl *ToNamespace;
  if (auto Imp = importSeq(
      D->getNamespaceLoc(), D->getAliasLoc(), D->getQualifierLoc(),
      D->getTargetNameLoc(), D->getNamespace()))
    std::tie(
        ToNamespaceLoc, ToAliasLoc, ToQualifierLoc, ToTargetNameLoc,
        ToNamespace) = *Imp;
  else
    return Imp.takeError();
  IdentifierInfo *ToIdentifier = Importer.Import(D->getIdentifier());

  NamespaceAliasDecl *ToD;
  if (GetImportedOrCreateDecl(
      ToD, D, Importer.getToContext(), DC, ToNamespaceLoc, ToAliasLoc,
      ToIdentifier, ToQualifierLoc, ToTargetNameLoc, ToNamespace))
    return ToD;

  ToD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToD);

  return ToD;
}

ExpectedDecl
ASTNodeImporter::VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias) {
  // Import the major distinguishing characteristics of this typedef.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // If this typedef is not in block scope, determine whether we've
  // seen a typedef with the same name (that we can merge with) or any
  // other entity by that name (which name lookup could conflict with).
  // Note: Repeated typedefs are not valid in C99:
  // 'typedef int T; typedef int T;' is invalid
  // We do not care about this now.
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;
      if (auto *FoundTypedef = dyn_cast<TypedefNameDecl>(FoundDecl)) {
        if (!hasSameVisibilityContext(FoundTypedef, D))
          continue;

        QualType FromUT = D->getUnderlyingType();
        QualType FoundUT = FoundTypedef->getUnderlyingType();
        if (Importer.IsStructurallyEquivalent(FromUT, FoundUT)) {
          // If the "From" context has a complete underlying type but we
          // already have a complete underlying type then return with that.
          if (!FromUT->isIncompleteType() && !FoundUT->isIncompleteType())
            return Importer.MapImported(D, FoundTypedef);
          // FIXME Handle redecl chain. When you do that make consistent changes
          // in ASTImporterLookupTable too.
        } else {
          ConflictingDecls.push_back(FoundDecl);
        }
      }
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, IDNS, ConflictingDecls.data(), ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  QualType ToUnderlyingType;
  TypeSourceInfo *ToTypeSourceInfo;
  SourceLocation ToBeginLoc;
  if (auto Imp = importSeq(
      D->getUnderlyingType(), D->getTypeSourceInfo(), D->getBeginLoc()))
    std::tie(ToUnderlyingType, ToTypeSourceInfo, ToBeginLoc) = *Imp;
  else
    return Imp.takeError();

  // Create the new typedef node.
  // FIXME: ToUnderlyingType is not used.
  TypedefNameDecl *ToTypedef;
  if (IsAlias) {
    if (GetImportedOrCreateDecl<TypeAliasDecl>(
        ToTypedef, D, Importer.getToContext(), DC, ToBeginLoc, Loc,
        Name.getAsIdentifierInfo(), ToTypeSourceInfo))
      return ToTypedef;
  } else if (GetImportedOrCreateDecl<TypedefDecl>(
      ToTypedef, D, Importer.getToContext(), DC, ToBeginLoc, Loc,
      Name.getAsIdentifierInfo(), ToTypeSourceInfo))
    return ToTypedef;

  ToTypedef->setAccess(D->getAccess());
  ToTypedef->setLexicalDeclContext(LexicalDC);

  // Templated declarations should not appear in DeclContext.
  TypeAliasDecl *FromAlias = IsAlias ? cast<TypeAliasDecl>(D) : nullptr;
  if (!FromAlias || !FromAlias->getDescribedAliasTemplate())
    LexicalDC->addDeclInternal(ToTypedef);

  return ToTypedef;
}

ExpectedDecl ASTNodeImporter::VisitTypedefDecl(TypedefDecl *D) {
  return VisitTypedefNameDecl(D, /*IsAlias=*/false);
}

ExpectedDecl ASTNodeImporter::VisitTypeAliasDecl(TypeAliasDecl *D) {
  return VisitTypedefNameDecl(D, /*IsAlias=*/true);
}

ExpectedDecl
ASTNodeImporter::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  // Import the major distinguishing characteristics of this typedef.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *FoundD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, FoundD, Loc))
    return std::move(Err);
  if (FoundD)
    return FoundD;

  // If this typedef is not in block scope, determine whether we've
  // seen a typedef with the same name (that we can merge with) or any
  // other entity by that name (which name lookup could conflict with).
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;
      if (auto *FoundAlias = dyn_cast<TypeAliasTemplateDecl>(FoundDecl))
        return Importer.MapImported(D, FoundAlias);
      ConflictingDecls.push_back(FoundDecl);
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, IDNS, ConflictingDecls.data(), ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  TemplateParameterList *ToTemplateParameters;
  TypeAliasDecl *ToTemplatedDecl;
  if (auto Imp = importSeq(D->getTemplateParameters(), D->getTemplatedDecl()))
    std::tie(ToTemplateParameters, ToTemplatedDecl) = *Imp;
  else
    return Imp.takeError();

  TypeAliasTemplateDecl *ToAlias;
  if (GetImportedOrCreateDecl(ToAlias, D, Importer.getToContext(), DC, Loc,
                              Name, ToTemplateParameters, ToTemplatedDecl))
    return ToAlias;

  ToTemplatedDecl->setDescribedAliasTemplate(ToAlias);

  ToAlias->setAccess(D->getAccess());
  ToAlias->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToAlias);
  return ToAlias;
}

ExpectedDecl ASTNodeImporter::VisitLabelDecl(LabelDecl *D) {
  // Import the major distinguishing characteristics of this label.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  assert(LexicalDC->isFunctionOrMethod());

  LabelDecl *ToLabel;
  if (D->isGnuLocal()) {
    ExpectedSLoc BeginLocOrErr = import(D->getBeginLoc());
    if (!BeginLocOrErr)
      return BeginLocOrErr.takeError();
    if (GetImportedOrCreateDecl(ToLabel, D, Importer.getToContext(), DC, Loc,
                                Name.getAsIdentifierInfo(), *BeginLocOrErr))
      return ToLabel;

  } else {
    if (GetImportedOrCreateDecl(ToLabel, D, Importer.getToContext(), DC, Loc,
                                Name.getAsIdentifierInfo()))
      return ToLabel;

  }

  Expected<LabelStmt *> ToStmtOrErr = import(D->getStmt());
  if (!ToStmtOrErr)
    return ToStmtOrErr.takeError();

  ToLabel->setStmt(*ToStmtOrErr);
  ToLabel->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToLabel);
  return ToLabel;
}

ExpectedDecl ASTNodeImporter::VisitEnumDecl(EnumDecl *D) {
  // Import the major distinguishing characteristics of this enum.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Figure out what enum name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    if (Error Err = importInto(
        SearchName, D->getTypedefNameForAnonDecl()->getDeclName()))
      return std::move(Err);
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOpts().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary;

  // We may already have an enum of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod() && SearchName) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    auto FoundDecls =
        Importer.findDeclsInToCtx(DC, SearchName);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;

      if (auto *Typedef = dyn_cast<TypedefNameDecl>(FoundDecl)) {
        if (const auto *Tag = Typedef->getUnderlyingType()->getAs<TagType>())
          FoundDecl = Tag->getDecl();
      }

      if (auto *FoundEnum = dyn_cast<EnumDecl>(FoundDecl)) {
        if (!hasSameVisibilityContext(FoundEnum, D))
          continue;
        if (IsStructuralMatch(D, FoundEnum))
          return Importer.MapImported(D, FoundEnum);
        ConflictingDecls.push_back(FoundDecl);
      }
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          SearchName, DC, IDNS, ConflictingDecls.data(),
          ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  SourceLocation ToBeginLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  QualType ToIntegerType;
  SourceRange ToBraceRange;
  if (auto Imp = importSeq(D->getBeginLoc(), D->getQualifierLoc(),
                           D->getIntegerType(), D->getBraceRange()))
    std::tie(ToBeginLoc, ToQualifierLoc, ToIntegerType, ToBraceRange) = *Imp;
  else
    return Imp.takeError();

  // Create the enum declaration.
  EnumDecl *D2;
  if (GetImportedOrCreateDecl(
          D2, D, Importer.getToContext(), DC, ToBeginLoc,
          Loc, Name.getAsIdentifierInfo(), nullptr, D->isScoped(),
          D->isScopedUsingClassTag(), D->isFixed()))
    return D2;

  D2->setQualifierInfo(ToQualifierLoc);
  D2->setIntegerType(ToIntegerType);
  D2->setBraceRange(ToBraceRange);
  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);

  // Import the definition
  if (D->isCompleteDefinition())
    if (Error Err = ImportDefinition(D, D2))
      return std::move(Err);

  return D2;
}

ExpectedDecl ASTNodeImporter::VisitRecordDecl(RecordDecl *D) {
  bool IsFriendTemplate = false;
  if (auto *DCXX = dyn_cast<CXXRecordDecl>(D)) {
    IsFriendTemplate =
        DCXX->getDescribedClassTemplate() &&
        DCXX->getDescribedClassTemplate()->getFriendObjectKind() !=
            Decl::FOK_None;
  }

  // Import the major distinguishing characteristics of this record.
  DeclContext *DC = nullptr, *LexicalDC = nullptr;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD = nullptr;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Figure out what structure name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    if (Error Err = importInto(
        SearchName, D->getTypedefNameForAnonDecl()->getDeclName()))
      return std::move(Err);
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOpts().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary | Decl::IDNS_TagFriend;

  // We may already have a record of the same name; try to find and match it.
  RecordDecl *PrevDecl = nullptr;
  if (!DC->isFunctionOrMethod() && !D->isLambda()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    auto FoundDecls =
        Importer.findDeclsInToCtx(DC, SearchName);
    if (!FoundDecls.empty()) {
      // We're going to have to compare D against potentially conflicting Decls,
      // so complete it.
      if (D->hasExternalLexicalStorage() && !D->isCompleteDefinition())
        D->getASTContext().getExternalSource()->CompleteType(D);
    }

    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;

      Decl *Found = FoundDecl;
      if (auto *Typedef = dyn_cast<TypedefNameDecl>(Found)) {
        if (const auto *Tag = Typedef->getUnderlyingType()->getAs<TagType>())
          Found = Tag->getDecl();
      }

      if (auto *FoundRecord = dyn_cast<RecordDecl>(Found)) {
        // Do not emit false positive diagnostic in case of unnamed
        // struct/union and in case of anonymous structs.  Would be false
        // because there may be several anonymous/unnamed structs in a class.
        // E.g. these are both valid:
        //  struct A { // unnamed structs
        //    struct { struct A *next; } entry0;
        //    struct { struct A *next; } entry1;
        //  };
        //  struct X { struct { int a; }; struct { int b; }; }; // anon structs
        if (!SearchName)
          if (!IsStructuralMatch(D, FoundRecord, false))
            continue;

        if (!hasSameVisibilityContext(FoundRecord, D))
          continue;

        if (IsStructuralMatch(D, FoundRecord)) {
          RecordDecl *FoundDef = FoundRecord->getDefinition();
          if (D->isThisDeclarationADefinition() && FoundDef) {
            // FIXME: Structural equivalence check should check for same
            // user-defined methods.
            Importer.MapImported(D, FoundDef);
            if (const auto *DCXX = dyn_cast<CXXRecordDecl>(D)) {
              auto *FoundCXX = dyn_cast<CXXRecordDecl>(FoundDef);
              assert(FoundCXX && "Record type mismatch");

              if (!Importer.isMinimalImport())
                // FoundDef may not have every implicit method that D has
                // because implicit methods are created only if they are used.
                if (Error Err = ImportImplicitMethods(DCXX, FoundCXX))
                  return std::move(Err);
            }
          }
          PrevDecl = FoundRecord->getMostRecentDecl();
          break;
        }
        ConflictingDecls.push_back(FoundDecl);
      } // kind is RecordDecl
    } // for

    if (!ConflictingDecls.empty() && SearchName) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          SearchName, DC, IDNS, ConflictingDecls.data(),
          ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  ExpectedSLoc BeginLocOrErr = import(D->getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();

  // Create the record declaration.
  RecordDecl *D2 = nullptr;
  CXXRecordDecl *D2CXX = nullptr;
  if (auto *DCXX = dyn_cast<CXXRecordDecl>(D)) {
    if (DCXX->isLambda()) {
      auto TInfoOrErr = import(DCXX->getLambdaTypeInfo());
      if (!TInfoOrErr)
        return TInfoOrErr.takeError();
      if (GetImportedOrCreateSpecialDecl(
              D2CXX, CXXRecordDecl::CreateLambda, D, Importer.getToContext(),
              DC, *TInfoOrErr, Loc, DCXX->isDependentLambda(),
              DCXX->isGenericLambda(), DCXX->getLambdaCaptureDefault()))
        return D2CXX;
      ExpectedDecl CDeclOrErr = import(DCXX->getLambdaContextDecl());
      if (!CDeclOrErr)
        return CDeclOrErr.takeError();
      D2CXX->setLambdaMangling(DCXX->getLambdaManglingNumber(), *CDeclOrErr,
                               DCXX->hasKnownLambdaInternalLinkage());
    } else if (DCXX->isInjectedClassName()) {
      // We have to be careful to do a similar dance to the one in
      // Sema::ActOnStartCXXMemberDeclarations
      const bool DelayTypeCreation = true;
      if (GetImportedOrCreateDecl(
              D2CXX, D, Importer.getToContext(), D->getTagKind(), DC,
              *BeginLocOrErr, Loc, Name.getAsIdentifierInfo(),
              cast_or_null<CXXRecordDecl>(PrevDecl), DelayTypeCreation))
        return D2CXX;
      Importer.getToContext().getTypeDeclType(
          D2CXX, dyn_cast<CXXRecordDecl>(DC));
    } else {
      if (GetImportedOrCreateDecl(D2CXX, D, Importer.getToContext(),
                                  D->getTagKind(), DC, *BeginLocOrErr, Loc,
                                  Name.getAsIdentifierInfo(),
                                  cast_or_null<CXXRecordDecl>(PrevDecl)))
        return D2CXX;
    }

    D2 = D2CXX;
    D2->setAccess(D->getAccess());
    D2->setLexicalDeclContext(LexicalDC);
    addDeclToContexts(D, D2);

    if (ClassTemplateDecl *FromDescribed =
        DCXX->getDescribedClassTemplate()) {
      ClassTemplateDecl *ToDescribed;
      if (Error Err = importInto(ToDescribed, FromDescribed))
        return std::move(Err);
      D2CXX->setDescribedClassTemplate(ToDescribed);
      if (!DCXX->isInjectedClassName() && !IsFriendTemplate) {
        // In a record describing a template the type should be an
        // InjectedClassNameType (see Sema::CheckClassTemplate). Update the
        // previously set type to the correct value here (ToDescribed is not
        // available at record create).
        // FIXME: The previous type is cleared but not removed from
        // ASTContext's internal storage.
        CXXRecordDecl *Injected = nullptr;
        for (NamedDecl *Found : D2CXX->noload_lookup(Name)) {
          auto *Record = dyn_cast<CXXRecordDecl>(Found);
          if (Record && Record->isInjectedClassName()) {
            Injected = Record;
            break;
          }
        }
        // Create an injected type for the whole redecl chain.
        SmallVector<Decl *, 2> Redecls =
            getCanonicalForwardRedeclChain(D2CXX);
        for (auto *R : Redecls) {
          auto *RI = cast<CXXRecordDecl>(R);
          RI->setTypeForDecl(nullptr);
          // Below we create a new injected type and assign that to the
          // canonical decl, subsequent declarations in the chain will reuse
          // that type.
          Importer.getToContext().getInjectedClassNameType(
              RI, ToDescribed->getInjectedClassNameSpecialization());
        }
        // Set the new type for the previous injected decl too.
        if (Injected) {
          Injected->setTypeForDecl(nullptr);
          Importer.getToContext().getTypeDeclType(Injected, D2CXX);
        }
      }
    } else if (MemberSpecializationInfo *MemberInfo =
                   DCXX->getMemberSpecializationInfo()) {
        TemplateSpecializationKind SK =
            MemberInfo->getTemplateSpecializationKind();
        CXXRecordDecl *FromInst = DCXX->getInstantiatedFromMemberClass();

        if (Expected<CXXRecordDecl *> ToInstOrErr = import(FromInst))
          D2CXX->setInstantiationOfMemberClass(*ToInstOrErr, SK);
        else
          return ToInstOrErr.takeError();

        if (ExpectedSLoc POIOrErr =
            import(MemberInfo->getPointOfInstantiation()))
          D2CXX->getMemberSpecializationInfo()->setPointOfInstantiation(
            *POIOrErr);
        else
          return POIOrErr.takeError();
    }

  } else {
    if (GetImportedOrCreateDecl(D2, D, Importer.getToContext(),
                                D->getTagKind(), DC, *BeginLocOrErr, Loc,
                                Name.getAsIdentifierInfo(), PrevDecl))
      return D2;
    D2->setLexicalDeclContext(LexicalDC);
    addDeclToContexts(D, D2);
  }

  if (auto BraceRangeOrErr = import(D->getBraceRange()))
    D2->setBraceRange(*BraceRangeOrErr);
  else
    return BraceRangeOrErr.takeError();
  if (auto QualifierLocOrErr = import(D->getQualifierLoc()))
    D2->setQualifierInfo(*QualifierLocOrErr);
  else
    return QualifierLocOrErr.takeError();

  if (D->isAnonymousStructOrUnion())
    D2->setAnonymousStructOrUnion(true);

  if (D->isCompleteDefinition())
    if (Error Err = ImportDefinition(D, D2, IDK_Default))
      return std::move(Err);

  return D2;
}

ExpectedDecl ASTNodeImporter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  // Import the major distinguishing characteristics of this enumerator.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Determine whether there are any other declarations with the same name and
  // in the same context.
  if (!LexicalDC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;

      if (auto *FoundEnumConstant = dyn_cast<EnumConstantDecl>(FoundDecl)) {
        if (IsStructuralMatch(D, FoundEnumConstant))
          return Importer.MapImported(D, FoundEnumConstant);
        ConflictingDecls.push_back(FoundDecl);
      }
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, IDNS, ConflictingDecls.data(), ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  ExpectedType TypeOrErr = import(D->getType());
  if (!TypeOrErr)
    return TypeOrErr.takeError();

  ExpectedExpr InitOrErr = import(D->getInitExpr());
  if (!InitOrErr)
    return InitOrErr.takeError();

  EnumConstantDecl *ToEnumerator;
  if (GetImportedOrCreateDecl(
          ToEnumerator, D, Importer.getToContext(), cast<EnumDecl>(DC), Loc,
          Name.getAsIdentifierInfo(), *TypeOrErr, *InitOrErr, D->getInitVal()))
    return ToEnumerator;

  ToEnumerator->setAccess(D->getAccess());
  ToEnumerator->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToEnumerator);
  return ToEnumerator;
}

Error ASTNodeImporter::ImportTemplateParameterLists(const DeclaratorDecl *FromD,
                                                    DeclaratorDecl *ToD) {
  unsigned int Num = FromD->getNumTemplateParameterLists();
  if (Num == 0)
    return Error::success();
  SmallVector<TemplateParameterList *, 2> ToTPLists(Num);
  for (unsigned int I = 0; I < Num; ++I)
    if (Expected<TemplateParameterList *> ToTPListOrErr =
            import(FromD->getTemplateParameterList(I)))
      ToTPLists[I] = *ToTPListOrErr;
    else
      return ToTPListOrErr.takeError();
  ToD->setTemplateParameterListsInfo(Importer.ToContext, ToTPLists);
  return Error::success();
}

Error ASTNodeImporter::ImportTemplateInformation(
    FunctionDecl *FromFD, FunctionDecl *ToFD) {
  switch (FromFD->getTemplatedKind()) {
  case FunctionDecl::TK_NonTemplate:
  case FunctionDecl::TK_FunctionTemplate:
    return Error::success();

  case FunctionDecl::TK_MemberSpecialization: {
    TemplateSpecializationKind TSK = FromFD->getTemplateSpecializationKind();

    if (Expected<FunctionDecl *> InstFDOrErr =
        import(FromFD->getInstantiatedFromMemberFunction()))
      ToFD->setInstantiationOfMemberFunction(*InstFDOrErr, TSK);
    else
      return InstFDOrErr.takeError();

    if (ExpectedSLoc POIOrErr = import(
        FromFD->getMemberSpecializationInfo()->getPointOfInstantiation()))
      ToFD->getMemberSpecializationInfo()->setPointOfInstantiation(*POIOrErr);
    else
      return POIOrErr.takeError();

    return Error::success();
  }

  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    auto FunctionAndArgsOrErr =
        ImportFunctionTemplateWithTemplateArgsFromSpecialization(FromFD);
    if (!FunctionAndArgsOrErr)
      return FunctionAndArgsOrErr.takeError();

    TemplateArgumentList *ToTAList = TemplateArgumentList::CreateCopy(
          Importer.getToContext(), std::get<1>(*FunctionAndArgsOrErr));

    auto *FTSInfo = FromFD->getTemplateSpecializationInfo();
    TemplateArgumentListInfo ToTAInfo;
    const auto *FromTAArgsAsWritten = FTSInfo->TemplateArgumentsAsWritten;
    if (FromTAArgsAsWritten)
      if (Error Err = ImportTemplateArgumentListInfo(
          *FromTAArgsAsWritten, ToTAInfo))
        return Err;

    ExpectedSLoc POIOrErr = import(FTSInfo->getPointOfInstantiation());
    if (!POIOrErr)
      return POIOrErr.takeError();

    if (Error Err = ImportTemplateParameterLists(FromFD, ToFD))
      return Err;

    TemplateSpecializationKind TSK = FTSInfo->getTemplateSpecializationKind();
    ToFD->setFunctionTemplateSpecialization(
        std::get<0>(*FunctionAndArgsOrErr), ToTAList, /* InsertPos= */ nullptr,
        TSK, FromTAArgsAsWritten ? &ToTAInfo : nullptr, *POIOrErr);
    return Error::success();
  }

  case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
    auto *FromInfo = FromFD->getDependentSpecializationInfo();
    UnresolvedSet<8> TemplDecls;
    unsigned NumTemplates = FromInfo->getNumTemplates();
    for (unsigned I = 0; I < NumTemplates; I++) {
      if (Expected<FunctionTemplateDecl *> ToFTDOrErr =
          import(FromInfo->getTemplate(I)))
        TemplDecls.addDecl(*ToFTDOrErr);
      else
        return ToFTDOrErr.takeError();
    }

    // Import TemplateArgumentListInfo.
    TemplateArgumentListInfo ToTAInfo;
    if (Error Err = ImportTemplateArgumentListInfo(
        FromInfo->getLAngleLoc(), FromInfo->getRAngleLoc(),
        llvm::makeArrayRef(
            FromInfo->getTemplateArgs(), FromInfo->getNumTemplateArgs()),
        ToTAInfo))
      return Err;

    ToFD->setDependentTemplateSpecialization(Importer.getToContext(),
                                             TemplDecls, ToTAInfo);
    return Error::success();
  }
  }
  llvm_unreachable("All cases should be covered!");
}

Expected<FunctionDecl *>
ASTNodeImporter::FindFunctionTemplateSpecialization(FunctionDecl *FromFD) {
  auto FunctionAndArgsOrErr =
      ImportFunctionTemplateWithTemplateArgsFromSpecialization(FromFD);
  if (!FunctionAndArgsOrErr)
    return FunctionAndArgsOrErr.takeError();

  FunctionTemplateDecl *Template;
  TemplateArgsTy ToTemplArgs;
  std::tie(Template, ToTemplArgs) = *FunctionAndArgsOrErr;
  void *InsertPos = nullptr;
  auto *FoundSpec = Template->findSpecialization(ToTemplArgs, InsertPos);
  return FoundSpec;
}

Error ASTNodeImporter::ImportFunctionDeclBody(FunctionDecl *FromFD,
                                              FunctionDecl *ToFD) {
  if (Stmt *FromBody = FromFD->getBody()) {
    if (ExpectedStmt ToBodyOrErr = import(FromBody))
      ToFD->setBody(*ToBodyOrErr);
    else
      return ToBodyOrErr.takeError();
  }
  return Error::success();
}

// Returns true if the given D has a DeclContext up to the TranslationUnitDecl
// which is equal to the given DC.
bool isAncestorDeclContextOf(const DeclContext *DC, const Decl *D) {
  const DeclContext *DCi = D->getDeclContext();
  while (DCi != D->getTranslationUnitDecl()) {
    if (DCi == DC)
      return true;
    DCi = DCi->getParent();
  }
  return false;
}

bool ASTNodeImporter::hasAutoReturnTypeDeclaredInside(FunctionDecl *D) {
  QualType FromTy = D->getType();
  const FunctionProtoType *FromFPT = FromTy->getAs<FunctionProtoType>();
  assert(FromFPT && "Must be called on FunctionProtoType");
  if (AutoType *AutoT = FromFPT->getReturnType()->getContainedAutoType()) {
    QualType DeducedT = AutoT->getDeducedType();
    if (const RecordType *RecordT =
            DeducedT.isNull() ? nullptr : dyn_cast<RecordType>(DeducedT)) {
      RecordDecl *RD = RecordT->getDecl();
      assert(RD);
      if (isAncestorDeclContextOf(D, RD)) {
        assert(RD->getLexicalDeclContext() == RD->getDeclContext());
        return true;
      }
    }
  }
  if (const TypedefType *TypedefT =
          dyn_cast<TypedefType>(FromFPT->getReturnType())) {
    TypedefNameDecl *TD = TypedefT->getDecl();
    assert(TD);
    if (isAncestorDeclContextOf(D, TD)) {
      assert(TD->getLexicalDeclContext() == TD->getDeclContext());
      return true;
    }
  }
  return false;
}

ExpectedDecl ASTNodeImporter::VisitFunctionDecl(FunctionDecl *D) {

  SmallVector<Decl *, 2> Redecls = getCanonicalForwardRedeclChain(D);
  auto RedeclIt = Redecls.begin();
  // Import the first part of the decl chain. I.e. import all previous
  // declarations starting from the canonical decl.
  for (; RedeclIt != Redecls.end() && *RedeclIt != D; ++RedeclIt) {
    ExpectedDecl ToRedeclOrErr = import(*RedeclIt);
    if (!ToRedeclOrErr)
      return ToRedeclOrErr.takeError();
  }
  assert(*RedeclIt == D);

  // Import the major distinguishing characteristics of this function.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  FunctionDecl *FoundByLookup = nullptr;
  FunctionTemplateDecl *FromFT = D->getDescribedFunctionTemplate();

  // If this is a function template specialization, then try to find the same
  // existing specialization in the "to" context. The lookup below will not
  // find any specialization, but would find the primary template; thus, we
  // have to skip normal lookup in case of specializations.
  // FIXME handle member function templates (TK_MemberSpecialization) similarly?
  if (D->getTemplatedKind() ==
      FunctionDecl::TK_FunctionTemplateSpecialization) {
    auto FoundFunctionOrErr = FindFunctionTemplateSpecialization(D);
    if (!FoundFunctionOrErr)
      return FoundFunctionOrErr.takeError();
    if (FunctionDecl *FoundFunction = *FoundFunctionOrErr) {
      if (Decl *Def = FindAndMapDefinition(D, FoundFunction))
        return Def;
      FoundByLookup = FoundFunction;
    }
  }
  // Try to find a function in our own ("to") context with the same name, same
  // type, and in the same context as the function we're importing.
  else if (!LexicalDC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary | Decl::IDNS_OrdinaryFriend;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;

      if (auto *FoundFunction = dyn_cast<FunctionDecl>(FoundDecl)) {
        if (!hasSameVisibilityContext(FoundFunction, D))
          continue;

        if (IsStructuralMatch(D, FoundFunction)) {
          if (Decl *Def = FindAndMapDefinition(D, FoundFunction))
            return Def;
          FoundByLookup = FoundFunction;
          break;
        }
        // FIXME: Check for overloading more carefully, e.g., by boosting
        // Sema::IsOverload out to the AST library.

        // Function overloading is okay in C++.
        if (Importer.getToContext().getLangOpts().CPlusPlus)
          continue;

        // Complain about inconsistent function types.
        Importer.ToDiag(Loc, diag::warn_odr_function_type_inconsistent)
            << Name << D->getType() << FoundFunction->getType();
        Importer.ToDiag(FoundFunction->getLocation(), diag::note_odr_value_here)
            << FoundFunction->getType();
        ConflictingDecls.push_back(FoundDecl);
      }
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, IDNS, ConflictingDecls.data(), ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  // We do not allow more than one in-class declaration of a function. This is
  // because AST clients like VTableBuilder asserts on this. VTableBuilder
  // assumes there is only one in-class declaration. Building a redecl
  // chain would result in more than one in-class declaration for
  // overrides (even if they are part of the same redecl chain inside the
  // derived class.)
  if (FoundByLookup) {
    if (isa<CXXMethodDecl>(FoundByLookup)) {
      if (D->getLexicalDeclContext() == D->getDeclContext()) {
        if (!D->doesThisDeclarationHaveABody()) {
          if (FunctionTemplateDecl *DescribedD =
                  D->getDescribedFunctionTemplate()) {
            // Handle a "templated" function together with its described
            // template. This avoids need for a similar check at import of the
            // described template.
            assert(FoundByLookup->getDescribedFunctionTemplate() &&
                   "Templated function mapped to non-templated?");
            Importer.MapImported(DescribedD,
                                 FoundByLookup->getDescribedFunctionTemplate());
          }
          return Importer.MapImported(D, FoundByLookup);
        } else {
          // Let's continue and build up the redecl chain in this case.
          // FIXME Merge the functions into one decl.
        }
      }
    }
  }

  DeclarationNameInfo NameInfo(Name, Loc);
  // Import additional name location/type info.
  if (Error Err = ImportDeclarationNameLoc(D->getNameInfo(), NameInfo))
    return std::move(Err);

  QualType FromTy = D->getType();
  // Set to true if we do not import the type of the function as is. There are
  // cases when the original type would result in an infinite recursion during
  // the import. To avoid an infinite recursion when importing, we create the
  // FunctionDecl with a simplified function type and update it only after the
  // relevant AST nodes are already imported.
  bool UsedDifferentProtoType = false;
  if (const auto *FromFPT = FromTy->getAs<FunctionProtoType>()) {
    QualType FromReturnTy = FromFPT->getReturnType();
    // Functions with auto return type may define a struct inside their body
    // and the return type could refer to that struct.
    // E.g.: auto foo() { struct X{}; return X(); }
    // To avoid an infinite recursion when importing, create the FunctionDecl
    // with a simplified return type.
    if (hasAutoReturnTypeDeclaredInside(D)) {
      FromReturnTy = Importer.getFromContext().VoidTy;
      UsedDifferentProtoType = true;
    }
    FunctionProtoType::ExtProtoInfo FromEPI = FromFPT->getExtProtoInfo();
    // FunctionProtoType::ExtProtoInfo's ExceptionSpecDecl can point to the
    // FunctionDecl that we are importing the FunctionProtoType for.
    // To avoid an infinite recursion when importing, create the FunctionDecl
    // with a simplified function type.
    if (FromEPI.ExceptionSpec.SourceDecl ||
        FromEPI.ExceptionSpec.SourceTemplate ||
        FromEPI.ExceptionSpec.NoexceptExpr) {
      FunctionProtoType::ExtProtoInfo DefaultEPI;
      FromEPI = DefaultEPI;
      UsedDifferentProtoType = true;
    }
    FromTy = Importer.getFromContext().getFunctionType(
        FromReturnTy, FromFPT->getParamTypes(), FromEPI);
  }

  QualType T;
  TypeSourceInfo *TInfo;
  SourceLocation ToInnerLocStart, ToEndLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  if (auto Imp = importSeq(
      FromTy, D->getTypeSourceInfo(), D->getInnerLocStart(),
      D->getQualifierLoc(), D->getEndLoc()))
    std::tie(T, TInfo, ToInnerLocStart, ToQualifierLoc, ToEndLoc) = *Imp;
  else
    return Imp.takeError();

  // Import the function parameters.
  SmallVector<ParmVarDecl *, 8> Parameters;
  for (auto P : D->parameters()) {
    if (Expected<ParmVarDecl *> ToPOrErr = import(P))
      Parameters.push_back(*ToPOrErr);
    else
      return ToPOrErr.takeError();
  }

  // Create the imported function.
  FunctionDecl *ToFunction = nullptr;
  if (auto *FromConstructor = dyn_cast<CXXConstructorDecl>(D)) {
    Expr *ExplicitExpr = nullptr;
    if (FromConstructor->getExplicitSpecifier().getExpr()) {
      auto Imp = importSeq(FromConstructor->getExplicitSpecifier().getExpr());
      if (!Imp)
        return Imp.takeError();
      std::tie(ExplicitExpr) = *Imp;
    }
    if (GetImportedOrCreateDecl<CXXConstructorDecl>(
            ToFunction, D, Importer.getToContext(), cast<CXXRecordDecl>(DC),
            ToInnerLocStart, NameInfo, T, TInfo,
            ExplicitSpecifier(
                ExplicitExpr,
                FromConstructor->getExplicitSpecifier().getKind()),
            D->isInlineSpecified(), D->isImplicit(), D->getConstexprKind()))
      return ToFunction;
  } else if (CXXDestructorDecl *FromDtor = dyn_cast<CXXDestructorDecl>(D)) {

    auto Imp =
        importSeq(const_cast<FunctionDecl *>(FromDtor->getOperatorDelete()),
                  FromDtor->getOperatorDeleteThisArg());

    if (!Imp)
      return Imp.takeError();

    FunctionDecl *ToOperatorDelete;
    Expr *ToThisArg;
    std::tie(ToOperatorDelete, ToThisArg) = *Imp;

    if (GetImportedOrCreateDecl<CXXDestructorDecl>(
        ToFunction, D, Importer.getToContext(), cast<CXXRecordDecl>(DC),
        ToInnerLocStart, NameInfo, T, TInfo, D->isInlineSpecified(),
        D->isImplicit(), D->getConstexprKind()))
      return ToFunction;

    CXXDestructorDecl *ToDtor = cast<CXXDestructorDecl>(ToFunction);

    ToDtor->setOperatorDelete(ToOperatorDelete, ToThisArg);
  } else if (CXXConversionDecl *FromConversion =
                 dyn_cast<CXXConversionDecl>(D)) {
    Expr *ExplicitExpr = nullptr;
    if (FromConversion->getExplicitSpecifier().getExpr()) {
      auto Imp = importSeq(FromConversion->getExplicitSpecifier().getExpr());
      if (!Imp)
        return Imp.takeError();
      std::tie(ExplicitExpr) = *Imp;
    }
    if (GetImportedOrCreateDecl<CXXConversionDecl>(
            ToFunction, D, Importer.getToContext(), cast<CXXRecordDecl>(DC),
            ToInnerLocStart, NameInfo, T, TInfo, D->isInlineSpecified(),
            ExplicitSpecifier(ExplicitExpr,
                              FromConversion->getExplicitSpecifier().getKind()),
            D->getConstexprKind(), SourceLocation()))
      return ToFunction;
  } else if (auto *Method = dyn_cast<CXXMethodDecl>(D)) {
    if (GetImportedOrCreateDecl<CXXMethodDecl>(
            ToFunction, D, Importer.getToContext(), cast<CXXRecordDecl>(DC),
            ToInnerLocStart, NameInfo, T, TInfo, Method->getStorageClass(),
            Method->isInlineSpecified(), D->getConstexprKind(),
            SourceLocation()))
      return ToFunction;
  } else {
    if (GetImportedOrCreateDecl(
            ToFunction, D, Importer.getToContext(), DC, ToInnerLocStart,
            NameInfo, T, TInfo, D->getStorageClass(), D->isInlineSpecified(),
            D->hasWrittenPrototype(), D->getConstexprKind()))
      return ToFunction;
  }

  // Connect the redecl chain.
  if (FoundByLookup) {
    auto *Recent = const_cast<FunctionDecl *>(
          FoundByLookup->getMostRecentDecl());
    ToFunction->setPreviousDecl(Recent);
    // FIXME Probably we should merge exception specifications.  E.g. In the
    // "To" context the existing function may have exception specification with
    // noexcept-unevaluated, while the newly imported function may have an
    // evaluated noexcept.  A call to adjustExceptionSpec() on the imported
    // decl and its redeclarations may be required.
  }

  ToFunction->setQualifierInfo(ToQualifierLoc);
  ToFunction->setAccess(D->getAccess());
  ToFunction->setLexicalDeclContext(LexicalDC);
  ToFunction->setVirtualAsWritten(D->isVirtualAsWritten());
  ToFunction->setTrivial(D->isTrivial());
  ToFunction->setPure(D->isPure());
  ToFunction->setDefaulted(D->isDefaulted());
  ToFunction->setExplicitlyDefaulted(D->isExplicitlyDefaulted());
  ToFunction->setDeletedAsWritten(D->isDeletedAsWritten());
  ToFunction->setRangeEnd(ToEndLoc);

  // Set the parameters.
  for (auto *Param : Parameters) {
    Param->setOwningFunction(ToFunction);
    ToFunction->addDeclInternal(Param);
  }
  ToFunction->setParams(Parameters);

  // We need to complete creation of FunctionProtoTypeLoc manually with setting
  // params it refers to.
  if (TInfo) {
    if (auto ProtoLoc =
        TInfo->getTypeLoc().IgnoreParens().getAs<FunctionProtoTypeLoc>()) {
      for (unsigned I = 0, N = Parameters.size(); I != N; ++I)
        ProtoLoc.setParam(I, Parameters[I]);
    }
  }

  // Import the describing template function, if any.
  if (FromFT) {
    auto ToFTOrErr = import(FromFT);
    if (!ToFTOrErr)
      return ToFTOrErr.takeError();
  }

  // Import Ctor initializers.
  if (auto *FromConstructor = dyn_cast<CXXConstructorDecl>(D)) {
    if (unsigned NumInitializers = FromConstructor->getNumCtorInitializers()) {
      SmallVector<CXXCtorInitializer *, 4> CtorInitializers(NumInitializers);
      // Import first, then allocate memory and copy if there was no error.
      if (Error Err = ImportContainerChecked(
          FromConstructor->inits(), CtorInitializers))
        return std::move(Err);
      auto **Memory =
          new (Importer.getToContext()) CXXCtorInitializer *[NumInitializers];
      std::copy(CtorInitializers.begin(), CtorInitializers.end(), Memory);
      auto *ToCtor = cast<CXXConstructorDecl>(ToFunction);
      ToCtor->setCtorInitializers(Memory);
      ToCtor->setNumCtorInitializers(NumInitializers);
    }
  }

  if (D->doesThisDeclarationHaveABody()) {
    Error Err = ImportFunctionDeclBody(D, ToFunction);

    if (Err)
      return std::move(Err);
  }

  // Import and set the original type in case we used another type.
  if (UsedDifferentProtoType) {
    if (ExpectedType TyOrErr = import(D->getType()))
      ToFunction->setType(*TyOrErr);
    else
      return TyOrErr.takeError();
  }

  // FIXME: Other bits to merge?

  // If it is a template, import all related things.
  if (Error Err = ImportTemplateInformation(D, ToFunction))
    return std::move(Err);

  addDeclToContexts(D, ToFunction);

  if (auto *FromCXXMethod = dyn_cast<CXXMethodDecl>(D))
    if (Error Err = ImportOverriddenMethods(cast<CXXMethodDecl>(ToFunction),
                                            FromCXXMethod))
      return std::move(Err);

  // Import the rest of the chain. I.e. import all subsequent declarations.
  for (++RedeclIt; RedeclIt != Redecls.end(); ++RedeclIt) {
    ExpectedDecl ToRedeclOrErr = import(*RedeclIt);
    if (!ToRedeclOrErr)
      return ToRedeclOrErr.takeError();
  }

  return ToFunction;
}

ExpectedDecl ASTNodeImporter::VisitCXXMethodDecl(CXXMethodDecl *D) {
  return VisitFunctionDecl(D);
}

ExpectedDecl ASTNodeImporter::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

ExpectedDecl ASTNodeImporter::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

ExpectedDecl ASTNodeImporter::VisitCXXConversionDecl(CXXConversionDecl *D) {
  return VisitCXXMethodDecl(D);
}

ExpectedDecl ASTNodeImporter::VisitFieldDecl(FieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Determine whether we've already imported this field.
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (FieldDecl *FoundField = dyn_cast<FieldDecl>(FoundDecl)) {
      // For anonymous fields, match up by index.
      if (!Name &&
          ASTImporter::getFieldIndex(D) !=
          ASTImporter::getFieldIndex(FoundField))
        continue;

      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundField->getType())) {
        Importer.MapImported(D, FoundField);
        // In case of a FieldDecl of a ClassTemplateSpecializationDecl, the
        // initializer of a FieldDecl might not had been instantiated in the
        // "To" context.  However, the "From" context might instantiated that,
        // thus we have to merge that.
        if (Expr *FromInitializer = D->getInClassInitializer()) {
          // We don't have yet the initializer set.
          if (FoundField->hasInClassInitializer() &&
              !FoundField->getInClassInitializer()) {
            if (ExpectedExpr ToInitializerOrErr = import(FromInitializer))
              FoundField->setInClassInitializer(*ToInitializerOrErr);
            else {
              // We can't return error here,
              // since we already mapped D as imported.
              // FIXME: warning message?
              consumeError(ToInitializerOrErr.takeError());
              return FoundField;
            }
          }
        }
        return FoundField;
      }

      // FIXME: Why is this case not handled with calling HandleNameConflict?
      Importer.ToDiag(Loc, diag::warn_odr_field_type_inconsistent)
        << Name << D->getType() << FoundField->getType();
      Importer.ToDiag(FoundField->getLocation(), diag::note_odr_value_here)
        << FoundField->getType();

      return make_error<ImportError>(ImportError::NameConflict);
    }
  }

  QualType ToType;
  TypeSourceInfo *ToTInfo;
  Expr *ToBitWidth;
  SourceLocation ToInnerLocStart;
  Expr *ToInitializer;
  if (auto Imp = importSeq(
      D->getType(), D->getTypeSourceInfo(), D->getBitWidth(),
      D->getInnerLocStart(), D->getInClassInitializer()))
    std::tie(
        ToType, ToTInfo, ToBitWidth, ToInnerLocStart, ToInitializer) = *Imp;
  else
    return Imp.takeError();

  FieldDecl *ToField;
  if (GetImportedOrCreateDecl(ToField, D, Importer.getToContext(), DC,
                              ToInnerLocStart, Loc, Name.getAsIdentifierInfo(),
                              ToType, ToTInfo, ToBitWidth, D->isMutable(),
                              D->getInClassInitStyle()))
    return ToField;

  ToField->setAccess(D->getAccess());
  ToField->setLexicalDeclContext(LexicalDC);
  if (ToInitializer)
    ToField->setInClassInitializer(ToInitializer);
  ToField->setImplicit(D->isImplicit());
  LexicalDC->addDeclInternal(ToField);
  return ToField;
}

ExpectedDecl ASTNodeImporter::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Determine whether we've already imported this field.
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (auto *FoundField = dyn_cast<IndirectFieldDecl>(FoundDecls[I])) {
      // For anonymous indirect fields, match up by index.
      if (!Name &&
          ASTImporter::getFieldIndex(D) !=
          ASTImporter::getFieldIndex(FoundField))
        continue;

      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundField->getType(),
                                            !Name.isEmpty())) {
        Importer.MapImported(D, FoundField);
        return FoundField;
      }

      // If there are more anonymous fields to check, continue.
      if (!Name && I < N-1)
        continue;

      // FIXME: Why is this case not handled with calling HandleNameConflict?
      Importer.ToDiag(Loc, diag::warn_odr_field_type_inconsistent)
        << Name << D->getType() << FoundField->getType();
      Importer.ToDiag(FoundField->getLocation(), diag::note_odr_value_here)
        << FoundField->getType();

      return make_error<ImportError>(ImportError::NameConflict);
    }
  }

  // Import the type.
  auto TypeOrErr = import(D->getType());
  if (!TypeOrErr)
    return TypeOrErr.takeError();

  auto **NamedChain =
    new (Importer.getToContext()) NamedDecl*[D->getChainingSize()];

  unsigned i = 0;
  for (auto *PI : D->chain())
    if (Expected<NamedDecl *> ToD = import(PI))
      NamedChain[i++] = *ToD;
    else
      return ToD.takeError();

  llvm::MutableArrayRef<NamedDecl *> CH = {NamedChain, D->getChainingSize()};
  IndirectFieldDecl *ToIndirectField;
  if (GetImportedOrCreateDecl(ToIndirectField, D, Importer.getToContext(), DC,
                              Loc, Name.getAsIdentifierInfo(), *TypeOrErr, CH))
    // FIXME here we leak `NamedChain` which is allocated before
    return ToIndirectField;

  ToIndirectField->setAccess(D->getAccess());
  ToIndirectField->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToIndirectField);
  return ToIndirectField;
}

ExpectedDecl ASTNodeImporter::VisitFriendDecl(FriendDecl *D) {
  // Import the major distinguishing characteristics of a declaration.
  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  // Determine whether we've already imported this decl.
  // FriendDecl is not a NamedDecl so we cannot use lookup.
  auto *RD = cast<CXXRecordDecl>(DC);
  FriendDecl *ImportedFriend = RD->getFirstFriend();

  while (ImportedFriend) {
    if (D->getFriendDecl() && ImportedFriend->getFriendDecl()) {
      if (IsStructuralMatch(D->getFriendDecl(), ImportedFriend->getFriendDecl(),
                            /*Complain=*/false))
        return Importer.MapImported(D, ImportedFriend);

    } else if (D->getFriendType() && ImportedFriend->getFriendType()) {
      if (Importer.IsStructurallyEquivalent(
            D->getFriendType()->getType(),
            ImportedFriend->getFriendType()->getType(), true))
        return Importer.MapImported(D, ImportedFriend);
    }
    ImportedFriend = ImportedFriend->getNextFriend();
  }

  // Not found. Create it.
  FriendDecl::FriendUnion ToFU;
  if (NamedDecl *FriendD = D->getFriendDecl()) {
    NamedDecl *ToFriendD;
    if (Error Err = importInto(ToFriendD, FriendD))
      return std::move(Err);

    if (FriendD->getFriendObjectKind() != Decl::FOK_None &&
        !(FriendD->isInIdentifierNamespace(Decl::IDNS_NonMemberOperator)))
      ToFriendD->setObjectOfFriendDecl(false);

    ToFU = ToFriendD;
  } else { // The friend is a type, not a decl.
    if (auto TSIOrErr = import(D->getFriendType()))
      ToFU = *TSIOrErr;
    else
      return TSIOrErr.takeError();
  }

  SmallVector<TemplateParameterList *, 1> ToTPLists(D->NumTPLists);
  auto **FromTPLists = D->getTrailingObjects<TemplateParameterList *>();
  for (unsigned I = 0; I < D->NumTPLists; I++) {
    if (auto ListOrErr = import(FromTPLists[I]))
      ToTPLists[I] = *ListOrErr;
    else
      return ListOrErr.takeError();
  }

  auto LocationOrErr = import(D->getLocation());
  if (!LocationOrErr)
    return LocationOrErr.takeError();
  auto FriendLocOrErr = import(D->getFriendLoc());
  if (!FriendLocOrErr)
    return FriendLocOrErr.takeError();

  FriendDecl *FrD;
  if (GetImportedOrCreateDecl(FrD, D, Importer.getToContext(), DC,
                              *LocationOrErr, ToFU,
                              *FriendLocOrErr, ToTPLists))
    return FrD;

  FrD->setAccess(D->getAccess());
  FrD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(FrD);
  return FrD;
}

ExpectedDecl ASTNodeImporter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  // Import the major distinguishing characteristics of an ivar.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Determine whether we've already imported this ivar
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (ObjCIvarDecl *FoundIvar = dyn_cast<ObjCIvarDecl>(FoundDecl)) {
      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundIvar->getType())) {
        Importer.MapImported(D, FoundIvar);
        return FoundIvar;
      }

      Importer.ToDiag(Loc, diag::warn_odr_ivar_type_inconsistent)
        << Name << D->getType() << FoundIvar->getType();
      Importer.ToDiag(FoundIvar->getLocation(), diag::note_odr_value_here)
        << FoundIvar->getType();

      return make_error<ImportError>(ImportError::NameConflict);
    }
  }

  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  Expr *ToBitWidth;
  SourceLocation ToInnerLocStart;
  if (auto Imp = importSeq(
      D->getType(), D->getTypeSourceInfo(), D->getBitWidth(), D->getInnerLocStart()))
    std::tie(ToType, ToTypeSourceInfo, ToBitWidth, ToInnerLocStart) = *Imp;
  else
    return Imp.takeError();

  ObjCIvarDecl *ToIvar;
  if (GetImportedOrCreateDecl(
          ToIvar, D, Importer.getToContext(), cast<ObjCContainerDecl>(DC),
          ToInnerLocStart, Loc, Name.getAsIdentifierInfo(),
          ToType, ToTypeSourceInfo,
          D->getAccessControl(),ToBitWidth, D->getSynthesize()))
    return ToIvar;

  ToIvar->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToIvar);
  return ToIvar;
}

ExpectedDecl ASTNodeImporter::VisitVarDecl(VarDecl *D) {

  SmallVector<Decl*, 2> Redecls = getCanonicalForwardRedeclChain(D);
  auto RedeclIt = Redecls.begin();
  // Import the first part of the decl chain. I.e. import all previous
  // declarations starting from the canonical decl.
  for (; RedeclIt != Redecls.end() && *RedeclIt != D; ++RedeclIt) {
    ExpectedDecl RedeclOrErr = import(*RedeclIt);
    if (!RedeclOrErr)
      return RedeclOrErr.takeError();
  }
  assert(*RedeclIt == D);

  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Try to find a variable in our own ("to") context with the same name and
  // in the same context as the variable we're importing.
  VarDecl *FoundByLookup = nullptr;
  if (D->isFileVarDecl()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;

      if (auto *FoundVar = dyn_cast<VarDecl>(FoundDecl)) {
        if (!hasSameVisibilityContext(FoundVar, D))
          continue;
        if (Importer.IsStructurallyEquivalent(D->getType(),
                                              FoundVar->getType())) {

          // The VarDecl in the "From" context has a definition, but in the
          // "To" context we already have a definition.
          VarDecl *FoundDef = FoundVar->getDefinition();
          if (D->isThisDeclarationADefinition() && FoundDef)
            // FIXME Check for ODR error if the two definitions have
            // different initializers?
            return Importer.MapImported(D, FoundDef);

          // The VarDecl in the "From" context has an initializer, but in the
          // "To" context we already have an initializer.
          const VarDecl *FoundDInit = nullptr;
          if (D->getInit() && FoundVar->getAnyInitializer(FoundDInit))
            // FIXME Diagnose ODR error if the two initializers are different?
            return Importer.MapImported(D, const_cast<VarDecl*>(FoundDInit));

          FoundByLookup = FoundVar;
          break;
        }

        const ArrayType *FoundArray
          = Importer.getToContext().getAsArrayType(FoundVar->getType());
        const ArrayType *TArray
          = Importer.getToContext().getAsArrayType(D->getType());
        if (FoundArray && TArray) {
          if (isa<IncompleteArrayType>(FoundArray) &&
              isa<ConstantArrayType>(TArray)) {
            // Import the type.
            if (auto TyOrErr = import(D->getType()))
              FoundVar->setType(*TyOrErr);
            else
              return TyOrErr.takeError();

            FoundByLookup = FoundVar;
            break;
          } else if (isa<IncompleteArrayType>(TArray) &&
                     isa<ConstantArrayType>(FoundArray)) {
            FoundByLookup = FoundVar;
            break;
          }
        }

        Importer.ToDiag(Loc, diag::warn_odr_variable_type_inconsistent)
          << Name << D->getType() << FoundVar->getType();
        Importer.ToDiag(FoundVar->getLocation(), diag::note_odr_value_here)
          << FoundVar->getType();
        ConflictingDecls.push_back(FoundDecl);
      }
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, IDNS, ConflictingDecls.data(), ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  SourceLocation ToInnerLocStart;
  NestedNameSpecifierLoc ToQualifierLoc;
  if (auto Imp = importSeq(
      D->getType(), D->getTypeSourceInfo(), D->getInnerLocStart(),
      D->getQualifierLoc()))
    std::tie(ToType, ToTypeSourceInfo, ToInnerLocStart, ToQualifierLoc) = *Imp;
  else
    return Imp.takeError();

  // Create the imported variable.
  VarDecl *ToVar;
  if (GetImportedOrCreateDecl(ToVar, D, Importer.getToContext(), DC,
                              ToInnerLocStart, Loc,
                              Name.getAsIdentifierInfo(),
                              ToType, ToTypeSourceInfo,
                              D->getStorageClass()))
    return ToVar;

  ToVar->setQualifierInfo(ToQualifierLoc);
  ToVar->setAccess(D->getAccess());
  ToVar->setLexicalDeclContext(LexicalDC);

  if (FoundByLookup) {
    auto *Recent = const_cast<VarDecl *>(FoundByLookup->getMostRecentDecl());
    ToVar->setPreviousDecl(Recent);
  }

  if (Error Err = ImportInitializer(D, ToVar))
    return std::move(Err);

  if (D->isConstexpr())
    ToVar->setConstexpr(true);

  addDeclToContexts(D, ToVar);

  // Import the rest of the chain. I.e. import all subsequent declarations.
  for (++RedeclIt; RedeclIt != Redecls.end(); ++RedeclIt) {
    ExpectedDecl RedeclOrErr = import(*RedeclIt);
    if (!RedeclOrErr)
      return RedeclOrErr.takeError();
  }

  return ToVar;
}

ExpectedDecl ASTNodeImporter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();

  DeclarationName ToDeclName;
  SourceLocation ToLocation;
  QualType ToType;
  if (auto Imp = importSeq(D->getDeclName(), D->getLocation(), D->getType()))
    std::tie(ToDeclName, ToLocation, ToType) = *Imp;
  else
    return Imp.takeError();

  // Create the imported parameter.
  ImplicitParamDecl *ToParm = nullptr;
  if (GetImportedOrCreateDecl(ToParm, D, Importer.getToContext(), DC,
                              ToLocation, ToDeclName.getAsIdentifierInfo(),
                              ToType, D->getParameterKind()))
    return ToParm;
  return ToParm;
}

Error ASTNodeImporter::ImportDefaultArgOfParmVarDecl(
    const ParmVarDecl *FromParam, ParmVarDecl *ToParam) {
  ToParam->setHasInheritedDefaultArg(FromParam->hasInheritedDefaultArg());
  ToParam->setKNRPromoted(FromParam->isKNRPromoted());

  if (FromParam->hasUninstantiatedDefaultArg()) {
    if (auto ToDefArgOrErr = import(FromParam->getUninstantiatedDefaultArg()))
      ToParam->setUninstantiatedDefaultArg(*ToDefArgOrErr);
    else
      return ToDefArgOrErr.takeError();
  } else if (FromParam->hasUnparsedDefaultArg()) {
    ToParam->setUnparsedDefaultArg();
  } else if (FromParam->hasDefaultArg()) {
    if (auto ToDefArgOrErr = import(FromParam->getDefaultArg()))
      ToParam->setDefaultArg(*ToDefArgOrErr);
    else
      return ToDefArgOrErr.takeError();
  }

  return Error::success();
}

ExpectedDecl ASTNodeImporter::VisitParmVarDecl(ParmVarDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();

  DeclarationName ToDeclName;
  SourceLocation ToLocation, ToInnerLocStart;
  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  if (auto Imp = importSeq(
      D->getDeclName(), D->getLocation(), D->getType(), D->getInnerLocStart(),
      D->getTypeSourceInfo()))
    std::tie(
        ToDeclName, ToLocation, ToType, ToInnerLocStart,
        ToTypeSourceInfo) = *Imp;
  else
    return Imp.takeError();

  ParmVarDecl *ToParm;
  if (GetImportedOrCreateDecl(ToParm, D, Importer.getToContext(), DC,
                              ToInnerLocStart, ToLocation,
                              ToDeclName.getAsIdentifierInfo(), ToType,
                              ToTypeSourceInfo, D->getStorageClass(),
                              /*DefaultArg*/ nullptr))
    return ToParm;

  // Set the default argument. It should be no problem if it was already done.
  // Do not import the default expression before GetImportedOrCreateDecl call
  // to avoid possible infinite import loop because circular dependency.
  if (Error Err = ImportDefaultArgOfParmVarDecl(D, ToParm))
    return std::move(Err);

  if (D->isObjCMethodParameter()) {
    ToParm->setObjCMethodScopeInfo(D->getFunctionScopeIndex());
    ToParm->setObjCDeclQualifier(D->getObjCDeclQualifier());
  } else {
    ToParm->setScopeInfo(D->getFunctionScopeDepth(),
                         D->getFunctionScopeIndex());
  }

  return ToParm;
}

ExpectedDecl ASTNodeImporter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  // Import the major distinguishing characteristics of a method.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (auto *FoundMethod = dyn_cast<ObjCMethodDecl>(FoundDecl)) {
      if (FoundMethod->isInstanceMethod() != D->isInstanceMethod())
        continue;

      // Check return types.
      if (!Importer.IsStructurallyEquivalent(D->getReturnType(),
                                             FoundMethod->getReturnType())) {
        Importer.ToDiag(Loc, diag::warn_odr_objc_method_result_type_inconsistent)
            << D->isInstanceMethod() << Name << D->getReturnType()
            << FoundMethod->getReturnType();
        Importer.ToDiag(FoundMethod->getLocation(),
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;

        return make_error<ImportError>(ImportError::NameConflict);
      }

      // Check the number of parameters.
      if (D->param_size() != FoundMethod->param_size()) {
        Importer.ToDiag(Loc, diag::warn_odr_objc_method_num_params_inconsistent)
          << D->isInstanceMethod() << Name
          << D->param_size() << FoundMethod->param_size();
        Importer.ToDiag(FoundMethod->getLocation(),
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;

        return make_error<ImportError>(ImportError::NameConflict);
      }

      // Check parameter types.
      for (ObjCMethodDecl::param_iterator P = D->param_begin(),
             PEnd = D->param_end(), FoundP = FoundMethod->param_begin();
           P != PEnd; ++P, ++FoundP) {
        if (!Importer.IsStructurallyEquivalent((*P)->getType(),
                                               (*FoundP)->getType())) {
          Importer.FromDiag((*P)->getLocation(),
                            diag::warn_odr_objc_method_param_type_inconsistent)
            << D->isInstanceMethod() << Name
            << (*P)->getType() << (*FoundP)->getType();
          Importer.ToDiag((*FoundP)->getLocation(), diag::note_odr_value_here)
            << (*FoundP)->getType();

          return make_error<ImportError>(ImportError::NameConflict);
        }
      }

      // Check variadic/non-variadic.
      // Check the number of parameters.
      if (D->isVariadic() != FoundMethod->isVariadic()) {
        Importer.ToDiag(Loc, diag::warn_odr_objc_method_variadic_inconsistent)
          << D->isInstanceMethod() << Name;
        Importer.ToDiag(FoundMethod->getLocation(),
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;

        return make_error<ImportError>(ImportError::NameConflict);
      }

      // FIXME: Any other bits we need to merge?
      return Importer.MapImported(D, FoundMethod);
    }
  }

  SourceLocation ToEndLoc;
  QualType ToReturnType;
  TypeSourceInfo *ToReturnTypeSourceInfo;
  if (auto Imp = importSeq(
      D->getEndLoc(), D->getReturnType(), D->getReturnTypeSourceInfo()))
    std::tie(ToEndLoc, ToReturnType, ToReturnTypeSourceInfo) = *Imp;
  else
    return Imp.takeError();

  ObjCMethodDecl *ToMethod;
  if (GetImportedOrCreateDecl(
          ToMethod, D, Importer.getToContext(), Loc, ToEndLoc,
          Name.getObjCSelector(), ToReturnType, ToReturnTypeSourceInfo, DC,
          D->isInstanceMethod(), D->isVariadic(), D->isPropertyAccessor(),
          D->isSynthesizedAccessorStub(), D->isImplicit(), D->isDefined(),
          D->getImplementationControl(), D->hasRelatedResultType()))
    return ToMethod;

  // FIXME: When we decide to merge method definitions, we'll need to
  // deal with implicit parameters.

  // Import the parameters
  SmallVector<ParmVarDecl *, 5> ToParams;
  for (auto *FromP : D->parameters()) {
    if (Expected<ParmVarDecl *> ToPOrErr = import(FromP))
      ToParams.push_back(*ToPOrErr);
    else
      return ToPOrErr.takeError();
  }

  // Set the parameters.
  for (auto *ToParam : ToParams) {
    ToParam->setOwningFunction(ToMethod);
    ToMethod->addDeclInternal(ToParam);
  }

  SmallVector<SourceLocation, 12> FromSelLocs;
  D->getSelectorLocs(FromSelLocs);
  SmallVector<SourceLocation, 12> ToSelLocs(FromSelLocs.size());
  if (Error Err = ImportContainerChecked(FromSelLocs, ToSelLocs))
    return std::move(Err);

  ToMethod->setMethodParams(Importer.getToContext(), ToParams, ToSelLocs);

  ToMethod->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToMethod);

  // Implicit params are declared when Sema encounters the definition but this
  // never happens when the method is imported. Manually declare the implicit
  // params now that the MethodDecl knows its class interface.
  if (D->getSelfDecl())
    ToMethod->createImplicitParams(Importer.getToContext(),
                                   ToMethod->getClassInterface());

  return ToMethod;
}

ExpectedDecl ASTNodeImporter::VisitObjCTypeParamDecl(ObjCTypeParamDecl *D) {
  // Import the major distinguishing characteristics of a category.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  SourceLocation ToVarianceLoc, ToLocation, ToColonLoc;
  TypeSourceInfo *ToTypeSourceInfo;
  if (auto Imp = importSeq(
      D->getVarianceLoc(), D->getLocation(), D->getColonLoc(),
      D->getTypeSourceInfo()))
    std::tie(ToVarianceLoc, ToLocation, ToColonLoc, ToTypeSourceInfo) = *Imp;
  else
    return Imp.takeError();

  ObjCTypeParamDecl *Result;
  if (GetImportedOrCreateDecl(
          Result, D, Importer.getToContext(), DC, D->getVariance(),
          ToVarianceLoc, D->getIndex(),
          ToLocation, Name.getAsIdentifierInfo(),
          ToColonLoc, ToTypeSourceInfo))
    return Result;

  Result->setLexicalDeclContext(LexicalDC);
  return Result;
}

ExpectedDecl ASTNodeImporter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  // Import the major distinguishing characteristics of a category.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  ObjCInterfaceDecl *ToInterface;
  if (Error Err = importInto(ToInterface, D->getClassInterface()))
    return std::move(Err);

  // Determine if we've already encountered this category.
  ObjCCategoryDecl *MergeWithCategory
    = ToInterface->FindCategoryDeclaration(Name.getAsIdentifierInfo());
  ObjCCategoryDecl *ToCategory = MergeWithCategory;
  if (!ToCategory) {
    SourceLocation ToAtStartLoc, ToCategoryNameLoc;
    SourceLocation ToIvarLBraceLoc, ToIvarRBraceLoc;
    if (auto Imp = importSeq(
        D->getAtStartLoc(), D->getCategoryNameLoc(),
        D->getIvarLBraceLoc(), D->getIvarRBraceLoc()))
      std::tie(
          ToAtStartLoc, ToCategoryNameLoc,
          ToIvarLBraceLoc, ToIvarRBraceLoc) = *Imp;
    else
      return Imp.takeError();

    if (GetImportedOrCreateDecl(ToCategory, D, Importer.getToContext(), DC,
                                ToAtStartLoc, Loc,
                                ToCategoryNameLoc,
                                Name.getAsIdentifierInfo(), ToInterface,
                                /*TypeParamList=*/nullptr,
                                ToIvarLBraceLoc,
                                ToIvarRBraceLoc))
      return ToCategory;

    ToCategory->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToCategory);
    // Import the type parameter list after MapImported, to avoid
    // loops when bringing in their DeclContext.
    if (auto PListOrErr = ImportObjCTypeParamList(D->getTypeParamList()))
      ToCategory->setTypeParamList(*PListOrErr);
    else
      return PListOrErr.takeError();

    // Import protocols
    SmallVector<ObjCProtocolDecl *, 4> Protocols;
    SmallVector<SourceLocation, 4> ProtocolLocs;
    ObjCCategoryDecl::protocol_loc_iterator FromProtoLoc
      = D->protocol_loc_begin();
    for (ObjCCategoryDecl::protocol_iterator FromProto = D->protocol_begin(),
                                          FromProtoEnd = D->protocol_end();
         FromProto != FromProtoEnd;
         ++FromProto, ++FromProtoLoc) {
      if (Expected<ObjCProtocolDecl *> ToProtoOrErr = import(*FromProto))
        Protocols.push_back(*ToProtoOrErr);
      else
        return ToProtoOrErr.takeError();

      if (ExpectedSLoc ToProtoLocOrErr = import(*FromProtoLoc))
        ProtocolLocs.push_back(*ToProtoLocOrErr);
      else
        return ToProtoLocOrErr.takeError();
    }

    // FIXME: If we're merging, make sure that the protocol list is the same.
    ToCategory->setProtocolList(Protocols.data(), Protocols.size(),
                                ProtocolLocs.data(), Importer.getToContext());

  } else {
    Importer.MapImported(D, ToCategory);
  }

  // Import all of the members of this category.
  if (Error Err = ImportDeclContext(D))
    return std::move(Err);

  // If we have an implementation, import it as well.
  if (D->getImplementation()) {
    if (Expected<ObjCCategoryImplDecl *> ToImplOrErr =
        import(D->getImplementation()))
      ToCategory->setImplementation(*ToImplOrErr);
    else
      return ToImplOrErr.takeError();
  }

  return ToCategory;
}

Error ASTNodeImporter::ImportDefinition(
    ObjCProtocolDecl *From, ObjCProtocolDecl *To, ImportDefinitionKind Kind) {
  if (To->getDefinition()) {
    if (shouldForceImportDeclContext(Kind))
      if (Error Err = ImportDeclContext(From))
        return Err;
    return Error::success();
  }

  // Start the protocol definition
  To->startDefinition();

  // Import protocols
  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  SmallVector<SourceLocation, 4> ProtocolLocs;
  ObjCProtocolDecl::protocol_loc_iterator FromProtoLoc =
      From->protocol_loc_begin();
  for (ObjCProtocolDecl::protocol_iterator FromProto = From->protocol_begin(),
                                        FromProtoEnd = From->protocol_end();
       FromProto != FromProtoEnd;
       ++FromProto, ++FromProtoLoc) {
    if (Expected<ObjCProtocolDecl *> ToProtoOrErr = import(*FromProto))
      Protocols.push_back(*ToProtoOrErr);
    else
      return ToProtoOrErr.takeError();

    if (ExpectedSLoc ToProtoLocOrErr = import(*FromProtoLoc))
      ProtocolLocs.push_back(*ToProtoLocOrErr);
    else
      return ToProtoLocOrErr.takeError();

  }

  // FIXME: If we're merging, make sure that the protocol list is the same.
  To->setProtocolList(Protocols.data(), Protocols.size(),
                      ProtocolLocs.data(), Importer.getToContext());

  if (shouldForceImportDeclContext(Kind)) {
    // Import all of the members of this protocol.
    if (Error Err = ImportDeclContext(From, /*ForceImport=*/true))
      return Err;
  }
  return Error::success();
}

ExpectedDecl ASTNodeImporter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  // If this protocol has a definition in the translation unit we're coming
  // from, but this particular declaration is not that definition, import the
  // definition and map to that.
  ObjCProtocolDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    if (ExpectedDecl ImportedDefOrErr = import(Definition))
      return Importer.MapImported(D, *ImportedDefOrErr);
    else
      return ImportedDefOrErr.takeError();
  }

  // Import the major distinguishing characteristics of a protocol.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  ObjCProtocolDecl *MergeWithProtocol = nullptr;
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (!FoundDecl->isInIdentifierNamespace(Decl::IDNS_ObjCProtocol))
      continue;

    if ((MergeWithProtocol = dyn_cast<ObjCProtocolDecl>(FoundDecl)))
      break;
  }

  ObjCProtocolDecl *ToProto = MergeWithProtocol;
  if (!ToProto) {
    auto ToAtBeginLocOrErr = import(D->getAtStartLoc());
    if (!ToAtBeginLocOrErr)
      return ToAtBeginLocOrErr.takeError();

    if (GetImportedOrCreateDecl(ToProto, D, Importer.getToContext(), DC,
                                Name.getAsIdentifierInfo(), Loc,
                                *ToAtBeginLocOrErr,
                                /*PrevDecl=*/nullptr))
      return ToProto;
    ToProto->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToProto);
  }

  Importer.MapImported(D, ToProto);

  if (D->isThisDeclarationADefinition())
    if (Error Err = ImportDefinition(D, ToProto))
      return std::move(Err);

  return ToProto;
}

ExpectedDecl ASTNodeImporter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  ExpectedSLoc ExternLocOrErr = import(D->getExternLoc());
  if (!ExternLocOrErr)
    return ExternLocOrErr.takeError();

  ExpectedSLoc LangLocOrErr = import(D->getLocation());
  if (!LangLocOrErr)
    return LangLocOrErr.takeError();

  bool HasBraces = D->hasBraces();

  LinkageSpecDecl *ToLinkageSpec;
  if (GetImportedOrCreateDecl(ToLinkageSpec, D, Importer.getToContext(), DC,
                              *ExternLocOrErr, *LangLocOrErr,
                              D->getLanguage(), HasBraces))
    return ToLinkageSpec;

  if (HasBraces) {
    ExpectedSLoc RBraceLocOrErr = import(D->getRBraceLoc());
    if (!RBraceLocOrErr)
      return RBraceLocOrErr.takeError();
    ToLinkageSpec->setRBraceLoc(*RBraceLocOrErr);
  }

  ToLinkageSpec->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToLinkageSpec);

  return ToLinkageSpec;
}

ExpectedDecl ASTNodeImporter::VisitUsingDecl(UsingDecl *D) {
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD = nullptr;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  SourceLocation ToLoc, ToUsingLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  if (auto Imp = importSeq(
      D->getNameInfo().getLoc(), D->getUsingLoc(), D->getQualifierLoc()))
    std::tie(ToLoc, ToUsingLoc, ToQualifierLoc) = *Imp;
  else
    return Imp.takeError();

  DeclarationNameInfo NameInfo(Name, ToLoc);
  if (Error Err = ImportDeclarationNameLoc(D->getNameInfo(), NameInfo))
    return std::move(Err);

  UsingDecl *ToUsing;
  if (GetImportedOrCreateDecl(ToUsing, D, Importer.getToContext(), DC,
                              ToUsingLoc, ToQualifierLoc, NameInfo,
                              D->hasTypename()))
    return ToUsing;

  ToUsing->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToUsing);

  if (NamedDecl *FromPattern =
      Importer.getFromContext().getInstantiatedFromUsingDecl(D)) {
    if (Expected<NamedDecl *> ToPatternOrErr = import(FromPattern))
      Importer.getToContext().setInstantiatedFromUsingDecl(
          ToUsing, *ToPatternOrErr);
    else
      return ToPatternOrErr.takeError();
  }

  for (UsingShadowDecl *FromShadow : D->shadows()) {
    if (Expected<UsingShadowDecl *> ToShadowOrErr = import(FromShadow))
      ToUsing->addShadowDecl(*ToShadowOrErr);
    else
      // FIXME: We return error here but the definition is already created
      // and available with lookups. How to fix this?..
      return ToShadowOrErr.takeError();
  }
  return ToUsing;
}

ExpectedDecl ASTNodeImporter::VisitUsingShadowDecl(UsingShadowDecl *D) {
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD = nullptr;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  Expected<UsingDecl *> ToUsingOrErr = import(D->getUsingDecl());
  if (!ToUsingOrErr)
    return ToUsingOrErr.takeError();

  Expected<NamedDecl *> ToTargetOrErr = import(D->getTargetDecl());
  if (!ToTargetOrErr)
    return ToTargetOrErr.takeError();

  UsingShadowDecl *ToShadow;
  if (GetImportedOrCreateDecl(ToShadow, D, Importer.getToContext(), DC, Loc,
                              *ToUsingOrErr, *ToTargetOrErr))
    return ToShadow;

  ToShadow->setLexicalDeclContext(LexicalDC);
  ToShadow->setAccess(D->getAccess());

  if (UsingShadowDecl *FromPattern =
      Importer.getFromContext().getInstantiatedFromUsingShadowDecl(D)) {
    if (Expected<UsingShadowDecl *> ToPatternOrErr = import(FromPattern))
      Importer.getToContext().setInstantiatedFromUsingShadowDecl(
          ToShadow, *ToPatternOrErr);
    else
      // FIXME: We return error here but the definition is already created
      // and available with lookups. How to fix this?..
      return ToPatternOrErr.takeError();
  }

  LexicalDC->addDeclInternal(ToShadow);

  return ToShadow;
}

ExpectedDecl ASTNodeImporter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD = nullptr;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  auto ToComAncestorOrErr = Importer.ImportContext(D->getCommonAncestor());
  if (!ToComAncestorOrErr)
    return ToComAncestorOrErr.takeError();

  NamespaceDecl *ToNominatedNamespace;
  SourceLocation ToUsingLoc, ToNamespaceKeyLocation, ToIdentLocation;
  NestedNameSpecifierLoc ToQualifierLoc;
  if (auto Imp = importSeq(
      D->getNominatedNamespace(), D->getUsingLoc(),
      D->getNamespaceKeyLocation(), D->getQualifierLoc(),
      D->getIdentLocation()))
    std::tie(
        ToNominatedNamespace, ToUsingLoc, ToNamespaceKeyLocation,
        ToQualifierLoc, ToIdentLocation) = *Imp;
  else
    return Imp.takeError();

  UsingDirectiveDecl *ToUsingDir;
  if (GetImportedOrCreateDecl(ToUsingDir, D, Importer.getToContext(), DC,
                              ToUsingLoc,
                              ToNamespaceKeyLocation,
                              ToQualifierLoc,
                              ToIdentLocation,
                              ToNominatedNamespace, *ToComAncestorOrErr))
    return ToUsingDir;

  ToUsingDir->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToUsingDir);

  return ToUsingDir;
}

ExpectedDecl ASTNodeImporter::VisitUnresolvedUsingValueDecl(
    UnresolvedUsingValueDecl *D) {
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD = nullptr;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  SourceLocation ToLoc, ToUsingLoc, ToEllipsisLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  if (auto Imp = importSeq(
      D->getNameInfo().getLoc(), D->getUsingLoc(), D->getQualifierLoc(),
      D->getEllipsisLoc()))
    std::tie(ToLoc, ToUsingLoc, ToQualifierLoc, ToEllipsisLoc) = *Imp;
  else
    return Imp.takeError();

  DeclarationNameInfo NameInfo(Name, ToLoc);
  if (Error Err = ImportDeclarationNameLoc(D->getNameInfo(), NameInfo))
    return std::move(Err);

  UnresolvedUsingValueDecl *ToUsingValue;
  if (GetImportedOrCreateDecl(ToUsingValue, D, Importer.getToContext(), DC,
                              ToUsingLoc, ToQualifierLoc, NameInfo,
                              ToEllipsisLoc))
    return ToUsingValue;

  ToUsingValue->setAccess(D->getAccess());
  ToUsingValue->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToUsingValue);

  return ToUsingValue;
}

ExpectedDecl ASTNodeImporter::VisitUnresolvedUsingTypenameDecl(
    UnresolvedUsingTypenameDecl *D) {
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD = nullptr;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  SourceLocation ToUsingLoc, ToTypenameLoc, ToEllipsisLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  if (auto Imp = importSeq(
      D->getUsingLoc(), D->getTypenameLoc(), D->getQualifierLoc(),
      D->getEllipsisLoc()))
    std::tie(ToUsingLoc, ToTypenameLoc, ToQualifierLoc, ToEllipsisLoc) = *Imp;
  else
    return Imp.takeError();

  UnresolvedUsingTypenameDecl *ToUsing;
  if (GetImportedOrCreateDecl(ToUsing, D, Importer.getToContext(), DC,
                              ToUsingLoc, ToTypenameLoc,
                              ToQualifierLoc, Loc, Name, ToEllipsisLoc))
    return ToUsing;

  ToUsing->setAccess(D->getAccess());
  ToUsing->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToUsing);

  return ToUsing;
}

ExpectedDecl ASTNodeImporter::VisitBuiltinTemplateDecl(BuiltinTemplateDecl *D) {
  Decl* ToD = nullptr;
  switch (D->getBuiltinTemplateKind()) {
  case BuiltinTemplateKind::BTK__make_integer_seq:
    ToD = Importer.getToContext().getMakeIntegerSeqDecl();
    break;
  case BuiltinTemplateKind::BTK__type_pack_element:
    ToD = Importer.getToContext().getTypePackElementDecl();
    break;
  }
  assert(ToD && "BuiltinTemplateDecl of unsupported kind!");
  Importer.MapImported(D, ToD);
  return ToD;
}

Error ASTNodeImporter::ImportDefinition(
    ObjCInterfaceDecl *From, ObjCInterfaceDecl *To, ImportDefinitionKind Kind) {
  if (To->getDefinition()) {
    // Check consistency of superclass.
    ObjCInterfaceDecl *FromSuper = From->getSuperClass();
    if (FromSuper) {
      if (auto FromSuperOrErr = import(FromSuper))
        FromSuper = *FromSuperOrErr;
      else
        return FromSuperOrErr.takeError();
    }

    ObjCInterfaceDecl *ToSuper = To->getSuperClass();
    if ((bool)FromSuper != (bool)ToSuper ||
        (FromSuper && !declaresSameEntity(FromSuper, ToSuper))) {
      Importer.ToDiag(To->getLocation(),
                      diag::warn_odr_objc_superclass_inconsistent)
        << To->getDeclName();
      if (ToSuper)
        Importer.ToDiag(To->getSuperClassLoc(), diag::note_odr_objc_superclass)
          << To->getSuperClass()->getDeclName();
      else
        Importer.ToDiag(To->getLocation(),
                        diag::note_odr_objc_missing_superclass);
      if (From->getSuperClass())
        Importer.FromDiag(From->getSuperClassLoc(),
                          diag::note_odr_objc_superclass)
        << From->getSuperClass()->getDeclName();
      else
        Importer.FromDiag(From->getLocation(),
                          diag::note_odr_objc_missing_superclass);
    }

    if (shouldForceImportDeclContext(Kind))
      if (Error Err = ImportDeclContext(From))
        return Err;
    return Error::success();
  }

  // Start the definition.
  To->startDefinition();

  // If this class has a superclass, import it.
  if (From->getSuperClass()) {
    if (auto SuperTInfoOrErr = import(From->getSuperClassTInfo()))
      To->setSuperClass(*SuperTInfoOrErr);
    else
      return SuperTInfoOrErr.takeError();
  }

  // Import protocols
  SmallVector<ObjCProtocolDecl *, 4> Protocols;
  SmallVector<SourceLocation, 4> ProtocolLocs;
  ObjCInterfaceDecl::protocol_loc_iterator FromProtoLoc =
      From->protocol_loc_begin();

  for (ObjCInterfaceDecl::protocol_iterator FromProto = From->protocol_begin(),
                                         FromProtoEnd = From->protocol_end();
       FromProto != FromProtoEnd;
       ++FromProto, ++FromProtoLoc) {
    if (Expected<ObjCProtocolDecl *> ToProtoOrErr = import(*FromProto))
      Protocols.push_back(*ToProtoOrErr);
    else
      return ToProtoOrErr.takeError();

    if (ExpectedSLoc ToProtoLocOrErr = import(*FromProtoLoc))
      ProtocolLocs.push_back(*ToProtoLocOrErr);
    else
      return ToProtoLocOrErr.takeError();

  }

  // FIXME: If we're merging, make sure that the protocol list is the same.
  To->setProtocolList(Protocols.data(), Protocols.size(),
                      ProtocolLocs.data(), Importer.getToContext());

  // Import categories. When the categories themselves are imported, they'll
  // hook themselves into this interface.
  for (auto *Cat : From->known_categories()) {
    auto ToCatOrErr = import(Cat);
    if (!ToCatOrErr)
      return ToCatOrErr.takeError();
  }

  // If we have an @implementation, import it as well.
  if (From->getImplementation()) {
    if (Expected<ObjCImplementationDecl *> ToImplOrErr =
        import(From->getImplementation()))
      To->setImplementation(*ToImplOrErr);
    else
      return ToImplOrErr.takeError();
  }

  if (shouldForceImportDeclContext(Kind)) {
    // Import all of the members of this class.
    if (Error Err = ImportDeclContext(From, /*ForceImport=*/true))
      return Err;
  }
  return Error::success();
}

Expected<ObjCTypeParamList *>
ASTNodeImporter::ImportObjCTypeParamList(ObjCTypeParamList *list) {
  if (!list)
    return nullptr;

  SmallVector<ObjCTypeParamDecl *, 4> toTypeParams;
  for (auto *fromTypeParam : *list) {
    if (auto toTypeParamOrErr = import(fromTypeParam))
      toTypeParams.push_back(*toTypeParamOrErr);
    else
      return toTypeParamOrErr.takeError();
  }

  auto LAngleLocOrErr = import(list->getLAngleLoc());
  if (!LAngleLocOrErr)
    return LAngleLocOrErr.takeError();

  auto RAngleLocOrErr = import(list->getRAngleLoc());
  if (!RAngleLocOrErr)
    return RAngleLocOrErr.takeError();

  return ObjCTypeParamList::create(Importer.getToContext(),
                                   *LAngleLocOrErr,
                                   toTypeParams,
                                   *RAngleLocOrErr);
}

ExpectedDecl ASTNodeImporter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // If this class has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  ObjCInterfaceDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    if (ExpectedDecl ImportedDefOrErr = import(Definition))
      return Importer.MapImported(D, *ImportedDefOrErr);
    else
      return ImportedDefOrErr.takeError();
  }

  // Import the major distinguishing characteristics of an @interface.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Look for an existing interface with the same name.
  ObjCInterfaceDecl *MergeWithIface = nullptr;
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (!FoundDecl->isInIdentifierNamespace(Decl::IDNS_Ordinary))
      continue;

    if ((MergeWithIface = dyn_cast<ObjCInterfaceDecl>(FoundDecl)))
      break;
  }

  // Create an interface declaration, if one does not already exist.
  ObjCInterfaceDecl *ToIface = MergeWithIface;
  if (!ToIface) {
    ExpectedSLoc AtBeginLocOrErr = import(D->getAtStartLoc());
    if (!AtBeginLocOrErr)
      return AtBeginLocOrErr.takeError();

    if (GetImportedOrCreateDecl(
            ToIface, D, Importer.getToContext(), DC,
            *AtBeginLocOrErr, Name.getAsIdentifierInfo(),
            /*TypeParamList=*/nullptr,
            /*PrevDecl=*/nullptr, Loc, D->isImplicitInterfaceDecl()))
      return ToIface;
    ToIface->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToIface);
  }
  Importer.MapImported(D, ToIface);
  // Import the type parameter list after MapImported, to avoid
  // loops when bringing in their DeclContext.
  if (auto ToPListOrErr =
      ImportObjCTypeParamList(D->getTypeParamListAsWritten()))
    ToIface->setTypeParamList(*ToPListOrErr);
  else
    return ToPListOrErr.takeError();

  if (D->isThisDeclarationADefinition())
    if (Error Err = ImportDefinition(D, ToIface))
      return std::move(Err);

  return ToIface;
}

ExpectedDecl
ASTNodeImporter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  ObjCCategoryDecl *Category;
  if (Error Err = importInto(Category, D->getCategoryDecl()))
    return std::move(Err);

  ObjCCategoryImplDecl *ToImpl = Category->getImplementation();
  if (!ToImpl) {
    DeclContext *DC, *LexicalDC;
    if (Error Err = ImportDeclContext(D, DC, LexicalDC))
      return std::move(Err);

    SourceLocation ToLocation, ToAtStartLoc, ToCategoryNameLoc;
    if (auto Imp = importSeq(
        D->getLocation(), D->getAtStartLoc(), D->getCategoryNameLoc()))
      std::tie(ToLocation, ToAtStartLoc, ToCategoryNameLoc) = *Imp;
    else
      return Imp.takeError();

    if (GetImportedOrCreateDecl(
            ToImpl, D, Importer.getToContext(), DC,
            Importer.Import(D->getIdentifier()), Category->getClassInterface(),
            ToLocation, ToAtStartLoc, ToCategoryNameLoc))
      return ToImpl;

    ToImpl->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToImpl);
    Category->setImplementation(ToImpl);
  }

  Importer.MapImported(D, ToImpl);
  if (Error Err = ImportDeclContext(D))
    return std::move(Err);

  return ToImpl;
}

ExpectedDecl
ASTNodeImporter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  // Find the corresponding interface.
  ObjCInterfaceDecl *Iface;
  if (Error Err = importInto(Iface, D->getClassInterface()))
    return std::move(Err);

  // Import the superclass, if any.
  ObjCInterfaceDecl *Super;
  if (Error Err = importInto(Super, D->getSuperClass()))
    return std::move(Err);

  ObjCImplementationDecl *Impl = Iface->getImplementation();
  if (!Impl) {
    // We haven't imported an implementation yet. Create a new @implementation
    // now.
    DeclContext *DC, *LexicalDC;
    if (Error Err = ImportDeclContext(D, DC, LexicalDC))
      return std::move(Err);

    SourceLocation ToLocation, ToAtStartLoc, ToSuperClassLoc;
    SourceLocation ToIvarLBraceLoc, ToIvarRBraceLoc;
    if (auto Imp = importSeq(
        D->getLocation(), D->getAtStartLoc(), D->getSuperClassLoc(),
        D->getIvarLBraceLoc(), D->getIvarRBraceLoc()))
      std::tie(
          ToLocation, ToAtStartLoc, ToSuperClassLoc,
          ToIvarLBraceLoc, ToIvarRBraceLoc) = *Imp;
    else
      return Imp.takeError();

    if (GetImportedOrCreateDecl(Impl, D, Importer.getToContext(),
                                DC, Iface, Super,
                                ToLocation,
                                ToAtStartLoc,
                                ToSuperClassLoc,
                                ToIvarLBraceLoc,
                                ToIvarRBraceLoc))
      return Impl;

    Impl->setLexicalDeclContext(LexicalDC);

    // Associate the implementation with the class it implements.
    Iface->setImplementation(Impl);
    Importer.MapImported(D, Iface->getImplementation());
  } else {
    Importer.MapImported(D, Iface->getImplementation());

    // Verify that the existing @implementation has the same superclass.
    if ((Super && !Impl->getSuperClass()) ||
        (!Super && Impl->getSuperClass()) ||
        (Super && Impl->getSuperClass() &&
         !declaresSameEntity(Super->getCanonicalDecl(),
                             Impl->getSuperClass()))) {
      Importer.ToDiag(Impl->getLocation(),
                      diag::warn_odr_objc_superclass_inconsistent)
        << Iface->getDeclName();
      // FIXME: It would be nice to have the location of the superclass
      // below.
      if (Impl->getSuperClass())
        Importer.ToDiag(Impl->getLocation(),
                        diag::note_odr_objc_superclass)
        << Impl->getSuperClass()->getDeclName();
      else
        Importer.ToDiag(Impl->getLocation(),
                        diag::note_odr_objc_missing_superclass);
      if (D->getSuperClass())
        Importer.FromDiag(D->getLocation(),
                          diag::note_odr_objc_superclass)
        << D->getSuperClass()->getDeclName();
      else
        Importer.FromDiag(D->getLocation(),
                          diag::note_odr_objc_missing_superclass);

      return make_error<ImportError>(ImportError::NameConflict);
    }
  }

  // Import all of the members of this @implementation.
  if (Error Err = ImportDeclContext(D))
    return std::move(Err);

  return Impl;
}

ExpectedDecl ASTNodeImporter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  // Import the major distinguishing characteristics of an @property.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // Check whether we have already imported this property.
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (auto *FoundProp = dyn_cast<ObjCPropertyDecl>(FoundDecl)) {
      // Check property types.
      if (!Importer.IsStructurallyEquivalent(D->getType(),
                                             FoundProp->getType())) {
        Importer.ToDiag(Loc, diag::warn_odr_objc_property_type_inconsistent)
          << Name << D->getType() << FoundProp->getType();
        Importer.ToDiag(FoundProp->getLocation(), diag::note_odr_value_here)
          << FoundProp->getType();

        return make_error<ImportError>(ImportError::NameConflict);
      }

      // FIXME: Check property attributes, getters, setters, etc.?

      // Consider these properties to be equivalent.
      Importer.MapImported(D, FoundProp);
      return FoundProp;
    }
  }

  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  SourceLocation ToAtLoc, ToLParenLoc;
  if (auto Imp = importSeq(
      D->getType(), D->getTypeSourceInfo(), D->getAtLoc(), D->getLParenLoc()))
    std::tie(ToType, ToTypeSourceInfo, ToAtLoc, ToLParenLoc) = *Imp;
  else
    return Imp.takeError();

  // Create the new property.
  ObjCPropertyDecl *ToProperty;
  if (GetImportedOrCreateDecl(
          ToProperty, D, Importer.getToContext(), DC, Loc,
          Name.getAsIdentifierInfo(), ToAtLoc,
          ToLParenLoc, ToType,
          ToTypeSourceInfo, D->getPropertyImplementation()))
    return ToProperty;

  Selector ToGetterName, ToSetterName;
  SourceLocation ToGetterNameLoc, ToSetterNameLoc;
  ObjCMethodDecl *ToGetterMethodDecl, *ToSetterMethodDecl;
  ObjCIvarDecl *ToPropertyIvarDecl;
  if (auto Imp = importSeq(
      D->getGetterName(), D->getSetterName(),
      D->getGetterNameLoc(), D->getSetterNameLoc(),
      D->getGetterMethodDecl(), D->getSetterMethodDecl(),
      D->getPropertyIvarDecl()))
    std::tie(
        ToGetterName, ToSetterName,
        ToGetterNameLoc, ToSetterNameLoc,
        ToGetterMethodDecl, ToSetterMethodDecl,
        ToPropertyIvarDecl) = *Imp;
  else
    return Imp.takeError();

  ToProperty->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToProperty);

  ToProperty->setPropertyAttributes(D->getPropertyAttributes());
  ToProperty->setPropertyAttributesAsWritten(
                                      D->getPropertyAttributesAsWritten());
  ToProperty->setGetterName(ToGetterName, ToGetterNameLoc);
  ToProperty->setSetterName(ToSetterName, ToSetterNameLoc);
  ToProperty->setGetterMethodDecl(ToGetterMethodDecl);
  ToProperty->setSetterMethodDecl(ToSetterMethodDecl);
  ToProperty->setPropertyIvarDecl(ToPropertyIvarDecl);
  return ToProperty;
}

ExpectedDecl
ASTNodeImporter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  ObjCPropertyDecl *Property;
  if (Error Err = importInto(Property, D->getPropertyDecl()))
    return std::move(Err);

  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  auto *InImpl = cast<ObjCImplDecl>(LexicalDC);

  // Import the ivar (for an @synthesize).
  ObjCIvarDecl *Ivar = nullptr;
  if (Error Err = importInto(Ivar, D->getPropertyIvarDecl()))
    return std::move(Err);

  ObjCPropertyImplDecl *ToImpl
    = InImpl->FindPropertyImplDecl(Property->getIdentifier(),
                                   Property->getQueryKind());
  if (!ToImpl) {
    SourceLocation ToBeginLoc, ToLocation, ToPropertyIvarDeclLoc;
    if (auto Imp = importSeq(
        D->getBeginLoc(), D->getLocation(), D->getPropertyIvarDeclLoc()))
      std::tie(ToBeginLoc, ToLocation, ToPropertyIvarDeclLoc) = *Imp;
    else
      return Imp.takeError();

    if (GetImportedOrCreateDecl(ToImpl, D, Importer.getToContext(), DC,
                                ToBeginLoc,
                                ToLocation, Property,
                                D->getPropertyImplementation(), Ivar,
                                ToPropertyIvarDeclLoc))
      return ToImpl;

    ToImpl->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(ToImpl);
  } else {
    // Check that we have the same kind of property implementation (@synthesize
    // vs. @dynamic).
    if (D->getPropertyImplementation() != ToImpl->getPropertyImplementation()) {
      Importer.ToDiag(ToImpl->getLocation(),
                      diag::warn_odr_objc_property_impl_kind_inconsistent)
        << Property->getDeclName()
        << (ToImpl->getPropertyImplementation()
                                              == ObjCPropertyImplDecl::Dynamic);
      Importer.FromDiag(D->getLocation(),
                        diag::note_odr_objc_property_impl_kind)
        << D->getPropertyDecl()->getDeclName()
        << (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic);

      return make_error<ImportError>(ImportError::NameConflict);
    }

    // For @synthesize, check that we have the same
    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize &&
        Ivar != ToImpl->getPropertyIvarDecl()) {
      Importer.ToDiag(ToImpl->getPropertyIvarDeclLoc(),
                      diag::warn_odr_objc_synthesize_ivar_inconsistent)
        << Property->getDeclName()
        << ToImpl->getPropertyIvarDecl()->getDeclName()
        << Ivar->getDeclName();
      Importer.FromDiag(D->getPropertyIvarDeclLoc(),
                        diag::note_odr_objc_synthesize_ivar_here)
        << D->getPropertyIvarDecl()->getDeclName();

      return make_error<ImportError>(ImportError::NameConflict);
    }

    // Merge the existing implementation with the new implementation.
    Importer.MapImported(D, ToImpl);
  }

  return ToImpl;
}

ExpectedDecl
ASTNodeImporter::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  // For template arguments, we adopt the translation unit as our declaration
  // context. This context will be fixed when the actual template declaration
  // is created.

  // FIXME: Import default argument.

  ExpectedSLoc BeginLocOrErr = import(D->getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();

  ExpectedSLoc LocationOrErr = import(D->getLocation());
  if (!LocationOrErr)
    return LocationOrErr.takeError();

  TemplateTypeParmDecl *ToD = nullptr;
  (void)GetImportedOrCreateDecl(
      ToD, D, Importer.getToContext(),
      Importer.getToContext().getTranslationUnitDecl(),
      *BeginLocOrErr, *LocationOrErr,
      D->getDepth(), D->getIndex(), Importer.Import(D->getIdentifier()),
      D->wasDeclaredWithTypename(), D->isParameterPack());
  return ToD;
}

ExpectedDecl
ASTNodeImporter::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  DeclarationName ToDeclName;
  SourceLocation ToLocation, ToInnerLocStart;
  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  if (auto Imp = importSeq(
      D->getDeclName(), D->getLocation(), D->getType(), D->getTypeSourceInfo(),
      D->getInnerLocStart()))
    std::tie(
        ToDeclName, ToLocation, ToType, ToTypeSourceInfo,
        ToInnerLocStart) = *Imp;
  else
    return Imp.takeError();

  // FIXME: Import default argument.

  NonTypeTemplateParmDecl *ToD = nullptr;
  (void)GetImportedOrCreateDecl(
      ToD, D, Importer.getToContext(),
      Importer.getToContext().getTranslationUnitDecl(),
      ToInnerLocStart, ToLocation, D->getDepth(),
      D->getPosition(), ToDeclName.getAsIdentifierInfo(), ToType,
      D->isParameterPack(), ToTypeSourceInfo);
  return ToD;
}

ExpectedDecl
ASTNodeImporter::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  // Import the name of this declaration.
  auto NameOrErr = import(D->getDeclName());
  if (!NameOrErr)
    return NameOrErr.takeError();

  // Import the location of this declaration.
  ExpectedSLoc LocationOrErr = import(D->getLocation());
  if (!LocationOrErr)
    return LocationOrErr.takeError();

  // Import template parameters.
  auto TemplateParamsOrErr = import(D->getTemplateParameters());
  if (!TemplateParamsOrErr)
    return TemplateParamsOrErr.takeError();

  // FIXME: Import default argument.

  TemplateTemplateParmDecl *ToD = nullptr;
  (void)GetImportedOrCreateDecl(
      ToD, D, Importer.getToContext(),
      Importer.getToContext().getTranslationUnitDecl(), *LocationOrErr,
      D->getDepth(), D->getPosition(), D->isParameterPack(),
      (*NameOrErr).getAsIdentifierInfo(),
      *TemplateParamsOrErr);
  return ToD;
}

// Returns the definition for a (forward) declaration of a TemplateDecl, if
// it has any definition in the redecl chain.
template <typename T> static auto getTemplateDefinition(T *D) -> T * {
  assert(D->getTemplatedDecl() && "Should be called on templates only");
  auto *ToTemplatedDef = D->getTemplatedDecl()->getDefinition();
  if (!ToTemplatedDef)
    return nullptr;
  auto *TemplateWithDef = ToTemplatedDef->getDescribedTemplate();
  return cast_or_null<T>(TemplateWithDef);
}

ExpectedDecl ASTNodeImporter::VisitClassTemplateDecl(ClassTemplateDecl *D) {

  // Import the major distinguishing characteristics of this class template.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  ClassTemplateDecl *FoundByLookup = nullptr;

  // We may already have a template of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod()) {
    SmallVector<NamedDecl *, 4> ConflictingDecls;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(Decl::IDNS_Ordinary |
                                              Decl::IDNS_TagFriend))
        continue;

      Decl *Found = FoundDecl;
      auto *FoundTemplate = dyn_cast<ClassTemplateDecl>(Found);
      if (FoundTemplate) {
        if (!hasSameVisibilityContext(FoundTemplate, D))
          continue;

        if (IsStructuralMatch(D, FoundTemplate)) {
          ClassTemplateDecl *TemplateWithDef =
              getTemplateDefinition(FoundTemplate);
          if (D->isThisDeclarationADefinition() && TemplateWithDef)
            return Importer.MapImported(D, TemplateWithDef);
          if (!FoundByLookup)
            FoundByLookup = FoundTemplate;
          // Search in all matches because there may be multiple decl chains,
          // see ASTTests test ImportExistingFriendClassTemplateDef.
          continue;
        }
        ConflictingDecls.push_back(FoundDecl);
      }
    }

    if (!ConflictingDecls.empty()) {
      ExpectedName NameOrErr = Importer.HandleNameConflict(
          Name, DC, Decl::IDNS_Ordinary, ConflictingDecls.data(),
          ConflictingDecls.size());
      if (NameOrErr)
        Name = NameOrErr.get();
      else
        return NameOrErr.takeError();
    }
  }

  CXXRecordDecl *FromTemplated = D->getTemplatedDecl();

  // Create the declaration that is being templated.
  CXXRecordDecl *ToTemplated;
  if (Error Err = importInto(ToTemplated, FromTemplated))
    return std::move(Err);

  // Create the class template declaration itself.
  auto TemplateParamsOrErr = import(D->getTemplateParameters());
  if (!TemplateParamsOrErr)
    return TemplateParamsOrErr.takeError();

  ClassTemplateDecl *D2;
  if (GetImportedOrCreateDecl(D2, D, Importer.getToContext(), DC, Loc, Name,
                              *TemplateParamsOrErr, ToTemplated))
    return D2;

  ToTemplated->setDescribedClassTemplate(D2);

  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);

  addDeclToContexts(D, D2);

  if (FoundByLookup) {
    auto *Recent =
        const_cast<ClassTemplateDecl *>(FoundByLookup->getMostRecentDecl());

    // It is possible that during the import of the class template definition
    // we start the import of a fwd friend decl of the very same class template
    // and we add the fwd friend decl to the lookup table. But the ToTemplated
    // had been created earlier and by that time the lookup could not find
    // anything existing, so it has no previous decl. Later, (still during the
    // import of the fwd friend decl) we start to import the definition again
    // and this time the lookup finds the previous fwd friend class template.
    // In this case we must set up the previous decl for the templated decl.
    if (!ToTemplated->getPreviousDecl()) {
      assert(FoundByLookup->getTemplatedDecl() &&
             "Found decl must have its templated decl set");
      CXXRecordDecl *PrevTemplated =
          FoundByLookup->getTemplatedDecl()->getMostRecentDecl();
      if (ToTemplated != PrevTemplated)
        ToTemplated->setPreviousDecl(PrevTemplated);
    }

    D2->setPreviousDecl(Recent);
  }

  if (FromTemplated->isCompleteDefinition() &&
      !ToTemplated->isCompleteDefinition()) {
    // FIXME: Import definition!
  }

  return D2;
}

ExpectedDecl ASTNodeImporter::VisitClassTemplateSpecializationDecl(
                                          ClassTemplateSpecializationDecl *D) {
  ClassTemplateDecl *ClassTemplate;
  if (Error Err = importInto(ClassTemplate, D->getSpecializedTemplate()))
    return std::move(Err);

  // Import the context of this declaration.
  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  // Import template arguments.
  SmallVector<TemplateArgument, 2> TemplateArgs;
  if (Error Err = ImportTemplateArguments(
      D->getTemplateArgs().data(), D->getTemplateArgs().size(), TemplateArgs))
    return std::move(Err);
  // Try to find an existing specialization with these template arguments and
  // template parameter list.
  void *InsertPos = nullptr;
  ClassTemplateSpecializationDecl *PrevDecl = nullptr;
  ClassTemplatePartialSpecializationDecl *PartialSpec =
            dyn_cast<ClassTemplatePartialSpecializationDecl>(D);

  // Import template parameters.
  TemplateParameterList *ToTPList = nullptr;

  if (PartialSpec) {
    auto ToTPListOrErr = import(PartialSpec->getTemplateParameters());
    if (!ToTPListOrErr)
      return ToTPListOrErr.takeError();
    ToTPList = *ToTPListOrErr;
    PrevDecl = ClassTemplate->findPartialSpecialization(TemplateArgs,
                                                        *ToTPListOrErr,
                                                        InsertPos);
  } else
    PrevDecl = ClassTemplate->findSpecialization(TemplateArgs, InsertPos);

  if (PrevDecl) {
    if (IsStructuralMatch(D, PrevDecl)) {
      if (D->isThisDeclarationADefinition() && PrevDecl->getDefinition()) {
        Importer.MapImported(D, PrevDecl->getDefinition());
        // Import those default field initializers which have been
        // instantiated in the "From" context, but not in the "To" context.
        for (auto *FromField : D->fields()) {
          auto ToOrErr = import(FromField);
          if (!ToOrErr)
            return ToOrErr.takeError();
        }

        // Import those methods which have been instantiated in the
        // "From" context, but not in the "To" context.
        for (CXXMethodDecl *FromM : D->methods()) {
          auto ToOrErr = import(FromM);
          if (!ToOrErr)
            return ToOrErr.takeError();
        }

        // TODO Import instantiated default arguments.
        // TODO Import instantiated exception specifications.
        //
        // Generally, ASTCommon.h/DeclUpdateKind enum gives a very good hint
        // what else could be fused during an AST merge.
        return PrevDecl;
      }
    } else { // ODR violation.
      // FIXME HandleNameConflict
      return make_error<ImportError>(ImportError::NameConflict);
    }
  }

  // Import the location of this declaration.
  ExpectedSLoc BeginLocOrErr = import(D->getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();
  ExpectedSLoc IdLocOrErr = import(D->getLocation());
  if (!IdLocOrErr)
    return IdLocOrErr.takeError();

  // Create the specialization.
  ClassTemplateSpecializationDecl *D2 = nullptr;
  if (PartialSpec) {
    // Import TemplateArgumentListInfo.
    TemplateArgumentListInfo ToTAInfo;
    const auto &ASTTemplateArgs = *PartialSpec->getTemplateArgsAsWritten();
    if (Error Err = ImportTemplateArgumentListInfo(ASTTemplateArgs, ToTAInfo))
      return std::move(Err);

    QualType CanonInjType;
    if (Error Err = importInto(
        CanonInjType, PartialSpec->getInjectedSpecializationType()))
      return std::move(Err);
    CanonInjType = CanonInjType.getCanonicalType();

    if (GetImportedOrCreateDecl<ClassTemplatePartialSpecializationDecl>(
            D2, D, Importer.getToContext(), D->getTagKind(), DC,
            *BeginLocOrErr, *IdLocOrErr, ToTPList, ClassTemplate,
            llvm::makeArrayRef(TemplateArgs.data(), TemplateArgs.size()),
            ToTAInfo, CanonInjType,
            cast_or_null<ClassTemplatePartialSpecializationDecl>(PrevDecl)))
      return D2;

    // Update InsertPos, because preceding import calls may have invalidated
    // it by adding new specializations.
    auto *PartSpec2 = cast<ClassTemplatePartialSpecializationDecl>(D2);
    if (!ClassTemplate->findPartialSpecialization(TemplateArgs, ToTPList,
                                                  InsertPos))
      // Add this partial specialization to the class template.
      ClassTemplate->AddPartialSpecialization(PartSpec2, InsertPos);

  } else { // Not a partial specialization.
    if (GetImportedOrCreateDecl(
            D2, D, Importer.getToContext(), D->getTagKind(), DC,
            *BeginLocOrErr, *IdLocOrErr, ClassTemplate, TemplateArgs,
            PrevDecl))
      return D2;

    // Update InsertPos, because preceding import calls may have invalidated
    // it by adding new specializations.
    if (!ClassTemplate->findSpecialization(TemplateArgs, InsertPos))
      // Add this specialization to the class template.
      ClassTemplate->AddSpecialization(D2, InsertPos);
  }

  D2->setSpecializationKind(D->getSpecializationKind());

  // Set the context of this specialization/instantiation.
  D2->setLexicalDeclContext(LexicalDC);

  // Add to the DC only if it was an explicit specialization/instantiation.
  if (D2->isExplicitInstantiationOrSpecialization()) {
    LexicalDC->addDeclInternal(D2);
  }

  if (auto BraceRangeOrErr = import(D->getBraceRange()))
    D2->setBraceRange(*BraceRangeOrErr);
  else
    return BraceRangeOrErr.takeError();

  // Import the qualifier, if any.
  if (auto LocOrErr = import(D->getQualifierLoc()))
    D2->setQualifierInfo(*LocOrErr);
  else
    return LocOrErr.takeError();

  if (auto *TSI = D->getTypeAsWritten()) {
    if (auto TInfoOrErr = import(TSI))
      D2->setTypeAsWritten(*TInfoOrErr);
    else
      return TInfoOrErr.takeError();

    if (auto LocOrErr = import(D->getTemplateKeywordLoc()))
      D2->setTemplateKeywordLoc(*LocOrErr);
    else
      return LocOrErr.takeError();

    if (auto LocOrErr = import(D->getExternLoc()))
      D2->setExternLoc(*LocOrErr);
    else
      return LocOrErr.takeError();
  }

  if (D->getPointOfInstantiation().isValid()) {
    if (auto POIOrErr = import(D->getPointOfInstantiation()))
      D2->setPointOfInstantiation(*POIOrErr);
    else
      return POIOrErr.takeError();
  }

  D2->setTemplateSpecializationKind(D->getTemplateSpecializationKind());

  if (D->isCompleteDefinition())
    if (Error Err = ImportDefinition(D, D2))
      return std::move(Err);

  return D2;
}

ExpectedDecl ASTNodeImporter::VisitVarTemplateDecl(VarTemplateDecl *D) {
  // If this variable has a definition in the translation unit we're coming
  // from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  auto *Definition =
      cast_or_null<VarDecl>(D->getTemplatedDecl()->getDefinition());
  if (Definition && Definition != D->getTemplatedDecl()) {
    if (ExpectedDecl ImportedDefOrErr = import(
        Definition->getDescribedVarTemplate()))
      return Importer.MapImported(D, *ImportedDefOrErr);
    else
      return ImportedDefOrErr.takeError();
  }

  // Import the major distinguishing characteristics of this variable template.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;
  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);
  if (ToD)
    return ToD;

  // We may already have a template of the same name; try to find and match it.
  assert(!DC->isFunctionOrMethod() &&
         "Variable templates cannot be declared at function scope");
  SmallVector<NamedDecl *, 4> ConflictingDecls;
  auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
  for (auto *FoundDecl : FoundDecls) {
    if (!FoundDecl->isInIdentifierNamespace(Decl::IDNS_Ordinary))
      continue;

    Decl *Found = FoundDecl;
    if (VarTemplateDecl *FoundTemplate = dyn_cast<VarTemplateDecl>(Found)) {
      if (IsStructuralMatch(D, FoundTemplate)) {
        // The variable templates structurally match; call it the same template.
        Importer.MapImported(D->getTemplatedDecl(),
                             FoundTemplate->getTemplatedDecl());
        return Importer.MapImported(D, FoundTemplate);
      }
      ConflictingDecls.push_back(FoundDecl);
    }
  }

  if (!ConflictingDecls.empty()) {
    ExpectedName NameOrErr = Importer.HandleNameConflict(
        Name, DC, Decl::IDNS_Ordinary, ConflictingDecls.data(),
        ConflictingDecls.size());
    if (NameOrErr)
      Name = NameOrErr.get();
    else
      return NameOrErr.takeError();
  }

  VarDecl *DTemplated = D->getTemplatedDecl();

  // Import the type.
  // FIXME: Value not used?
  ExpectedType TypeOrErr = import(DTemplated->getType());
  if (!TypeOrErr)
    return TypeOrErr.takeError();

  // Create the declaration that is being templated.
  VarDecl *ToTemplated;
  if (Error Err = importInto(ToTemplated, DTemplated))
    return std::move(Err);

  // Create the variable template declaration itself.
  auto TemplateParamsOrErr = import(D->getTemplateParameters());
  if (!TemplateParamsOrErr)
    return TemplateParamsOrErr.takeError();

  VarTemplateDecl *ToVarTD;
  if (GetImportedOrCreateDecl(ToVarTD, D, Importer.getToContext(), DC, Loc,
                              Name, *TemplateParamsOrErr, ToTemplated))
    return ToVarTD;

  ToTemplated->setDescribedVarTemplate(ToVarTD);

  ToVarTD->setAccess(D->getAccess());
  ToVarTD->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToVarTD);

  if (DTemplated->isThisDeclarationADefinition() &&
      !ToTemplated->isThisDeclarationADefinition()) {
    // FIXME: Import definition!
  }

  return ToVarTD;
}

ExpectedDecl ASTNodeImporter::VisitVarTemplateSpecializationDecl(
    VarTemplateSpecializationDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  VarDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    if (ExpectedDecl ImportedDefOrErr = import(Definition))
      return Importer.MapImported(D, *ImportedDefOrErr);
    else
      return ImportedDefOrErr.takeError();
  }

  VarTemplateDecl *VarTemplate = nullptr;
  if (Error Err = importInto(VarTemplate, D->getSpecializedTemplate()))
    return std::move(Err);

  // Import the context of this declaration.
  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  // Import the location of this declaration.
  ExpectedSLoc BeginLocOrErr = import(D->getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();

  auto IdLocOrErr = import(D->getLocation());
  if (!IdLocOrErr)
    return IdLocOrErr.takeError();

  // Import template arguments.
  SmallVector<TemplateArgument, 2> TemplateArgs;
  if (Error Err = ImportTemplateArguments(
      D->getTemplateArgs().data(), D->getTemplateArgs().size(), TemplateArgs))
    return std::move(Err);

  // Try to find an existing specialization with these template arguments.
  void *InsertPos = nullptr;
  VarTemplateSpecializationDecl *D2 = VarTemplate->findSpecialization(
      TemplateArgs, InsertPos);
  if (D2) {
    // We already have a variable template specialization with these template
    // arguments.

    // FIXME: Check for specialization vs. instantiation errors.

    if (VarDecl *FoundDef = D2->getDefinition()) {
      if (!D->isThisDeclarationADefinition() ||
          IsStructuralMatch(D, FoundDef)) {
        // The record types structurally match, or the "from" translation
        // unit only had a forward declaration anyway; call it the same
        // variable.
        return Importer.MapImported(D, FoundDef);
      }
    }
  } else {
    // Import the type.
    QualType T;
    if (Error Err = importInto(T, D->getType()))
      return std::move(Err);

    auto TInfoOrErr = import(D->getTypeSourceInfo());
    if (!TInfoOrErr)
      return TInfoOrErr.takeError();

    TemplateArgumentListInfo ToTAInfo;
    if (Error Err = ImportTemplateArgumentListInfo(
        D->getTemplateArgsInfo(), ToTAInfo))
      return std::move(Err);

    using PartVarSpecDecl = VarTemplatePartialSpecializationDecl;
    // Create a new specialization.
    if (auto *FromPartial = dyn_cast<PartVarSpecDecl>(D)) {
      // Import TemplateArgumentListInfo
      TemplateArgumentListInfo ArgInfos;
      const auto *FromTAArgsAsWritten = FromPartial->getTemplateArgsAsWritten();
      // NOTE: FromTAArgsAsWritten and template parameter list are non-null.
      if (Error Err = ImportTemplateArgumentListInfo(
          *FromTAArgsAsWritten, ArgInfos))
        return std::move(Err);

      auto ToTPListOrErr = import(FromPartial->getTemplateParameters());
      if (!ToTPListOrErr)
        return ToTPListOrErr.takeError();

      PartVarSpecDecl *ToPartial;
      if (GetImportedOrCreateDecl(ToPartial, D, Importer.getToContext(), DC,
                                  *BeginLocOrErr, *IdLocOrErr, *ToTPListOrErr,
                                  VarTemplate, T, *TInfoOrErr,
                                  D->getStorageClass(), TemplateArgs, ArgInfos))
        return ToPartial;

      if (Expected<PartVarSpecDecl *> ToInstOrErr = import(
          FromPartial->getInstantiatedFromMember()))
        ToPartial->setInstantiatedFromMember(*ToInstOrErr);
      else
        return ToInstOrErr.takeError();

      if (FromPartial->isMemberSpecialization())
        ToPartial->setMemberSpecialization();

      D2 = ToPartial;

    } else { // Full specialization
      if (GetImportedOrCreateDecl(D2, D, Importer.getToContext(), DC,
                                  *BeginLocOrErr, *IdLocOrErr, VarTemplate,
                                  T, *TInfoOrErr,
                                  D->getStorageClass(), TemplateArgs))
        return D2;
    }

    if (D->getPointOfInstantiation().isValid()) {
      if (ExpectedSLoc POIOrErr = import(D->getPointOfInstantiation()))
        D2->setPointOfInstantiation(*POIOrErr);
      else
        return POIOrErr.takeError();
    }

    D2->setSpecializationKind(D->getSpecializationKind());
    D2->setTemplateArgsInfo(ToTAInfo);

    // Add this specialization to the class template.
    VarTemplate->AddSpecialization(D2, InsertPos);

    // Import the qualifier, if any.
    if (auto LocOrErr = import(D->getQualifierLoc()))
      D2->setQualifierInfo(*LocOrErr);
    else
      return LocOrErr.takeError();

    if (D->isConstexpr())
      D2->setConstexpr(true);

    // Add the specialization to this context.
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDeclInternal(D2);

    D2->setAccess(D->getAccess());
  }

  if (Error Err = ImportInitializer(D, D2))
    return std::move(Err);

  return D2;
}

ExpectedDecl
ASTNodeImporter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  NamedDecl *ToD;

  if (Error Err = ImportDeclParts(D, DC, LexicalDC, Name, ToD, Loc))
    return std::move(Err);

  if (ToD)
    return ToD;

  const FunctionTemplateDecl *FoundByLookup = nullptr;

  // Try to find a function in our own ("to") context with the same name, same
  // type, and in the same context as the function we're importing.
  // FIXME Split this into a separate function.
  if (!LexicalDC->isFunctionOrMethod()) {
    unsigned IDNS = Decl::IDNS_Ordinary | Decl::IDNS_OrdinaryFriend;
    auto FoundDecls = Importer.findDeclsInToCtx(DC, Name);
    for (auto *FoundDecl : FoundDecls) {
      if (!FoundDecl->isInIdentifierNamespace(IDNS))
        continue;

      if (auto *FoundTemplate = dyn_cast<FunctionTemplateDecl>(FoundDecl)) {
        if (!hasSameVisibilityContext(FoundTemplate, D))
          continue;
        if (IsStructuralMatch(D, FoundTemplate)) {
          FunctionTemplateDecl *TemplateWithDef =
              getTemplateDefinition(FoundTemplate);
          if (D->isThisDeclarationADefinition() && TemplateWithDef)
            return Importer.MapImported(D, TemplateWithDef);

          FoundByLookup = FoundTemplate;
          break;
          // TODO: handle conflicting names
        }
      }
    }
  }

  auto ParamsOrErr = import(D->getTemplateParameters());
  if (!ParamsOrErr)
    return ParamsOrErr.takeError();

  FunctionDecl *TemplatedFD;
  if (Error Err = importInto(TemplatedFD, D->getTemplatedDecl()))
    return std::move(Err);

  FunctionTemplateDecl *ToFunc;
  if (GetImportedOrCreateDecl(ToFunc, D, Importer.getToContext(), DC, Loc, Name,
                              *ParamsOrErr, TemplatedFD))
    return ToFunc;

  TemplatedFD->setDescribedFunctionTemplate(ToFunc);

  ToFunc->setAccess(D->getAccess());
  ToFunc->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(ToFunc);

  if (FoundByLookup) {
    auto *Recent =
        const_cast<FunctionTemplateDecl *>(FoundByLookup->getMostRecentDecl());
    if (!TemplatedFD->getPreviousDecl()) {
      assert(FoundByLookup->getTemplatedDecl() &&
             "Found decl must have its templated decl set");
      auto *PrevTemplated =
          FoundByLookup->getTemplatedDecl()->getMostRecentDecl();
      if (TemplatedFD != PrevTemplated)
        TemplatedFD->setPreviousDecl(PrevTemplated);
    }
    ToFunc->setPreviousDecl(Recent);
  }

  return ToFunc;
}

//----------------------------------------------------------------------------
// Import Statements
//----------------------------------------------------------------------------

ExpectedStmt ASTNodeImporter::VisitStmt(Stmt *S) {
  Importer.FromDiag(S->getBeginLoc(), diag::err_unsupported_ast_node)
      << S->getStmtClassName();
  return make_error<ImportError>(ImportError::UnsupportedConstruct);
}


ExpectedStmt ASTNodeImporter::VisitGCCAsmStmt(GCCAsmStmt *S) {
  if (Importer.returnWithErrorInTest())
    return make_error<ImportError>(ImportError::UnsupportedConstruct);
  SmallVector<IdentifierInfo *, 4> Names;
  for (unsigned I = 0, E = S->getNumOutputs(); I != E; I++) {
    IdentifierInfo *ToII = Importer.Import(S->getOutputIdentifier(I));
    // ToII is nullptr when no symbolic name is given for output operand
    // see ParseStmtAsm::ParseAsmOperandsOpt
    Names.push_back(ToII);
  }

  for (unsigned I = 0, E = S->getNumInputs(); I != E; I++) {
    IdentifierInfo *ToII = Importer.Import(S->getInputIdentifier(I));
    // ToII is nullptr when no symbolic name is given for input operand
    // see ParseStmtAsm::ParseAsmOperandsOpt
    Names.push_back(ToII);
  }

  SmallVector<StringLiteral *, 4> Clobbers;
  for (unsigned I = 0, E = S->getNumClobbers(); I != E; I++) {
    if (auto ClobberOrErr = import(S->getClobberStringLiteral(I)))
      Clobbers.push_back(*ClobberOrErr);
    else
      return ClobberOrErr.takeError();

  }

  SmallVector<StringLiteral *, 4> Constraints;
  for (unsigned I = 0, E = S->getNumOutputs(); I != E; I++) {
    if (auto OutputOrErr = import(S->getOutputConstraintLiteral(I)))
      Constraints.push_back(*OutputOrErr);
    else
      return OutputOrErr.takeError();
  }

  for (unsigned I = 0, E = S->getNumInputs(); I != E; I++) {
    if (auto InputOrErr = import(S->getInputConstraintLiteral(I)))
      Constraints.push_back(*InputOrErr);
    else
      return InputOrErr.takeError();
  }

  SmallVector<Expr *, 4> Exprs(S->getNumOutputs() + S->getNumInputs() +
                               S->getNumLabels());
  if (Error Err = ImportContainerChecked(S->outputs(), Exprs))
    return std::move(Err);

  if (Error Err =
          ImportArrayChecked(S->inputs(), Exprs.begin() + S->getNumOutputs()))
    return std::move(Err);

  if (Error Err = ImportArrayChecked(
          S->labels(), Exprs.begin() + S->getNumOutputs() + S->getNumInputs()))
    return std::move(Err);

  ExpectedSLoc AsmLocOrErr = import(S->getAsmLoc());
  if (!AsmLocOrErr)
    return AsmLocOrErr.takeError();
  auto AsmStrOrErr = import(S->getAsmString());
  if (!AsmStrOrErr)
    return AsmStrOrErr.takeError();
  ExpectedSLoc RParenLocOrErr = import(S->getRParenLoc());
  if (!RParenLocOrErr)
    return RParenLocOrErr.takeError();

  return new (Importer.getToContext()) GCCAsmStmt(
      Importer.getToContext(),
      *AsmLocOrErr,
      S->isSimple(),
      S->isVolatile(),
      S->getNumOutputs(),
      S->getNumInputs(),
      Names.data(),
      Constraints.data(),
      Exprs.data(),
      *AsmStrOrErr,
      S->getNumClobbers(),
      Clobbers.data(),
      S->getNumLabels(),
      *RParenLocOrErr);
}

ExpectedStmt ASTNodeImporter::VisitDeclStmt(DeclStmt *S) {
  auto Imp = importSeq(S->getDeclGroup(), S->getBeginLoc(), S->getEndLoc());
  if (!Imp)
    return Imp.takeError();

  DeclGroupRef ToDG;
  SourceLocation ToBeginLoc, ToEndLoc;
  std::tie(ToDG, ToBeginLoc, ToEndLoc) = *Imp;

  return new (Importer.getToContext()) DeclStmt(ToDG, ToBeginLoc, ToEndLoc);
}

ExpectedStmt ASTNodeImporter::VisitNullStmt(NullStmt *S) {
  ExpectedSLoc ToSemiLocOrErr = import(S->getSemiLoc());
  if (!ToSemiLocOrErr)
    return ToSemiLocOrErr.takeError();
  return new (Importer.getToContext()) NullStmt(
      *ToSemiLocOrErr, S->hasLeadingEmptyMacro());
}

ExpectedStmt ASTNodeImporter::VisitCompoundStmt(CompoundStmt *S) {
  SmallVector<Stmt *, 8> ToStmts(S->size());

  if (Error Err = ImportContainerChecked(S->body(), ToStmts))
    return std::move(Err);

  ExpectedSLoc ToLBracLocOrErr = import(S->getLBracLoc());
  if (!ToLBracLocOrErr)
    return ToLBracLocOrErr.takeError();

  ExpectedSLoc ToRBracLocOrErr = import(S->getRBracLoc());
  if (!ToRBracLocOrErr)
    return ToRBracLocOrErr.takeError();

  return CompoundStmt::Create(
      Importer.getToContext(), ToStmts,
      *ToLBracLocOrErr, *ToRBracLocOrErr);
}

ExpectedStmt ASTNodeImporter::VisitCaseStmt(CaseStmt *S) {
  auto Imp = importSeq(
      S->getLHS(), S->getRHS(), S->getSubStmt(), S->getCaseLoc(),
      S->getEllipsisLoc(), S->getColonLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToLHS, *ToRHS;
  Stmt *ToSubStmt;
  SourceLocation ToCaseLoc, ToEllipsisLoc, ToColonLoc;
  std::tie(ToLHS, ToRHS, ToSubStmt, ToCaseLoc, ToEllipsisLoc, ToColonLoc) =
      *Imp;

  auto *ToStmt = CaseStmt::Create(Importer.getToContext(), ToLHS, ToRHS,
                                  ToCaseLoc, ToEllipsisLoc, ToColonLoc);
  ToStmt->setSubStmt(ToSubStmt);

  return ToStmt;
}

ExpectedStmt ASTNodeImporter::VisitDefaultStmt(DefaultStmt *S) {
  auto Imp = importSeq(S->getDefaultLoc(), S->getColonLoc(), S->getSubStmt());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToDefaultLoc, ToColonLoc;
  Stmt *ToSubStmt;
  std::tie(ToDefaultLoc, ToColonLoc, ToSubStmt) = *Imp;

  return new (Importer.getToContext()) DefaultStmt(
    ToDefaultLoc, ToColonLoc, ToSubStmt);
}

ExpectedStmt ASTNodeImporter::VisitLabelStmt(LabelStmt *S) {
  auto Imp = importSeq(S->getIdentLoc(), S->getDecl(), S->getSubStmt());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToIdentLoc;
  LabelDecl *ToLabelDecl;
  Stmt *ToSubStmt;
  std::tie(ToIdentLoc, ToLabelDecl, ToSubStmt) = *Imp;

  return new (Importer.getToContext()) LabelStmt(
      ToIdentLoc, ToLabelDecl, ToSubStmt);
}

ExpectedStmt ASTNodeImporter::VisitAttributedStmt(AttributedStmt *S) {
  ExpectedSLoc ToAttrLocOrErr = import(S->getAttrLoc());
  if (!ToAttrLocOrErr)
    return ToAttrLocOrErr.takeError();
  ArrayRef<const Attr*> FromAttrs(S->getAttrs());
  SmallVector<const Attr *, 1> ToAttrs(FromAttrs.size());
  if (Error Err = ImportContainerChecked(FromAttrs, ToAttrs))
    return std::move(Err);
  ExpectedStmt ToSubStmtOrErr = import(S->getSubStmt());
  if (!ToSubStmtOrErr)
    return ToSubStmtOrErr.takeError();

  return AttributedStmt::Create(
      Importer.getToContext(), *ToAttrLocOrErr, ToAttrs, *ToSubStmtOrErr);
}

ExpectedStmt ASTNodeImporter::VisitIfStmt(IfStmt *S) {
  auto Imp = importSeq(
      S->getIfLoc(), S->getInit(), S->getConditionVariable(), S->getCond(),
      S->getThen(), S->getElseLoc(), S->getElse());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToIfLoc, ToElseLoc;
  Stmt *ToInit, *ToThen, *ToElse;
  VarDecl *ToConditionVariable;
  Expr *ToCond;
  std::tie(
      ToIfLoc, ToInit, ToConditionVariable, ToCond, ToThen, ToElseLoc, ToElse) =
          *Imp;

  return IfStmt::Create(Importer.getToContext(), ToIfLoc, S->isConstexpr(),
                        ToInit, ToConditionVariable, ToCond, ToThen, ToElseLoc,
                        ToElse);
}

ExpectedStmt ASTNodeImporter::VisitSwitchStmt(SwitchStmt *S) {
  auto Imp = importSeq(
      S->getInit(), S->getConditionVariable(), S->getCond(),
      S->getBody(), S->getSwitchLoc());
  if (!Imp)
    return Imp.takeError();

  Stmt *ToInit, *ToBody;
  VarDecl *ToConditionVariable;
  Expr *ToCond;
  SourceLocation ToSwitchLoc;
  std::tie(ToInit, ToConditionVariable, ToCond, ToBody, ToSwitchLoc) = *Imp;

  auto *ToStmt = SwitchStmt::Create(Importer.getToContext(), ToInit,
                                    ToConditionVariable, ToCond);
  ToStmt->setBody(ToBody);
  ToStmt->setSwitchLoc(ToSwitchLoc);

  // Now we have to re-chain the cases.
  SwitchCase *LastChainedSwitchCase = nullptr;
  for (SwitchCase *SC = S->getSwitchCaseList(); SC != nullptr;
       SC = SC->getNextSwitchCase()) {
    Expected<SwitchCase *> ToSCOrErr = import(SC);
    if (!ToSCOrErr)
      return ToSCOrErr.takeError();
    if (LastChainedSwitchCase)
      LastChainedSwitchCase->setNextSwitchCase(*ToSCOrErr);
    else
      ToStmt->setSwitchCaseList(*ToSCOrErr);
    LastChainedSwitchCase = *ToSCOrErr;
  }

  return ToStmt;
}

ExpectedStmt ASTNodeImporter::VisitWhileStmt(WhileStmt *S) {
  auto Imp = importSeq(
      S->getConditionVariable(), S->getCond(), S->getBody(), S->getWhileLoc());
  if (!Imp)
    return Imp.takeError();

  VarDecl *ToConditionVariable;
  Expr *ToCond;
  Stmt *ToBody;
  SourceLocation ToWhileLoc;
  std::tie(ToConditionVariable, ToCond, ToBody, ToWhileLoc) = *Imp;

  return WhileStmt::Create(Importer.getToContext(), ToConditionVariable, ToCond,
                           ToBody, ToWhileLoc);
}

ExpectedStmt ASTNodeImporter::VisitDoStmt(DoStmt *S) {
  auto Imp = importSeq(
      S->getBody(), S->getCond(), S->getDoLoc(), S->getWhileLoc(),
      S->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  Stmt *ToBody;
  Expr *ToCond;
  SourceLocation ToDoLoc, ToWhileLoc, ToRParenLoc;
  std::tie(ToBody, ToCond, ToDoLoc, ToWhileLoc, ToRParenLoc) = *Imp;

  return new (Importer.getToContext()) DoStmt(
      ToBody, ToCond, ToDoLoc, ToWhileLoc, ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitForStmt(ForStmt *S) {
  auto Imp = importSeq(
      S->getInit(), S->getCond(), S->getConditionVariable(), S->getInc(),
      S->getBody(), S->getForLoc(), S->getLParenLoc(), S->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  Stmt *ToInit;
  Expr *ToCond, *ToInc;
  VarDecl *ToConditionVariable;
  Stmt *ToBody;
  SourceLocation ToForLoc, ToLParenLoc, ToRParenLoc;
  std::tie(
      ToInit, ToCond, ToConditionVariable,  ToInc, ToBody, ToForLoc,
      ToLParenLoc, ToRParenLoc) = *Imp;

  return new (Importer.getToContext()) ForStmt(
      Importer.getToContext(),
      ToInit, ToCond, ToConditionVariable, ToInc, ToBody, ToForLoc, ToLParenLoc,
      ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitGotoStmt(GotoStmt *S) {
  auto Imp = importSeq(S->getLabel(), S->getGotoLoc(), S->getLabelLoc());
  if (!Imp)
    return Imp.takeError();

  LabelDecl *ToLabel;
  SourceLocation ToGotoLoc, ToLabelLoc;
  std::tie(ToLabel, ToGotoLoc, ToLabelLoc) = *Imp;

  return new (Importer.getToContext()) GotoStmt(
      ToLabel, ToGotoLoc, ToLabelLoc);
}

ExpectedStmt ASTNodeImporter::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  auto Imp = importSeq(S->getGotoLoc(), S->getStarLoc(), S->getTarget());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToGotoLoc, ToStarLoc;
  Expr *ToTarget;
  std::tie(ToGotoLoc, ToStarLoc, ToTarget) = *Imp;

  return new (Importer.getToContext()) IndirectGotoStmt(
      ToGotoLoc, ToStarLoc, ToTarget);
}

ExpectedStmt ASTNodeImporter::VisitContinueStmt(ContinueStmt *S) {
  ExpectedSLoc ToContinueLocOrErr = import(S->getContinueLoc());
  if (!ToContinueLocOrErr)
    return ToContinueLocOrErr.takeError();
  return new (Importer.getToContext()) ContinueStmt(*ToContinueLocOrErr);
}

ExpectedStmt ASTNodeImporter::VisitBreakStmt(BreakStmt *S) {
  auto ToBreakLocOrErr = import(S->getBreakLoc());
  if (!ToBreakLocOrErr)
    return ToBreakLocOrErr.takeError();
  return new (Importer.getToContext()) BreakStmt(*ToBreakLocOrErr);
}

ExpectedStmt ASTNodeImporter::VisitReturnStmt(ReturnStmt *S) {
  auto Imp = importSeq(
      S->getReturnLoc(), S->getRetValue(), S->getNRVOCandidate());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToReturnLoc;
  Expr *ToRetValue;
  const VarDecl *ToNRVOCandidate;
  std::tie(ToReturnLoc, ToRetValue, ToNRVOCandidate) = *Imp;

  return ReturnStmt::Create(Importer.getToContext(), ToReturnLoc, ToRetValue,
                            ToNRVOCandidate);
}

ExpectedStmt ASTNodeImporter::VisitCXXCatchStmt(CXXCatchStmt *S) {
  auto Imp = importSeq(
      S->getCatchLoc(), S->getExceptionDecl(), S->getHandlerBlock());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToCatchLoc;
  VarDecl *ToExceptionDecl;
  Stmt *ToHandlerBlock;
  std::tie(ToCatchLoc, ToExceptionDecl, ToHandlerBlock) = *Imp;

  return new (Importer.getToContext()) CXXCatchStmt (
      ToCatchLoc, ToExceptionDecl, ToHandlerBlock);
}

ExpectedStmt ASTNodeImporter::VisitCXXTryStmt(CXXTryStmt *S) {
  ExpectedSLoc ToTryLocOrErr = import(S->getTryLoc());
  if (!ToTryLocOrErr)
    return ToTryLocOrErr.takeError();

  ExpectedStmt ToTryBlockOrErr = import(S->getTryBlock());
  if (!ToTryBlockOrErr)
    return ToTryBlockOrErr.takeError();

  SmallVector<Stmt *, 1> ToHandlers(S->getNumHandlers());
  for (unsigned HI = 0, HE = S->getNumHandlers(); HI != HE; ++HI) {
    CXXCatchStmt *FromHandler = S->getHandler(HI);
    if (auto ToHandlerOrErr = import(FromHandler))
      ToHandlers[HI] = *ToHandlerOrErr;
    else
      return ToHandlerOrErr.takeError();
  }

  return CXXTryStmt::Create(
      Importer.getToContext(), *ToTryLocOrErr,*ToTryBlockOrErr, ToHandlers);
}

ExpectedStmt ASTNodeImporter::VisitCXXForRangeStmt(CXXForRangeStmt *S) {
  auto Imp1 = importSeq(
      S->getInit(), S->getRangeStmt(), S->getBeginStmt(), S->getEndStmt(),
      S->getCond(), S->getInc(), S->getLoopVarStmt(), S->getBody());
  if (!Imp1)
    return Imp1.takeError();
  auto Imp2 = importSeq(
      S->getForLoc(), S->getCoawaitLoc(), S->getColonLoc(), S->getRParenLoc());
  if (!Imp2)
    return Imp2.takeError();

  DeclStmt *ToRangeStmt, *ToBeginStmt, *ToEndStmt, *ToLoopVarStmt;
  Expr *ToCond, *ToInc;
  Stmt *ToInit, *ToBody;
  std::tie(
      ToInit, ToRangeStmt, ToBeginStmt, ToEndStmt, ToCond, ToInc, ToLoopVarStmt,
      ToBody) = *Imp1;
  SourceLocation ToForLoc, ToCoawaitLoc, ToColonLoc, ToRParenLoc;
  std::tie(ToForLoc, ToCoawaitLoc, ToColonLoc, ToRParenLoc) = *Imp2;

  return new (Importer.getToContext()) CXXForRangeStmt(
      ToInit, ToRangeStmt, ToBeginStmt, ToEndStmt, ToCond, ToInc, ToLoopVarStmt,
      ToBody, ToForLoc, ToCoawaitLoc, ToColonLoc, ToRParenLoc);
}

ExpectedStmt
ASTNodeImporter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *S) {
  auto Imp = importSeq(
      S->getElement(), S->getCollection(), S->getBody(),
      S->getForLoc(), S->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  Stmt *ToElement, *ToBody;
  Expr *ToCollection;
  SourceLocation ToForLoc, ToRParenLoc;
  std::tie(ToElement, ToCollection, ToBody, ToForLoc, ToRParenLoc) = *Imp;

  return new (Importer.getToContext()) ObjCForCollectionStmt(ToElement,
                                                             ToCollection,
                                                             ToBody,
                                                             ToForLoc,
                                                             ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  auto Imp = importSeq(
      S->getAtCatchLoc(), S->getRParenLoc(), S->getCatchParamDecl(),
      S->getCatchBody());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToAtCatchLoc, ToRParenLoc;
  VarDecl *ToCatchParamDecl;
  Stmt *ToCatchBody;
  std::tie(ToAtCatchLoc, ToRParenLoc, ToCatchParamDecl, ToCatchBody) = *Imp;

  return new (Importer.getToContext()) ObjCAtCatchStmt (
      ToAtCatchLoc, ToRParenLoc, ToCatchParamDecl, ToCatchBody);
}

ExpectedStmt ASTNodeImporter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  ExpectedSLoc ToAtFinallyLocOrErr = import(S->getAtFinallyLoc());
  if (!ToAtFinallyLocOrErr)
    return ToAtFinallyLocOrErr.takeError();
  ExpectedStmt ToAtFinallyStmtOrErr = import(S->getFinallyBody());
  if (!ToAtFinallyStmtOrErr)
    return ToAtFinallyStmtOrErr.takeError();
  return new (Importer.getToContext()) ObjCAtFinallyStmt(*ToAtFinallyLocOrErr,
                                                         *ToAtFinallyStmtOrErr);
}

ExpectedStmt ASTNodeImporter::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  auto Imp = importSeq(
      S->getAtTryLoc(), S->getTryBody(), S->getFinallyStmt());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToAtTryLoc;
  Stmt *ToTryBody, *ToFinallyStmt;
  std::tie(ToAtTryLoc, ToTryBody, ToFinallyStmt) = *Imp;

  SmallVector<Stmt *, 1> ToCatchStmts(S->getNumCatchStmts());
  for (unsigned CI = 0, CE = S->getNumCatchStmts(); CI != CE; ++CI) {
    ObjCAtCatchStmt *FromCatchStmt = S->getCatchStmt(CI);
    if (ExpectedStmt ToCatchStmtOrErr = import(FromCatchStmt))
      ToCatchStmts[CI] = *ToCatchStmtOrErr;
    else
      return ToCatchStmtOrErr.takeError();
  }

  return ObjCAtTryStmt::Create(Importer.getToContext(),
                               ToAtTryLoc, ToTryBody,
                               ToCatchStmts.begin(), ToCatchStmts.size(),
                               ToFinallyStmt);
}

ExpectedStmt ASTNodeImporter::VisitObjCAtSynchronizedStmt
  (ObjCAtSynchronizedStmt *S) {
  auto Imp = importSeq(
      S->getAtSynchronizedLoc(), S->getSynchExpr(), S->getSynchBody());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToAtSynchronizedLoc;
  Expr *ToSynchExpr;
  Stmt *ToSynchBody;
  std::tie(ToAtSynchronizedLoc, ToSynchExpr, ToSynchBody) = *Imp;

  return new (Importer.getToContext()) ObjCAtSynchronizedStmt(
    ToAtSynchronizedLoc, ToSynchExpr, ToSynchBody);
}

ExpectedStmt ASTNodeImporter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  ExpectedSLoc ToThrowLocOrErr = import(S->getThrowLoc());
  if (!ToThrowLocOrErr)
    return ToThrowLocOrErr.takeError();
  ExpectedExpr ToThrowExprOrErr = import(S->getThrowExpr());
  if (!ToThrowExprOrErr)
    return ToThrowExprOrErr.takeError();
  return new (Importer.getToContext()) ObjCAtThrowStmt(
      *ToThrowLocOrErr, *ToThrowExprOrErr);
}

ExpectedStmt ASTNodeImporter::VisitObjCAutoreleasePoolStmt(
    ObjCAutoreleasePoolStmt *S) {
  ExpectedSLoc ToAtLocOrErr = import(S->getAtLoc());
  if (!ToAtLocOrErr)
    return ToAtLocOrErr.takeError();
  ExpectedStmt ToSubStmtOrErr = import(S->getSubStmt());
  if (!ToSubStmtOrErr)
    return ToSubStmtOrErr.takeError();
  return new (Importer.getToContext()) ObjCAutoreleasePoolStmt(*ToAtLocOrErr,
                                                               *ToSubStmtOrErr);
}

//----------------------------------------------------------------------------
// Import Expressions
//----------------------------------------------------------------------------
ExpectedStmt ASTNodeImporter::VisitExpr(Expr *E) {
  Importer.FromDiag(E->getBeginLoc(), diag::err_unsupported_ast_node)
      << E->getStmtClassName();
  return make_error<ImportError>(ImportError::UnsupportedConstruct);
}

ExpectedStmt ASTNodeImporter::VisitVAArgExpr(VAArgExpr *E) {
  auto Imp = importSeq(
      E->getBuiltinLoc(), E->getSubExpr(), E->getWrittenTypeInfo(),
      E->getRParenLoc(), E->getType());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToBuiltinLoc, ToRParenLoc;
  Expr *ToSubExpr;
  TypeSourceInfo *ToWrittenTypeInfo;
  QualType ToType;
  std::tie(ToBuiltinLoc, ToSubExpr, ToWrittenTypeInfo, ToRParenLoc, ToType) =
      *Imp;

  return new (Importer.getToContext()) VAArgExpr(
      ToBuiltinLoc, ToSubExpr, ToWrittenTypeInfo, ToRParenLoc, ToType,
      E->isMicrosoftABI());
}

ExpectedStmt ASTNodeImporter::VisitChooseExpr(ChooseExpr *E) {
  auto Imp = importSeq(E->getCond(), E->getLHS(), E->getRHS(),
                       E->getBuiltinLoc(), E->getRParenLoc(), E->getType());
  if (!Imp)
    return Imp.takeError();

  Expr *ToCond;
  Expr *ToLHS;
  Expr *ToRHS;
  SourceLocation ToBuiltinLoc, ToRParenLoc;
  QualType ToType;
  std::tie(ToCond, ToLHS, ToRHS, ToBuiltinLoc, ToRParenLoc, ToType) = *Imp;

  ExprValueKind VK = E->getValueKind();
  ExprObjectKind OK = E->getObjectKind();

  bool TypeDependent = ToCond->isTypeDependent();
  bool ValueDependent = ToCond->isValueDependent();

  // The value of CondIsTrue only matters if the value is not
  // condition-dependent.
  bool CondIsTrue = !E->isConditionDependent() && E->isConditionTrue();

  return new (Importer.getToContext())
      ChooseExpr(ToBuiltinLoc, ToCond, ToLHS, ToRHS, ToType, VK, OK,
                 ToRParenLoc, CondIsTrue, TypeDependent, ValueDependent);
}

ExpectedStmt ASTNodeImporter::VisitGNUNullExpr(GNUNullExpr *E) {
  ExpectedType TypeOrErr = import(E->getType());
  if (!TypeOrErr)
    return TypeOrErr.takeError();

  ExpectedSLoc BeginLocOrErr = import(E->getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();

  return new (Importer.getToContext()) GNUNullExpr(*TypeOrErr, *BeginLocOrErr);
}

ExpectedStmt ASTNodeImporter::VisitPredefinedExpr(PredefinedExpr *E) {
  auto Imp = importSeq(
      E->getBeginLoc(), E->getType(), E->getFunctionName());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToBeginLoc;
  QualType ToType;
  StringLiteral *ToFunctionName;
  std::tie(ToBeginLoc, ToType, ToFunctionName) = *Imp;

  return PredefinedExpr::Create(Importer.getToContext(), ToBeginLoc, ToType,
                                E->getIdentKind(), ToFunctionName);
}

ExpectedStmt ASTNodeImporter::VisitDeclRefExpr(DeclRefExpr *E) {
  auto Imp = importSeq(
      E->getQualifierLoc(), E->getTemplateKeywordLoc(), E->getDecl(),
      E->getLocation(), E->getType());
  if (!Imp)
    return Imp.takeError();

  NestedNameSpecifierLoc ToQualifierLoc;
  SourceLocation ToTemplateKeywordLoc, ToLocation;
  ValueDecl *ToDecl;
  QualType ToType;
  std::tie(ToQualifierLoc, ToTemplateKeywordLoc, ToDecl, ToLocation, ToType) =
      *Imp;

  NamedDecl *ToFoundD = nullptr;
  if (E->getDecl() != E->getFoundDecl()) {
    auto FoundDOrErr = import(E->getFoundDecl());
    if (!FoundDOrErr)
      return FoundDOrErr.takeError();
    ToFoundD = *FoundDOrErr;
  }

  TemplateArgumentListInfo ToTAInfo;
  TemplateArgumentListInfo *ToResInfo = nullptr;
  if (E->hasExplicitTemplateArgs()) {
    if (Error Err =
            ImportTemplateArgumentListInfo(E->getLAngleLoc(), E->getRAngleLoc(),
                                           E->template_arguments(), ToTAInfo))
      return std::move(Err);
    ToResInfo = &ToTAInfo;
  }

  auto *ToE = DeclRefExpr::Create(
      Importer.getToContext(), ToQualifierLoc, ToTemplateKeywordLoc, ToDecl,
      E->refersToEnclosingVariableOrCapture(), ToLocation, ToType,
      E->getValueKind(), ToFoundD, ToResInfo, E->isNonOdrUse());
  if (E->hadMultipleCandidates())
    ToE->setHadMultipleCandidates(true);
  return ToE;
}

ExpectedStmt ASTNodeImporter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  ExpectedType TypeOrErr = import(E->getType());
  if (!TypeOrErr)
    return TypeOrErr.takeError();

  return new (Importer.getToContext()) ImplicitValueInitExpr(*TypeOrErr);
}

ExpectedStmt ASTNodeImporter::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  ExpectedExpr ToInitOrErr = import(E->getInit());
  if (!ToInitOrErr)
    return ToInitOrErr.takeError();

  ExpectedSLoc ToEqualOrColonLocOrErr = import(E->getEqualOrColonLoc());
  if (!ToEqualOrColonLocOrErr)
    return ToEqualOrColonLocOrErr.takeError();

  SmallVector<Expr *, 4> ToIndexExprs(E->getNumSubExprs() - 1);
  // List elements from the second, the first is Init itself
  for (unsigned I = 1, N = E->getNumSubExprs(); I < N; I++) {
    if (ExpectedExpr ToArgOrErr = import(E->getSubExpr(I)))
      ToIndexExprs[I - 1] = *ToArgOrErr;
    else
      return ToArgOrErr.takeError();
  }

  SmallVector<Designator, 4> ToDesignators(E->size());
  if (Error Err = ImportContainerChecked(E->designators(), ToDesignators))
    return std::move(Err);

  return DesignatedInitExpr::Create(
        Importer.getToContext(), ToDesignators,
        ToIndexExprs, *ToEqualOrColonLocOrErr,
        E->usesGNUSyntax(), *ToInitOrErr);
}

ExpectedStmt
ASTNodeImporter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedSLoc ToLocationOrErr = import(E->getLocation());
  if (!ToLocationOrErr)
    return ToLocationOrErr.takeError();

  return new (Importer.getToContext()) CXXNullPtrLiteralExpr(
      *ToTypeOrErr, *ToLocationOrErr);
}

ExpectedStmt ASTNodeImporter::VisitIntegerLiteral(IntegerLiteral *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedSLoc ToLocationOrErr = import(E->getLocation());
  if (!ToLocationOrErr)
    return ToLocationOrErr.takeError();

  return IntegerLiteral::Create(
      Importer.getToContext(), E->getValue(), *ToTypeOrErr, *ToLocationOrErr);
}


ExpectedStmt ASTNodeImporter::VisitFloatingLiteral(FloatingLiteral *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedSLoc ToLocationOrErr = import(E->getLocation());
  if (!ToLocationOrErr)
    return ToLocationOrErr.takeError();

  return FloatingLiteral::Create(
      Importer.getToContext(), E->getValue(), E->isExact(),
      *ToTypeOrErr, *ToLocationOrErr);
}

ExpectedStmt ASTNodeImporter::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  auto ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedExpr ToSubExprOrErr = import(E->getSubExpr());
  if (!ToSubExprOrErr)
    return ToSubExprOrErr.takeError();

  return new (Importer.getToContext()) ImaginaryLiteral(
      *ToSubExprOrErr, *ToTypeOrErr);
}

ExpectedStmt ASTNodeImporter::VisitCharacterLiteral(CharacterLiteral *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedSLoc ToLocationOrErr = import(E->getLocation());
  if (!ToLocationOrErr)
    return ToLocationOrErr.takeError();

  return new (Importer.getToContext()) CharacterLiteral(
      E->getValue(), E->getKind(), *ToTypeOrErr, *ToLocationOrErr);
}

ExpectedStmt ASTNodeImporter::VisitStringLiteral(StringLiteral *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  SmallVector<SourceLocation, 4> ToLocations(E->getNumConcatenated());
  if (Error Err = ImportArrayChecked(
      E->tokloc_begin(), E->tokloc_end(), ToLocations.begin()))
    return std::move(Err);

  return StringLiteral::Create(
      Importer.getToContext(), E->getBytes(), E->getKind(), E->isPascal(),
      *ToTypeOrErr, ToLocations.data(), ToLocations.size());
}

ExpectedStmt ASTNodeImporter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  auto Imp = importSeq(
      E->getLParenLoc(), E->getTypeSourceInfo(), E->getType(),
      E->getInitializer());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToLParenLoc;
  TypeSourceInfo *ToTypeSourceInfo;
  QualType ToType;
  Expr *ToInitializer;
  std::tie(ToLParenLoc, ToTypeSourceInfo, ToType, ToInitializer) = *Imp;

  return new (Importer.getToContext()) CompoundLiteralExpr(
        ToLParenLoc, ToTypeSourceInfo, ToType, E->getValueKind(),
        ToInitializer, E->isFileScope());
}

ExpectedStmt ASTNodeImporter::VisitAtomicExpr(AtomicExpr *E) {
  auto Imp = importSeq(
      E->getBuiltinLoc(), E->getType(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToBuiltinLoc, ToRParenLoc;
  QualType ToType;
  std::tie(ToBuiltinLoc, ToType, ToRParenLoc) = *Imp;

  SmallVector<Expr *, 6> ToExprs(E->getNumSubExprs());
  if (Error Err = ImportArrayChecked(
      E->getSubExprs(), E->getSubExprs() + E->getNumSubExprs(),
      ToExprs.begin()))
    return std::move(Err);

  return new (Importer.getToContext()) AtomicExpr(
      ToBuiltinLoc, ToExprs, ToType, E->getOp(), ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitAddrLabelExpr(AddrLabelExpr *E) {
  auto Imp = importSeq(
      E->getAmpAmpLoc(), E->getLabelLoc(), E->getLabel(), E->getType());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToAmpAmpLoc, ToLabelLoc;
  LabelDecl *ToLabel;
  QualType ToType;
  std::tie(ToAmpAmpLoc, ToLabelLoc, ToLabel, ToType) = *Imp;

  return new (Importer.getToContext()) AddrLabelExpr(
      ToAmpAmpLoc, ToLabelLoc, ToLabel, ToType);
}

ExpectedStmt ASTNodeImporter::VisitConstantExpr(ConstantExpr *E) {
  auto Imp = importSeq(E->getSubExpr());
  if (!Imp)
    return Imp.takeError();

  Expr *ToSubExpr;
  std::tie(ToSubExpr) = *Imp;

  // TODO : Handle APValue::ValueKind that require importing.
  APValue::ValueKind Kind = E->getResultAPValueKind();
  if (Kind == APValue::Int || Kind == APValue::Float ||
      Kind == APValue::FixedPoint || Kind == APValue::ComplexFloat ||
      Kind == APValue::ComplexInt)
    return ConstantExpr::Create(Importer.getToContext(), ToSubExpr,
                                E->getAPValueResult());
  return ConstantExpr::Create(Importer.getToContext(), ToSubExpr);
}

ExpectedStmt ASTNodeImporter::VisitParenExpr(ParenExpr *E) {
  auto Imp = importSeq(E->getLParen(), E->getRParen(), E->getSubExpr());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToLParen, ToRParen;
  Expr *ToSubExpr;
  std::tie(ToLParen, ToRParen, ToSubExpr) = *Imp;

  return new (Importer.getToContext())
      ParenExpr(ToLParen, ToRParen, ToSubExpr);
}

ExpectedStmt ASTNodeImporter::VisitParenListExpr(ParenListExpr *E) {
  SmallVector<Expr *, 4> ToExprs(E->getNumExprs());
  if (Error Err = ImportContainerChecked(E->exprs(), ToExprs))
    return std::move(Err);

  ExpectedSLoc ToLParenLocOrErr = import(E->getLParenLoc());
  if (!ToLParenLocOrErr)
    return ToLParenLocOrErr.takeError();

  ExpectedSLoc ToRParenLocOrErr = import(E->getRParenLoc());
  if (!ToRParenLocOrErr)
    return ToRParenLocOrErr.takeError();

  return ParenListExpr::Create(Importer.getToContext(), *ToLParenLocOrErr,
                               ToExprs, *ToRParenLocOrErr);
}

ExpectedStmt ASTNodeImporter::VisitStmtExpr(StmtExpr *E) {
  auto Imp = importSeq(
      E->getSubStmt(), E->getType(), E->getLParenLoc(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  CompoundStmt *ToSubStmt;
  QualType ToType;
  SourceLocation ToLParenLoc, ToRParenLoc;
  std::tie(ToSubStmt, ToType, ToLParenLoc, ToRParenLoc) = *Imp;

  return new (Importer.getToContext()) StmtExpr(
      ToSubStmt, ToType, ToLParenLoc, ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitUnaryOperator(UnaryOperator *E) {
  auto Imp = importSeq(
      E->getSubExpr(), E->getType(), E->getOperatorLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToSubExpr;
  QualType ToType;
  SourceLocation ToOperatorLoc;
  std::tie(ToSubExpr, ToType, ToOperatorLoc) = *Imp;

  return new (Importer.getToContext()) UnaryOperator(
      ToSubExpr, E->getOpcode(), ToType, E->getValueKind(), E->getObjectKind(),
      ToOperatorLoc, E->canOverflow());
}

ExpectedStmt
ASTNodeImporter::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
  auto Imp = importSeq(E->getType(), E->getOperatorLoc(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  SourceLocation ToOperatorLoc, ToRParenLoc;
  std::tie(ToType, ToOperatorLoc, ToRParenLoc) = *Imp;

  if (E->isArgumentType()) {
    Expected<TypeSourceInfo *> ToArgumentTypeInfoOrErr =
        import(E->getArgumentTypeInfo());
    if (!ToArgumentTypeInfoOrErr)
      return ToArgumentTypeInfoOrErr.takeError();

    return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(
        E->getKind(), *ToArgumentTypeInfoOrErr, ToType, ToOperatorLoc,
        ToRParenLoc);
  }

  ExpectedExpr ToArgumentExprOrErr = import(E->getArgumentExpr());
  if (!ToArgumentExprOrErr)
    return ToArgumentExprOrErr.takeError();

  return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(
      E->getKind(), *ToArgumentExprOrErr, ToType, ToOperatorLoc, ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitBinaryOperator(BinaryOperator *E) {
  auto Imp = importSeq(
      E->getLHS(), E->getRHS(), E->getType(), E->getOperatorLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToLHS, *ToRHS;
  QualType ToType;
  SourceLocation  ToOperatorLoc;
  std::tie(ToLHS, ToRHS, ToType, ToOperatorLoc) = *Imp;

  return new (Importer.getToContext()) BinaryOperator(
      ToLHS, ToRHS, E->getOpcode(), ToType, E->getValueKind(),
      E->getObjectKind(), ToOperatorLoc, E->getFPFeatures());
}

ExpectedStmt ASTNodeImporter::VisitConditionalOperator(ConditionalOperator *E) {
  auto Imp = importSeq(
      E->getCond(), E->getQuestionLoc(), E->getLHS(), E->getColonLoc(),
      E->getRHS(), E->getType());
  if (!Imp)
    return Imp.takeError();

  Expr *ToCond, *ToLHS, *ToRHS;
  SourceLocation ToQuestionLoc, ToColonLoc;
  QualType ToType;
  std::tie(ToCond, ToQuestionLoc, ToLHS, ToColonLoc, ToRHS, ToType) = *Imp;

  return new (Importer.getToContext()) ConditionalOperator(
      ToCond, ToQuestionLoc, ToLHS, ToColonLoc, ToRHS, ToType,
      E->getValueKind(), E->getObjectKind());
}

ExpectedStmt ASTNodeImporter::VisitBinaryConditionalOperator(
    BinaryConditionalOperator *E) {
  auto Imp = importSeq(
      E->getCommon(), E->getOpaqueValue(), E->getCond(), E->getTrueExpr(),
      E->getFalseExpr(), E->getQuestionLoc(), E->getColonLoc(), E->getType());
  if (!Imp)
    return Imp.takeError();

  Expr *ToCommon, *ToCond, *ToTrueExpr, *ToFalseExpr;
  OpaqueValueExpr *ToOpaqueValue;
  SourceLocation ToQuestionLoc, ToColonLoc;
  QualType ToType;
  std::tie(
      ToCommon, ToOpaqueValue, ToCond, ToTrueExpr, ToFalseExpr, ToQuestionLoc,
      ToColonLoc, ToType) = *Imp;

  return new (Importer.getToContext()) BinaryConditionalOperator(
      ToCommon, ToOpaqueValue, ToCond, ToTrueExpr, ToFalseExpr,
      ToQuestionLoc, ToColonLoc, ToType, E->getValueKind(),
      E->getObjectKind());
}

ExpectedStmt ASTNodeImporter::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  auto Imp = importSeq(
      E->getBeginLoc(), E->getQueriedTypeSourceInfo(),
      E->getDimensionExpression(), E->getEndLoc(), E->getType());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToBeginLoc, ToEndLoc;
  TypeSourceInfo *ToQueriedTypeSourceInfo;
  Expr *ToDimensionExpression;
  QualType ToType;
  std::tie(
      ToBeginLoc, ToQueriedTypeSourceInfo, ToDimensionExpression, ToEndLoc,
      ToType) = *Imp;

  return new (Importer.getToContext()) ArrayTypeTraitExpr(
      ToBeginLoc, E->getTrait(), ToQueriedTypeSourceInfo, E->getValue(),
      ToDimensionExpression, ToEndLoc, ToType);
}

ExpectedStmt ASTNodeImporter::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  auto Imp = importSeq(
      E->getBeginLoc(), E->getQueriedExpression(), E->getEndLoc(), E->getType());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToBeginLoc, ToEndLoc;
  Expr *ToQueriedExpression;
  QualType ToType;
  std::tie(ToBeginLoc, ToQueriedExpression, ToEndLoc, ToType) = *Imp;

  return new (Importer.getToContext()) ExpressionTraitExpr(
      ToBeginLoc, E->getTrait(), ToQueriedExpression, E->getValue(),
      ToEndLoc, ToType);
}

ExpectedStmt ASTNodeImporter::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  auto Imp = importSeq(
      E->getLocation(), E->getType(), E->getSourceExpr());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToLocation;
  QualType ToType;
  Expr *ToSourceExpr;
  std::tie(ToLocation, ToType, ToSourceExpr) = *Imp;

  return new (Importer.getToContext()) OpaqueValueExpr(
      ToLocation, ToType, E->getValueKind(), E->getObjectKind(), ToSourceExpr);
}

ExpectedStmt ASTNodeImporter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  auto Imp = importSeq(
      E->getLHS(), E->getRHS(), E->getType(), E->getRBracketLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToLHS, *ToRHS;
  SourceLocation ToRBracketLoc;
  QualType ToType;
  std::tie(ToLHS, ToRHS, ToType, ToRBracketLoc) = *Imp;

  return new (Importer.getToContext()) ArraySubscriptExpr(
      ToLHS, ToRHS, ToType, E->getValueKind(), E->getObjectKind(),
      ToRBracketLoc);
}

ExpectedStmt
ASTNodeImporter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  auto Imp = importSeq(
      E->getLHS(), E->getRHS(), E->getType(), E->getComputationLHSType(),
      E->getComputationResultType(), E->getOperatorLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToLHS, *ToRHS;
  QualType ToType, ToComputationLHSType, ToComputationResultType;
  SourceLocation ToOperatorLoc;
  std::tie(ToLHS, ToRHS, ToType, ToComputationLHSType, ToComputationResultType,
      ToOperatorLoc) = *Imp;

  return new (Importer.getToContext()) CompoundAssignOperator(
      ToLHS, ToRHS, E->getOpcode(), ToType, E->getValueKind(),
      E->getObjectKind(), ToComputationLHSType, ToComputationResultType,
      ToOperatorLoc, E->getFPFeatures());
}

Expected<CXXCastPath>
ASTNodeImporter::ImportCastPath(CastExpr *CE) {
  CXXCastPath Path;
  for (auto I = CE->path_begin(), E = CE->path_end(); I != E; ++I) {
    if (auto SpecOrErr = import(*I))
      Path.push_back(*SpecOrErr);
    else
      return SpecOrErr.takeError();
  }
  return Path;
}

ExpectedStmt ASTNodeImporter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedExpr ToSubExprOrErr = import(E->getSubExpr());
  if (!ToSubExprOrErr)
    return ToSubExprOrErr.takeError();

  Expected<CXXCastPath> ToBasePathOrErr = ImportCastPath(E);
  if (!ToBasePathOrErr)
    return ToBasePathOrErr.takeError();

  return ImplicitCastExpr::Create(
      Importer.getToContext(), *ToTypeOrErr, E->getCastKind(), *ToSubExprOrErr,
      &(*ToBasePathOrErr), E->getValueKind());
}

ExpectedStmt ASTNodeImporter::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  auto Imp1 = importSeq(
      E->getType(), E->getSubExpr(), E->getTypeInfoAsWritten());
  if (!Imp1)
    return Imp1.takeError();

  QualType ToType;
  Expr *ToSubExpr;
  TypeSourceInfo *ToTypeInfoAsWritten;
  std::tie(ToType, ToSubExpr, ToTypeInfoAsWritten) = *Imp1;

  Expected<CXXCastPath> ToBasePathOrErr = ImportCastPath(E);
  if (!ToBasePathOrErr)
    return ToBasePathOrErr.takeError();
  CXXCastPath *ToBasePath = &(*ToBasePathOrErr);

  switch (E->getStmtClass()) {
  case Stmt::CStyleCastExprClass: {
    auto *CCE = cast<CStyleCastExpr>(E);
    ExpectedSLoc ToLParenLocOrErr = import(CCE->getLParenLoc());
    if (!ToLParenLocOrErr)
      return ToLParenLocOrErr.takeError();
    ExpectedSLoc ToRParenLocOrErr = import(CCE->getRParenLoc());
    if (!ToRParenLocOrErr)
      return ToRParenLocOrErr.takeError();
    return CStyleCastExpr::Create(
        Importer.getToContext(), ToType, E->getValueKind(), E->getCastKind(),
        ToSubExpr, ToBasePath, ToTypeInfoAsWritten, *ToLParenLocOrErr,
        *ToRParenLocOrErr);
  }

  case Stmt::CXXFunctionalCastExprClass: {
    auto *FCE = cast<CXXFunctionalCastExpr>(E);
    ExpectedSLoc ToLParenLocOrErr = import(FCE->getLParenLoc());
    if (!ToLParenLocOrErr)
      return ToLParenLocOrErr.takeError();
    ExpectedSLoc ToRParenLocOrErr = import(FCE->getRParenLoc());
    if (!ToRParenLocOrErr)
      return ToRParenLocOrErr.takeError();
    return CXXFunctionalCastExpr::Create(
        Importer.getToContext(), ToType, E->getValueKind(), ToTypeInfoAsWritten,
        E->getCastKind(), ToSubExpr, ToBasePath, *ToLParenLocOrErr,
        *ToRParenLocOrErr);
  }

  case Stmt::ObjCBridgedCastExprClass: {
    auto *OCE = cast<ObjCBridgedCastExpr>(E);
    ExpectedSLoc ToLParenLocOrErr = import(OCE->getLParenLoc());
    if (!ToLParenLocOrErr)
      return ToLParenLocOrErr.takeError();
    ExpectedSLoc ToBridgeKeywordLocOrErr = import(OCE->getBridgeKeywordLoc());
    if (!ToBridgeKeywordLocOrErr)
      return ToBridgeKeywordLocOrErr.takeError();
    return new (Importer.getToContext()) ObjCBridgedCastExpr(
        *ToLParenLocOrErr, OCE->getBridgeKind(), E->getCastKind(),
        *ToBridgeKeywordLocOrErr, ToTypeInfoAsWritten, ToSubExpr);
  }
  default:
    llvm_unreachable("Cast expression of unsupported type!");
    return make_error<ImportError>(ImportError::UnsupportedConstruct);
  }
}

ExpectedStmt ASTNodeImporter::VisitOffsetOfExpr(OffsetOfExpr *E) {
  SmallVector<OffsetOfNode, 4> ToNodes;
  for (int I = 0, N = E->getNumComponents(); I < N; ++I) {
    const OffsetOfNode &FromNode = E->getComponent(I);

    SourceLocation ToBeginLoc, ToEndLoc;
    if (FromNode.getKind() != OffsetOfNode::Base) {
      auto Imp = importSeq(FromNode.getBeginLoc(), FromNode.getEndLoc());
      if (!Imp)
        return Imp.takeError();
      std::tie(ToBeginLoc, ToEndLoc) = *Imp;
    }

    switch (FromNode.getKind()) {
    case OffsetOfNode::Array:
      ToNodes.push_back(
          OffsetOfNode(ToBeginLoc, FromNode.getArrayExprIndex(), ToEndLoc));
      break;
    case OffsetOfNode::Base: {
      auto ToBSOrErr = import(FromNode.getBase());
      if (!ToBSOrErr)
        return ToBSOrErr.takeError();
      ToNodes.push_back(OffsetOfNode(*ToBSOrErr));
      break;
    }
    case OffsetOfNode::Field: {
      auto ToFieldOrErr = import(FromNode.getField());
      if (!ToFieldOrErr)
        return ToFieldOrErr.takeError();
      ToNodes.push_back(OffsetOfNode(ToBeginLoc, *ToFieldOrErr, ToEndLoc));
      break;
    }
    case OffsetOfNode::Identifier: {
      IdentifierInfo *ToII = Importer.Import(FromNode.getFieldName());
      ToNodes.push_back(OffsetOfNode(ToBeginLoc, ToII, ToEndLoc));
      break;
    }
    }
  }

  SmallVector<Expr *, 4> ToExprs(E->getNumExpressions());
  for (int I = 0, N = E->getNumExpressions(); I < N; ++I) {
    ExpectedExpr ToIndexExprOrErr = import(E->getIndexExpr(I));
    if (!ToIndexExprOrErr)
      return ToIndexExprOrErr.takeError();
    ToExprs[I] = *ToIndexExprOrErr;
  }

  auto Imp = importSeq(
      E->getType(), E->getTypeSourceInfo(), E->getOperatorLoc(),
      E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  SourceLocation ToOperatorLoc, ToRParenLoc;
  std::tie(ToType, ToTypeSourceInfo, ToOperatorLoc, ToRParenLoc) = *Imp;

  return OffsetOfExpr::Create(
      Importer.getToContext(), ToType, ToOperatorLoc, ToTypeSourceInfo, ToNodes,
      ToExprs, ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getOperand(), E->getBeginLoc(), E->getEndLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  Expr *ToOperand;
  SourceLocation ToBeginLoc, ToEndLoc;
  std::tie(ToType, ToOperand, ToBeginLoc, ToEndLoc) = *Imp;

  CanThrowResult ToCanThrow;
  if (E->isValueDependent())
    ToCanThrow = CT_Dependent;
  else
    ToCanThrow = E->getValue() ? CT_Can : CT_Cannot;

  return new (Importer.getToContext()) CXXNoexceptExpr(
      ToType, ToOperand, ToCanThrow, ToBeginLoc, ToEndLoc);
}

ExpectedStmt ASTNodeImporter::VisitCXXThrowExpr(CXXThrowExpr *E) {
  auto Imp = importSeq(E->getSubExpr(), E->getType(), E->getThrowLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToSubExpr;
  QualType ToType;
  SourceLocation ToThrowLoc;
  std::tie(ToSubExpr, ToType, ToThrowLoc) = *Imp;

  return new (Importer.getToContext()) CXXThrowExpr(
      ToSubExpr, ToType, ToThrowLoc, E->isThrownVariableInScope());
}

ExpectedStmt ASTNodeImporter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  ExpectedSLoc ToUsedLocOrErr = import(E->getUsedLocation());
  if (!ToUsedLocOrErr)
    return ToUsedLocOrErr.takeError();

  auto ToParamOrErr = import(E->getParam());
  if (!ToParamOrErr)
    return ToParamOrErr.takeError();

  auto UsedContextOrErr = Importer.ImportContext(E->getUsedContext());
  if (!UsedContextOrErr)
    return UsedContextOrErr.takeError();

  // Import the default arg if it was not imported yet.
  // This is needed because it can happen that during the import of the
  // default expression (from VisitParmVarDecl) the same ParmVarDecl is
  // encountered here. The default argument for a ParmVarDecl is set in the
  // ParmVarDecl only after it is imported (set in VisitParmVarDecl if not here,
  // see VisitParmVarDecl).
  ParmVarDecl *ToParam = *ToParamOrErr;
  if (!ToParam->getDefaultArg()) {
    Optional<ParmVarDecl *> FromParam = Importer.getImportedFromDecl(ToParam);
    assert(FromParam && "ParmVarDecl was not imported?");

    if (Error Err = ImportDefaultArgOfParmVarDecl(*FromParam, ToParam))
      return std::move(Err);
  }

  return CXXDefaultArgExpr::Create(Importer.getToContext(), *ToUsedLocOrErr,
                                   *ToParamOrErr, *UsedContextOrErr);
}

ExpectedStmt
ASTNodeImporter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getTypeSourceInfo(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  SourceLocation ToRParenLoc;
  std::tie(ToType, ToTypeSourceInfo, ToRParenLoc) = *Imp;

  return new (Importer.getToContext()) CXXScalarValueInitExpr(
      ToType, ToTypeSourceInfo, ToRParenLoc);
}

ExpectedStmt
ASTNodeImporter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  ExpectedExpr ToSubExprOrErr = import(E->getSubExpr());
  if (!ToSubExprOrErr)
    return ToSubExprOrErr.takeError();

  auto ToDtorOrErr = import(E->getTemporary()->getDestructor());
  if (!ToDtorOrErr)
    return ToDtorOrErr.takeError();

  ASTContext &ToCtx = Importer.getToContext();
  CXXTemporary *Temp = CXXTemporary::Create(ToCtx, *ToDtorOrErr);
  return CXXBindTemporaryExpr::Create(ToCtx, Temp, *ToSubExprOrErr);
}

ExpectedStmt
ASTNodeImporter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  auto Imp = importSeq(
      E->getConstructor(), E->getType(), E->getTypeSourceInfo(),
      E->getParenOrBraceRange());
  if (!Imp)
    return Imp.takeError();

  CXXConstructorDecl *ToConstructor;
  QualType ToType;
  TypeSourceInfo *ToTypeSourceInfo;
  SourceRange ToParenOrBraceRange;
  std::tie(ToConstructor, ToType, ToTypeSourceInfo, ToParenOrBraceRange) = *Imp;

  SmallVector<Expr *, 8> ToArgs(E->getNumArgs());
  if (Error Err = ImportContainerChecked(E->arguments(), ToArgs))
    return std::move(Err);

  return CXXTemporaryObjectExpr::Create(
      Importer.getToContext(), ToConstructor, ToType, ToTypeSourceInfo, ToArgs,
      ToParenOrBraceRange, E->hadMultipleCandidates(),
      E->isListInitialization(), E->isStdInitListInitialization(),
      E->requiresZeroInitialization());
}

ExpectedDecl ASTNodeImporter::VisitLifetimeExtendedTemporaryDecl(
    LifetimeExtendedTemporaryDecl *D) {
  DeclContext *DC, *LexicalDC;
  if (Error Err = ImportDeclContext(D, DC, LexicalDC))
    return std::move(Err);

  auto Imp = importSeq(D->getTemporaryExpr(), D->getExtendingDecl());
  // FIXME: the APValue should be imported as well if present.
  if (!Imp)
    return Imp.takeError();

  Expr *Temporary;
  ValueDecl *ExtendingDecl;
  std::tie(Temporary, ExtendingDecl) = *Imp;
  // FIXME: Should ManglingNumber get numbers associated with 'to' context?

  LifetimeExtendedTemporaryDecl *To;
  if (GetImportedOrCreateDecl(To, D, Temporary, ExtendingDecl,
                              D->getManglingNumber()))
    return To;

  To->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(To);
  return To;
}

ExpectedStmt
ASTNodeImporter::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
  auto Imp = importSeq(E->getType(),
                       E->getLifetimeExtendedTemporaryDecl() ? nullptr
                                                             : E->getSubExpr(),
                       E->getLifetimeExtendedTemporaryDecl());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  Expr *ToTemporaryExpr;
  LifetimeExtendedTemporaryDecl *ToMaterializedDecl;
  std::tie(ToType, ToTemporaryExpr, ToMaterializedDecl) = *Imp;
  if (!ToTemporaryExpr)
    ToTemporaryExpr = cast<Expr>(ToMaterializedDecl->getTemporaryExpr());

  auto *ToMTE = new (Importer.getToContext()) MaterializeTemporaryExpr(
      ToType, ToTemporaryExpr, E->isBoundToLvalueReference(),
      ToMaterializedDecl);

  return ToMTE;
}

ExpectedStmt ASTNodeImporter::VisitPackExpansionExpr(PackExpansionExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getPattern(), E->getEllipsisLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  Expr *ToPattern;
  SourceLocation ToEllipsisLoc;
  std::tie(ToType, ToPattern, ToEllipsisLoc) = *Imp;

  return new (Importer.getToContext()) PackExpansionExpr(
      ToType, ToPattern, ToEllipsisLoc, E->getNumExpansions());
}

ExpectedStmt ASTNodeImporter::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  auto Imp = importSeq(
      E->getOperatorLoc(), E->getPack(), E->getPackLoc(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToOperatorLoc, ToPackLoc, ToRParenLoc;
  NamedDecl *ToPack;
  std::tie(ToOperatorLoc, ToPack, ToPackLoc, ToRParenLoc) = *Imp;

  Optional<unsigned> Length;
  if (!E->isValueDependent())
    Length = E->getPackLength();

  SmallVector<TemplateArgument, 8> ToPartialArguments;
  if (E->isPartiallySubstituted()) {
    if (Error Err = ImportTemplateArguments(
        E->getPartialArguments().data(),
        E->getPartialArguments().size(),
        ToPartialArguments))
      return std::move(Err);
  }

  return SizeOfPackExpr::Create(
      Importer.getToContext(), ToOperatorLoc, ToPack, ToPackLoc, ToRParenLoc,
      Length, ToPartialArguments);
}


ExpectedStmt ASTNodeImporter::VisitCXXNewExpr(CXXNewExpr *E) {
  auto Imp = importSeq(
      E->getOperatorNew(), E->getOperatorDelete(), E->getTypeIdParens(),
      E->getArraySize(), E->getInitializer(), E->getType(),
      E->getAllocatedTypeSourceInfo(), E->getSourceRange(),
      E->getDirectInitRange());
  if (!Imp)
    return Imp.takeError();

  FunctionDecl *ToOperatorNew, *ToOperatorDelete;
  SourceRange ToTypeIdParens, ToSourceRange, ToDirectInitRange;
  Optional<Expr *> ToArraySize;
  Expr *ToInitializer;
  QualType ToType;
  TypeSourceInfo *ToAllocatedTypeSourceInfo;
  std::tie(
    ToOperatorNew, ToOperatorDelete, ToTypeIdParens, ToArraySize, ToInitializer,
    ToType, ToAllocatedTypeSourceInfo, ToSourceRange, ToDirectInitRange) = *Imp;

  SmallVector<Expr *, 4> ToPlacementArgs(E->getNumPlacementArgs());
  if (Error Err =
      ImportContainerChecked(E->placement_arguments(), ToPlacementArgs))
    return std::move(Err);

  return CXXNewExpr::Create(
      Importer.getToContext(), E->isGlobalNew(), ToOperatorNew,
      ToOperatorDelete, E->passAlignment(), E->doesUsualArrayDeleteWantSize(),
      ToPlacementArgs, ToTypeIdParens, ToArraySize, E->getInitializationStyle(),
      ToInitializer, ToType, ToAllocatedTypeSourceInfo, ToSourceRange,
      ToDirectInitRange);
}

ExpectedStmt ASTNodeImporter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getOperatorDelete(), E->getArgument(), E->getBeginLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  FunctionDecl *ToOperatorDelete;
  Expr *ToArgument;
  SourceLocation ToBeginLoc;
  std::tie(ToType, ToOperatorDelete, ToArgument, ToBeginLoc) = *Imp;

  return new (Importer.getToContext()) CXXDeleteExpr(
      ToType, E->isGlobalDelete(), E->isArrayForm(), E->isArrayFormAsWritten(),
      E->doesUsualArrayDeleteWantSize(), ToOperatorDelete, ToArgument,
      ToBeginLoc);
}

ExpectedStmt ASTNodeImporter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getLocation(), E->getConstructor(),
      E->getParenOrBraceRange());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  SourceLocation ToLocation;
  CXXConstructorDecl *ToConstructor;
  SourceRange ToParenOrBraceRange;
  std::tie(ToType, ToLocation, ToConstructor, ToParenOrBraceRange) = *Imp;

  SmallVector<Expr *, 6> ToArgs(E->getNumArgs());
  if (Error Err = ImportContainerChecked(E->arguments(), ToArgs))
    return std::move(Err);

  return CXXConstructExpr::Create(
      Importer.getToContext(), ToType, ToLocation, ToConstructor,
      E->isElidable(), ToArgs, E->hadMultipleCandidates(),
      E->isListInitialization(), E->isStdInitListInitialization(),
      E->requiresZeroInitialization(), E->getConstructionKind(),
      ToParenOrBraceRange);
}

ExpectedStmt ASTNodeImporter::VisitExprWithCleanups(ExprWithCleanups *E) {
  ExpectedExpr ToSubExprOrErr = import(E->getSubExpr());
  if (!ToSubExprOrErr)
    return ToSubExprOrErr.takeError();

  SmallVector<ExprWithCleanups::CleanupObject, 8> ToObjects(E->getNumObjects());
  if (Error Err = ImportContainerChecked(E->getObjects(), ToObjects))
    return std::move(Err);

  return ExprWithCleanups::Create(
      Importer.getToContext(), *ToSubExprOrErr, E->cleanupsHaveSideEffects(),
      ToObjects);
}

ExpectedStmt ASTNodeImporter::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  auto Imp = importSeq(
      E->getCallee(), E->getType(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToCallee;
  QualType ToType;
  SourceLocation ToRParenLoc;
  std::tie(ToCallee, ToType, ToRParenLoc) = *Imp;

  SmallVector<Expr *, 4> ToArgs(E->getNumArgs());
  if (Error Err = ImportContainerChecked(E->arguments(), ToArgs))
    return std::move(Err);

  return CXXMemberCallExpr::Create(Importer.getToContext(), ToCallee, ToArgs,
                                   ToType, E->getValueKind(), ToRParenLoc);
}

ExpectedStmt ASTNodeImporter::VisitCXXThisExpr(CXXThisExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedSLoc ToLocationOrErr = import(E->getLocation());
  if (!ToLocationOrErr)
    return ToLocationOrErr.takeError();

  return new (Importer.getToContext()) CXXThisExpr(
      *ToLocationOrErr, *ToTypeOrErr, E->isImplicit());
}

ExpectedStmt ASTNodeImporter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedSLoc ToLocationOrErr = import(E->getLocation());
  if (!ToLocationOrErr)
    return ToLocationOrErr.takeError();

  return new (Importer.getToContext()) CXXBoolLiteralExpr(
      E->getValue(), *ToTypeOrErr, *ToLocationOrErr);
}

ExpectedStmt ASTNodeImporter::VisitMemberExpr(MemberExpr *E) {
  auto Imp1 = importSeq(
      E->getBase(), E->getOperatorLoc(), E->getQualifierLoc(),
      E->getTemplateKeywordLoc(), E->getMemberDecl(), E->getType());
  if (!Imp1)
    return Imp1.takeError();

  Expr *ToBase;
  SourceLocation ToOperatorLoc, ToTemplateKeywordLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  ValueDecl *ToMemberDecl;
  QualType ToType;
  std::tie(
      ToBase, ToOperatorLoc, ToQualifierLoc, ToTemplateKeywordLoc, ToMemberDecl,
      ToType) = *Imp1;

  auto Imp2 = importSeq(
      E->getFoundDecl().getDecl(), E->getMemberNameInfo().getName(),
      E->getMemberNameInfo().getLoc(), E->getLAngleLoc(), E->getRAngleLoc());
  if (!Imp2)
    return Imp2.takeError();
  NamedDecl *ToDecl;
  DeclarationName ToName;
  SourceLocation ToLoc, ToLAngleLoc, ToRAngleLoc;
  std::tie(ToDecl, ToName, ToLoc, ToLAngleLoc, ToRAngleLoc) = *Imp2;

  DeclAccessPair ToFoundDecl =
      DeclAccessPair::make(ToDecl, E->getFoundDecl().getAccess());

  DeclarationNameInfo ToMemberNameInfo(ToName, ToLoc);

  TemplateArgumentListInfo ToTAInfo, *ResInfo = nullptr;
  if (E->hasExplicitTemplateArgs()) {
    if (Error Err =
            ImportTemplateArgumentListInfo(E->getLAngleLoc(), E->getRAngleLoc(),
                                           E->template_arguments(), ToTAInfo))
      return std::move(Err);
    ResInfo = &ToTAInfo;
  }

  return MemberExpr::Create(Importer.getToContext(), ToBase, E->isArrow(),
                            ToOperatorLoc, ToQualifierLoc, ToTemplateKeywordLoc,
                            ToMemberDecl, ToFoundDecl, ToMemberNameInfo,
                            ResInfo, ToType, E->getValueKind(),
                            E->getObjectKind(), E->isNonOdrUse());
}

ExpectedStmt
ASTNodeImporter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  auto Imp = importSeq(
      E->getBase(), E->getOperatorLoc(), E->getQualifierLoc(),
      E->getScopeTypeInfo(), E->getColonColonLoc(), E->getTildeLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToBase;
  SourceLocation ToOperatorLoc, ToColonColonLoc, ToTildeLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  TypeSourceInfo *ToScopeTypeInfo;
  std::tie(
      ToBase, ToOperatorLoc, ToQualifierLoc, ToScopeTypeInfo, ToColonColonLoc,
      ToTildeLoc) = *Imp;

  PseudoDestructorTypeStorage Storage;
  if (IdentifierInfo *FromII = E->getDestroyedTypeIdentifier()) {
    IdentifierInfo *ToII = Importer.Import(FromII);
    ExpectedSLoc ToDestroyedTypeLocOrErr = import(E->getDestroyedTypeLoc());
    if (!ToDestroyedTypeLocOrErr)
      return ToDestroyedTypeLocOrErr.takeError();
    Storage = PseudoDestructorTypeStorage(ToII, *ToDestroyedTypeLocOrErr);
  } else {
    if (auto ToTIOrErr = import(E->getDestroyedTypeInfo()))
      Storage = PseudoDestructorTypeStorage(*ToTIOrErr);
    else
      return ToTIOrErr.takeError();
  }

  return new (Importer.getToContext()) CXXPseudoDestructorExpr(
      Importer.getToContext(), ToBase, E->isArrow(), ToOperatorLoc,
      ToQualifierLoc, ToScopeTypeInfo, ToColonColonLoc, ToTildeLoc, Storage);
}

ExpectedStmt ASTNodeImporter::VisitCXXDependentScopeMemberExpr(
    CXXDependentScopeMemberExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getOperatorLoc(), E->getQualifierLoc(),
      E->getTemplateKeywordLoc(), E->getFirstQualifierFoundInScope());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  SourceLocation ToOperatorLoc, ToTemplateKeywordLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  NamedDecl *ToFirstQualifierFoundInScope;
  std::tie(
      ToType, ToOperatorLoc, ToQualifierLoc, ToTemplateKeywordLoc,
      ToFirstQualifierFoundInScope) = *Imp;

  Expr *ToBase = nullptr;
  if (!E->isImplicitAccess()) {
    if (ExpectedExpr ToBaseOrErr = import(E->getBase()))
      ToBase = *ToBaseOrErr;
    else
      return ToBaseOrErr.takeError();
  }

  TemplateArgumentListInfo ToTAInfo, *ResInfo = nullptr;
  if (E->hasExplicitTemplateArgs()) {
    if (Error Err = ImportTemplateArgumentListInfo(
        E->getLAngleLoc(), E->getRAngleLoc(), E->template_arguments(),
        ToTAInfo))
      return std::move(Err);
    ResInfo = &ToTAInfo;
  }

  auto ToMemberNameInfoOrErr = importSeq(E->getMember(), E->getMemberLoc());
  if (!ToMemberNameInfoOrErr)
    return ToMemberNameInfoOrErr.takeError();
  DeclarationNameInfo ToMemberNameInfo(
      std::get<0>(*ToMemberNameInfoOrErr), std::get<1>(*ToMemberNameInfoOrErr));
  // Import additional name location/type info.
  if (Error Err = ImportDeclarationNameLoc(
      E->getMemberNameInfo(), ToMemberNameInfo))
    return std::move(Err);

  return CXXDependentScopeMemberExpr::Create(
      Importer.getToContext(), ToBase, ToType, E->isArrow(), ToOperatorLoc,
      ToQualifierLoc, ToTemplateKeywordLoc, ToFirstQualifierFoundInScope,
      ToMemberNameInfo, ResInfo);
}

ExpectedStmt
ASTNodeImporter::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  auto Imp = importSeq(E->getQualifierLoc(), E->getTemplateKeywordLoc(),
                       E->getDeclName(), E->getNameInfo().getLoc(),
                       E->getLAngleLoc(), E->getRAngleLoc());
  if (!Imp)
    return Imp.takeError();

  NestedNameSpecifierLoc ToQualifierLoc;
  SourceLocation ToTemplateKeywordLoc, ToNameLoc, ToLAngleLoc, ToRAngleLoc;
  DeclarationName ToDeclName;
  std::tie(ToQualifierLoc, ToTemplateKeywordLoc, ToDeclName, ToNameLoc,
           ToLAngleLoc, ToRAngleLoc) = *Imp;

  DeclarationNameInfo ToNameInfo(ToDeclName, ToNameLoc);
  if (Error Err = ImportDeclarationNameLoc(E->getNameInfo(), ToNameInfo))
    return std::move(Err);

  TemplateArgumentListInfo ToTAInfo(ToLAngleLoc, ToRAngleLoc);
  TemplateArgumentListInfo *ResInfo = nullptr;
  if (E->hasExplicitTemplateArgs()) {
    if (Error Err =
        ImportTemplateArgumentListInfo(E->template_arguments(), ToTAInfo))
      return std::move(Err);
    ResInfo = &ToTAInfo;
  }

  return DependentScopeDeclRefExpr::Create(
      Importer.getToContext(), ToQualifierLoc, ToTemplateKeywordLoc,
      ToNameInfo, ResInfo);
}

ExpectedStmt ASTNodeImporter::VisitCXXUnresolvedConstructExpr(
    CXXUnresolvedConstructExpr *E) {
  auto Imp = importSeq(
      E->getLParenLoc(), E->getRParenLoc(), E->getTypeSourceInfo());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToLParenLoc, ToRParenLoc;
  TypeSourceInfo *ToTypeSourceInfo;
  std::tie(ToLParenLoc, ToRParenLoc, ToTypeSourceInfo) = *Imp;

  SmallVector<Expr *, 8> ToArgs(E->arg_size());
  if (Error Err =
      ImportArrayChecked(E->arg_begin(), E->arg_end(), ToArgs.begin()))
    return std::move(Err);

  return CXXUnresolvedConstructExpr::Create(
      Importer.getToContext(), ToTypeSourceInfo, ToLParenLoc,
      llvm::makeArrayRef(ToArgs), ToRParenLoc);
}

ExpectedStmt
ASTNodeImporter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *E) {
  Expected<CXXRecordDecl *> ToNamingClassOrErr = import(E->getNamingClass());
  if (!ToNamingClassOrErr)
    return ToNamingClassOrErr.takeError();

  auto ToQualifierLocOrErr = import(E->getQualifierLoc());
  if (!ToQualifierLocOrErr)
    return ToQualifierLocOrErr.takeError();

  auto ToNameInfoOrErr = importSeq(E->getName(), E->getNameLoc());
  if (!ToNameInfoOrErr)
    return ToNameInfoOrErr.takeError();
  DeclarationNameInfo ToNameInfo(
      std::get<0>(*ToNameInfoOrErr), std::get<1>(*ToNameInfoOrErr));
  // Import additional name location/type info.
  if (Error Err = ImportDeclarationNameLoc(E->getNameInfo(), ToNameInfo))
    return std::move(Err);

  UnresolvedSet<8> ToDecls;
  for (auto *D : E->decls())
    if (auto ToDOrErr = import(D))
      ToDecls.addDecl(cast<NamedDecl>(*ToDOrErr));
    else
      return ToDOrErr.takeError();

  if (E->hasExplicitTemplateArgs()) {
    TemplateArgumentListInfo ToTAInfo;
    if (Error Err = ImportTemplateArgumentListInfo(
        E->getLAngleLoc(), E->getRAngleLoc(), E->template_arguments(),
        ToTAInfo))
      return std::move(Err);

    ExpectedSLoc ToTemplateKeywordLocOrErr = import(E->getTemplateKeywordLoc());
    if (!ToTemplateKeywordLocOrErr)
      return ToTemplateKeywordLocOrErr.takeError();

    return UnresolvedLookupExpr::Create(
        Importer.getToContext(), *ToNamingClassOrErr, *ToQualifierLocOrErr,
        *ToTemplateKeywordLocOrErr, ToNameInfo, E->requiresADL(), &ToTAInfo,
        ToDecls.begin(), ToDecls.end());
  }

  return UnresolvedLookupExpr::Create(
      Importer.getToContext(), *ToNamingClassOrErr, *ToQualifierLocOrErr,
      ToNameInfo, E->requiresADL(), E->isOverloaded(), ToDecls.begin(),
      ToDecls.end());
}

ExpectedStmt
ASTNodeImporter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *E) {
  auto Imp1 = importSeq(
      E->getType(), E->getOperatorLoc(), E->getQualifierLoc(),
      E->getTemplateKeywordLoc());
  if (!Imp1)
    return Imp1.takeError();

  QualType ToType;
  SourceLocation ToOperatorLoc, ToTemplateKeywordLoc;
  NestedNameSpecifierLoc ToQualifierLoc;
  std::tie(ToType, ToOperatorLoc, ToQualifierLoc, ToTemplateKeywordLoc) = *Imp1;

  auto Imp2 = importSeq(E->getName(), E->getNameLoc());
  if (!Imp2)
    return Imp2.takeError();
  DeclarationNameInfo ToNameInfo(std::get<0>(*Imp2), std::get<1>(*Imp2));
  // Import additional name location/type info.
  if (Error Err = ImportDeclarationNameLoc(E->getNameInfo(), ToNameInfo))
    return std::move(Err);

  UnresolvedSet<8> ToDecls;
  for (Decl *D : E->decls())
    if (auto ToDOrErr = import(D))
      ToDecls.addDecl(cast<NamedDecl>(*ToDOrErr));
    else
      return ToDOrErr.takeError();

  TemplateArgumentListInfo ToTAInfo;
  TemplateArgumentListInfo *ResInfo = nullptr;
  if (E->hasExplicitTemplateArgs()) {
    TemplateArgumentListInfo FromTAInfo;
    E->copyTemplateArgumentsInto(FromTAInfo);
    if (Error Err = ImportTemplateArgumentListInfo(FromTAInfo, ToTAInfo))
      return std::move(Err);
    ResInfo = &ToTAInfo;
  }

  Expr *ToBase = nullptr;
  if (!E->isImplicitAccess()) {
    if (ExpectedExpr ToBaseOrErr = import(E->getBase()))
      ToBase = *ToBaseOrErr;
    else
      return ToBaseOrErr.takeError();
  }

  return UnresolvedMemberExpr::Create(
      Importer.getToContext(), E->hasUnresolvedUsing(), ToBase, ToType,
      E->isArrow(), ToOperatorLoc, ToQualifierLoc, ToTemplateKeywordLoc,
      ToNameInfo, ResInfo, ToDecls.begin(), ToDecls.end());
}

ExpectedStmt ASTNodeImporter::VisitCallExpr(CallExpr *E) {
  auto Imp = importSeq(E->getCallee(), E->getType(), E->getRParenLoc());
  if (!Imp)
    return Imp.takeError();

  Expr *ToCallee;
  QualType ToType;
  SourceLocation ToRParenLoc;
  std::tie(ToCallee, ToType, ToRParenLoc) = *Imp;

  unsigned NumArgs = E->getNumArgs();
  llvm::SmallVector<Expr *, 2> ToArgs(NumArgs);
  if (Error Err = ImportContainerChecked(E->arguments(), ToArgs))
     return std::move(Err);

  if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    return CXXOperatorCallExpr::Create(
        Importer.getToContext(), OCE->getOperator(), ToCallee, ToArgs, ToType,
        OCE->getValueKind(), ToRParenLoc, OCE->getFPFeatures(),
        OCE->getADLCallKind());
  }

  return CallExpr::Create(Importer.getToContext(), ToCallee, ToArgs, ToType,
                          E->getValueKind(), ToRParenLoc, /*MinNumArgs=*/0,
                          E->getADLCallKind());
}

ExpectedStmt ASTNodeImporter::VisitLambdaExpr(LambdaExpr *E) {
  CXXRecordDecl *FromClass = E->getLambdaClass();
  auto ToClassOrErr = import(FromClass);
  if (!ToClassOrErr)
    return ToClassOrErr.takeError();
  CXXRecordDecl *ToClass = *ToClassOrErr;

  auto ToCallOpOrErr = import(E->getCallOperator());
  if (!ToCallOpOrErr)
    return ToCallOpOrErr.takeError();

  SmallVector<LambdaCapture, 8> ToCaptures;
  ToCaptures.reserve(E->capture_size());
  for (const auto &FromCapture : E->captures()) {
    if (auto ToCaptureOrErr = import(FromCapture))
      ToCaptures.push_back(*ToCaptureOrErr);
    else
      return ToCaptureOrErr.takeError();
  }

  SmallVector<Expr *, 8> ToCaptureInits(E->capture_size());
  if (Error Err = ImportContainerChecked(E->capture_inits(), ToCaptureInits))
    return std::move(Err);

  auto Imp = importSeq(
      E->getIntroducerRange(), E->getCaptureDefaultLoc(), E->getEndLoc());
  if (!Imp)
    return Imp.takeError();

  SourceRange ToIntroducerRange;
  SourceLocation ToCaptureDefaultLoc, ToEndLoc;
  std::tie(ToIntroducerRange, ToCaptureDefaultLoc, ToEndLoc) = *Imp;

  return LambdaExpr::Create(
      Importer.getToContext(), ToClass, ToIntroducerRange,
      E->getCaptureDefault(), ToCaptureDefaultLoc, ToCaptures,
      E->hasExplicitParameters(), E->hasExplicitResultType(), ToCaptureInits,
      ToEndLoc, E->containsUnexpandedParameterPack());
}


ExpectedStmt ASTNodeImporter::VisitInitListExpr(InitListExpr *E) {
  auto Imp = importSeq(E->getLBraceLoc(), E->getRBraceLoc(), E->getType());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToLBraceLoc, ToRBraceLoc;
  QualType ToType;
  std::tie(ToLBraceLoc, ToRBraceLoc, ToType) = *Imp;

  SmallVector<Expr *, 4> ToExprs(E->getNumInits());
  if (Error Err = ImportContainerChecked(E->inits(), ToExprs))
    return std::move(Err);

  ASTContext &ToCtx = Importer.getToContext();
  InitListExpr *To = new (ToCtx) InitListExpr(
      ToCtx, ToLBraceLoc, ToExprs, ToRBraceLoc);
  To->setType(ToType);

  if (E->hasArrayFiller()) {
    if (ExpectedExpr ToFillerOrErr = import(E->getArrayFiller()))
      To->setArrayFiller(*ToFillerOrErr);
    else
      return ToFillerOrErr.takeError();
  }

  if (FieldDecl *FromFD = E->getInitializedFieldInUnion()) {
    if (auto ToFDOrErr = import(FromFD))
      To->setInitializedFieldInUnion(*ToFDOrErr);
    else
      return ToFDOrErr.takeError();
  }

  if (InitListExpr *SyntForm = E->getSyntacticForm()) {
    if (auto ToSyntFormOrErr = import(SyntForm))
      To->setSyntacticForm(*ToSyntFormOrErr);
    else
      return ToSyntFormOrErr.takeError();
  }

  // Copy InitListExprBitfields, which are not handled in the ctor of
  // InitListExpr.
  To->sawArrayRangeDesignator(E->hadArrayRangeDesignator());

  return To;
}

ExpectedStmt ASTNodeImporter::VisitCXXStdInitializerListExpr(
    CXXStdInitializerListExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  ExpectedExpr ToSubExprOrErr = import(E->getSubExpr());
  if (!ToSubExprOrErr)
    return ToSubExprOrErr.takeError();

  return new (Importer.getToContext()) CXXStdInitializerListExpr(
      *ToTypeOrErr, *ToSubExprOrErr);
}

ExpectedStmt ASTNodeImporter::VisitCXXInheritedCtorInitExpr(
    CXXInheritedCtorInitExpr *E) {
  auto Imp = importSeq(E->getLocation(), E->getType(), E->getConstructor());
  if (!Imp)
    return Imp.takeError();

  SourceLocation ToLocation;
  QualType ToType;
  CXXConstructorDecl *ToConstructor;
  std::tie(ToLocation, ToType, ToConstructor) = *Imp;

  return new (Importer.getToContext()) CXXInheritedCtorInitExpr(
      ToLocation, ToType, ToConstructor, E->constructsVBase(),
      E->inheritedFromVBase());
}

ExpectedStmt ASTNodeImporter::VisitArrayInitLoopExpr(ArrayInitLoopExpr *E) {
  auto Imp = importSeq(E->getType(), E->getCommonExpr(), E->getSubExpr());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  Expr *ToCommonExpr, *ToSubExpr;
  std::tie(ToType, ToCommonExpr, ToSubExpr) = *Imp;

  return new (Importer.getToContext()) ArrayInitLoopExpr(
      ToType, ToCommonExpr, ToSubExpr);
}

ExpectedStmt ASTNodeImporter::VisitArrayInitIndexExpr(ArrayInitIndexExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();
  return new (Importer.getToContext()) ArrayInitIndexExpr(*ToTypeOrErr);
}

ExpectedStmt ASTNodeImporter::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E) {
  ExpectedSLoc ToBeginLocOrErr = import(E->getBeginLoc());
  if (!ToBeginLocOrErr)
    return ToBeginLocOrErr.takeError();

  auto ToFieldOrErr = import(E->getField());
  if (!ToFieldOrErr)
    return ToFieldOrErr.takeError();

  auto UsedContextOrErr = Importer.ImportContext(E->getUsedContext());
  if (!UsedContextOrErr)
    return UsedContextOrErr.takeError();

  return CXXDefaultInitExpr::Create(
      Importer.getToContext(), *ToBeginLocOrErr, *ToFieldOrErr, *UsedContextOrErr);
}

ExpectedStmt ASTNodeImporter::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getSubExpr(), E->getTypeInfoAsWritten(),
      E->getOperatorLoc(), E->getRParenLoc(), E->getAngleBrackets());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  Expr *ToSubExpr;
  TypeSourceInfo *ToTypeInfoAsWritten;
  SourceLocation ToOperatorLoc, ToRParenLoc;
  SourceRange ToAngleBrackets;
  std::tie(
      ToType, ToSubExpr, ToTypeInfoAsWritten, ToOperatorLoc, ToRParenLoc,
      ToAngleBrackets) = *Imp;

  ExprValueKind VK = E->getValueKind();
  CastKind CK = E->getCastKind();
  auto ToBasePathOrErr = ImportCastPath(E);
  if (!ToBasePathOrErr)
    return ToBasePathOrErr.takeError();

  if (isa<CXXStaticCastExpr>(E)) {
    return CXXStaticCastExpr::Create(
        Importer.getToContext(), ToType, VK, CK, ToSubExpr, &(*ToBasePathOrErr),
        ToTypeInfoAsWritten, ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else if (isa<CXXDynamicCastExpr>(E)) {
    return CXXDynamicCastExpr::Create(
        Importer.getToContext(), ToType, VK, CK, ToSubExpr, &(*ToBasePathOrErr),
        ToTypeInfoAsWritten, ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else if (isa<CXXReinterpretCastExpr>(E)) {
    return CXXReinterpretCastExpr::Create(
        Importer.getToContext(), ToType, VK, CK, ToSubExpr, &(*ToBasePathOrErr),
        ToTypeInfoAsWritten, ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else if (isa<CXXConstCastExpr>(E)) {
    return CXXConstCastExpr::Create(
        Importer.getToContext(), ToType, VK, ToSubExpr, ToTypeInfoAsWritten,
        ToOperatorLoc, ToRParenLoc, ToAngleBrackets);
  } else {
    llvm_unreachable("Unknown cast type");
    return make_error<ImportError>();
  }
}

ExpectedStmt ASTNodeImporter::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getExprLoc(), E->getParameter(), E->getReplacement());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  SourceLocation ToExprLoc;
  NonTypeTemplateParmDecl *ToParameter;
  Expr *ToReplacement;
  std::tie(ToType, ToExprLoc, ToParameter, ToReplacement) = *Imp;

  return new (Importer.getToContext()) SubstNonTypeTemplateParmExpr(
      ToType, E->getValueKind(), ToExprLoc, ToParameter, ToReplacement);
}

ExpectedStmt ASTNodeImporter::VisitTypeTraitExpr(TypeTraitExpr *E) {
  auto Imp = importSeq(
      E->getType(), E->getBeginLoc(), E->getEndLoc());
  if (!Imp)
    return Imp.takeError();

  QualType ToType;
  SourceLocation ToBeginLoc, ToEndLoc;
  std::tie(ToType, ToBeginLoc, ToEndLoc) = *Imp;

  SmallVector<TypeSourceInfo *, 4> ToArgs(E->getNumArgs());
  if (Error Err = ImportContainerChecked(E->getArgs(), ToArgs))
    return std::move(Err);

  // According to Sema::BuildTypeTrait(), if E is value-dependent,
  // Value is always false.
  bool ToValue = (E->isValueDependent() ? false : E->getValue());

  return TypeTraitExpr::Create(
      Importer.getToContext(), ToType, ToBeginLoc, E->getTrait(), ToArgs,
      ToEndLoc, ToValue);
}

ExpectedStmt ASTNodeImporter::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  ExpectedType ToTypeOrErr = import(E->getType());
  if (!ToTypeOrErr)
    return ToTypeOrErr.takeError();

  auto ToSourceRangeOrErr = import(E->getSourceRange());
  if (!ToSourceRangeOrErr)
    return ToSourceRangeOrErr.takeError();

  if (E->isTypeOperand()) {
    if (auto ToTSIOrErr = import(E->getTypeOperandSourceInfo()))
      return new (Importer.getToContext()) CXXTypeidExpr(
          *ToTypeOrErr, *ToTSIOrErr, *ToSourceRangeOrErr);
    else
      return ToTSIOrErr.takeError();
  }

  ExpectedExpr ToExprOperandOrErr = import(E->getExprOperand());
  if (!ToExprOperandOrErr)
    return ToExprOperandOrErr.takeError();

  return new (Importer.getToContext()) CXXTypeidExpr(
      *ToTypeOrErr, *ToExprOperandOrErr, *ToSourceRangeOrErr);
}

Error ASTNodeImporter::ImportOverriddenMethods(CXXMethodDecl *ToMethod,
                                               CXXMethodDecl *FromMethod) {
  Error ImportErrors = Error::success();
  for (auto *FromOverriddenMethod : FromMethod->overridden_methods()) {
    if (auto ImportedOrErr = import(FromOverriddenMethod))
      ToMethod->getCanonicalDecl()->addOverriddenMethod(cast<CXXMethodDecl>(
          (*ImportedOrErr)->getCanonicalDecl()));
    else
      ImportErrors =
          joinErrors(std::move(ImportErrors), ImportedOrErr.takeError());
  }
  return ImportErrors;
}

ASTImporter::ASTImporter(ASTContext &ToContext, FileManager &ToFileManager,
                         ASTContext &FromContext, FileManager &FromFileManager,
                         bool MinimalImport,
                         std::shared_ptr<ASTImporterSharedState> SharedState)
    : SharedState(SharedState), ToContext(ToContext), FromContext(FromContext),
      ToFileManager(ToFileManager), FromFileManager(FromFileManager),
      Minimal(MinimalImport), ODRHandling(ODRHandlingType::Conservative) {

  // Create a default state without the lookup table: LLDB case.
  if (!SharedState) {
    this->SharedState = std::make_shared<ASTImporterSharedState>();
  }

  ImportedDecls[FromContext.getTranslationUnitDecl()] =
      ToContext.getTranslationUnitDecl();
}

ASTImporter::~ASTImporter() = default;

Optional<unsigned> ASTImporter::getFieldIndex(Decl *F) {
  assert(F && (isa<FieldDecl>(*F) || isa<IndirectFieldDecl>(*F)) &&
      "Try to get field index for non-field.");

  auto *Owner = dyn_cast<RecordDecl>(F->getDeclContext());
  if (!Owner)
    return None;

  unsigned Index = 0;
  for (const auto *D : Owner->decls()) {
    if (D == F)
      return Index;

    if (isa<FieldDecl>(*D) || isa<IndirectFieldDecl>(*D))
      ++Index;
  }

  llvm_unreachable("Field was not found in its parent context.");

  return None;
}

ASTImporter::FoundDeclsTy
ASTImporter::findDeclsInToCtx(DeclContext *DC, DeclarationName Name) {
  // We search in the redecl context because of transparent contexts.
  // E.g. a simple C language enum is a transparent context:
  //   enum E { A, B };
  // Now if we had a global variable in the TU
  //   int A;
  // then the enum constant 'A' and the variable 'A' violates ODR.
  // We can diagnose this only if we search in the redecl context.
  DeclContext *ReDC = DC->getRedeclContext();
  if (SharedState->getLookupTable()) {
    ASTImporterLookupTable::LookupResult LookupResult =
        SharedState->getLookupTable()->lookup(ReDC, Name);
    return FoundDeclsTy(LookupResult.begin(), LookupResult.end());
  } else {
    DeclContext::lookup_result NoloadLookupResult = ReDC->noload_lookup(Name);
    FoundDeclsTy Result(NoloadLookupResult.begin(), NoloadLookupResult.end());
    // We must search by the slow case of localUncachedLookup because that is
    // working even if there is no LookupPtr for the DC. We could use
    // DC::buildLookup() to create the LookupPtr, but that would load external
    // decls again, we must avoid that case.
    // Also, even if we had the LookupPtr, we must find Decls which are not
    // in the LookupPtr, so we need the slow case.
    // These cases are handled in ASTImporterLookupTable, but we cannot use
    // that with LLDB since that traverses through the AST which initiates the
    // load of external decls again via DC::decls().  And again, we must avoid
    // loading external decls during the import.
    if (Result.empty())
      ReDC->localUncachedLookup(Name, Result);
    return Result;
  }
}

void ASTImporter::AddToLookupTable(Decl *ToD) {
  SharedState->addDeclToLookup(ToD);
}

Expected<Decl *> ASTImporter::ImportImpl(Decl *FromD) {
  // Import the decl using ASTNodeImporter.
  ASTNodeImporter Importer(*this);
  return Importer.Visit(FromD);
}

void ASTImporter::RegisterImportedDecl(Decl *FromD, Decl *ToD) {
  MapImported(FromD, ToD);
}

Expected<QualType> ASTImporter::Import(QualType FromT) {
  if (FromT.isNull())
    return QualType{};

  const Type *FromTy = FromT.getTypePtr();

  // Check whether we've already imported this type.
  llvm::DenseMap<const Type *, const Type *>::iterator Pos
    = ImportedTypes.find(FromTy);
  if (Pos != ImportedTypes.end())
    return ToContext.getQualifiedType(Pos->second, FromT.getLocalQualifiers());

  // Import the type
  ASTNodeImporter Importer(*this);
  ExpectedType ToTOrErr = Importer.Visit(FromTy);
  if (!ToTOrErr)
    return ToTOrErr.takeError();

  // Record the imported type.
  ImportedTypes[FromTy] = (*ToTOrErr).getTypePtr();

  return ToContext.getQualifiedType(*ToTOrErr, FromT.getLocalQualifiers());
}

Expected<TypeSourceInfo *> ASTImporter::Import(TypeSourceInfo *FromTSI) {
  if (!FromTSI)
    return FromTSI;

  // FIXME: For now we just create a "trivial" type source info based
  // on the type and a single location. Implement a real version of this.
  ExpectedType TOrErr = Import(FromTSI->getType());
  if (!TOrErr)
    return TOrErr.takeError();
  ExpectedSLoc BeginLocOrErr = Import(FromTSI->getTypeLoc().getBeginLoc());
  if (!BeginLocOrErr)
    return BeginLocOrErr.takeError();

  return ToContext.getTrivialTypeSourceInfo(*TOrErr, *BeginLocOrErr);
}

Expected<Attr *> ASTImporter::Import(const Attr *FromAttr) {
  Attr *ToAttr = FromAttr->clone(ToContext);
  if (auto ToRangeOrErr = Import(FromAttr->getRange()))
    ToAttr->setRange(*ToRangeOrErr);
  else
    return ToRangeOrErr.takeError();

  return ToAttr;
}

Decl *ASTImporter::GetAlreadyImportedOrNull(const Decl *FromD) const {
  auto Pos = ImportedDecls.find(FromD);
  if (Pos != ImportedDecls.end())
    return Pos->second;
  else
    return nullptr;
}

TranslationUnitDecl *ASTImporter::GetFromTU(Decl *ToD) {
  auto FromDPos = ImportedFromDecls.find(ToD);
  if (FromDPos == ImportedFromDecls.end())
    return nullptr;
  return FromDPos->second->getTranslationUnitDecl();
}

Expected<Decl *> ASTImporter::Import(Decl *FromD) {
  if (!FromD)
    return nullptr;

  // Push FromD to the stack, and remove that when we return.
  ImportPath.push(FromD);
  auto ImportPathBuilder =
      llvm::make_scope_exit([this]() { ImportPath.pop(); });

  // Check whether there was a previous failed import.
  // If yes return the existing error.
  if (auto Error = getImportDeclErrorIfAny(FromD))
    return make_error<ImportError>(*Error);

  // Check whether we've already imported this declaration.
  Decl *ToD = GetAlreadyImportedOrNull(FromD);
  if (ToD) {
    // Already imported (possibly from another TU) and with an error.
    if (auto Error = SharedState->getImportDeclErrorIfAny(ToD)) {
      setImportDeclError(FromD, *Error);
      return make_error<ImportError>(*Error);
    }

    // If FromD has some updated flags after last import, apply it
    updateFlags(FromD, ToD);
    // If we encounter a cycle during an import then we save the relevant part
    // of the import path associated to the Decl.
    if (ImportPath.hasCycleAtBack())
      SavedImportPaths[FromD].push_back(ImportPath.copyCycleAtBack());
    return ToD;
  }

  // Import the declaration.
  ExpectedDecl ToDOrErr = ImportImpl(FromD);
  if (!ToDOrErr) {
    // Failed to import.

    auto Pos = ImportedDecls.find(FromD);
    if (Pos != ImportedDecls.end()) {
      // Import failed after the object was created.
      // Remove all references to it.
      auto *ToD = Pos->second;
      ImportedDecls.erase(Pos);

      // ImportedDecls and ImportedFromDecls are not symmetric.  It may happen
      // (e.g. with namespaces) that several decls from the 'from' context are
      // mapped to the same decl in the 'to' context.  If we removed entries
      // from the LookupTable here then we may end up removing them multiple
      // times.

      // The Lookuptable contains decls only which are in the 'to' context.
      // Remove from the Lookuptable only if it is *imported* into the 'to'
      // context (and do not remove it if it was added during the initial
      // traverse of the 'to' context).
      auto PosF = ImportedFromDecls.find(ToD);
      if (PosF != ImportedFromDecls.end()) {
        SharedState->removeDeclFromLookup(ToD);
        ImportedFromDecls.erase(PosF);
      }

      // FIXME: AST may contain remaining references to the failed object.
      // However, the ImportDeclErrors in the shared state contains all the
      // failed objects together with their error.
    }

    // Error encountered for the first time.
    // After takeError the error is not usable any more in ToDOrErr.
    // Get a copy of the error object (any more simple solution for this?).
    ImportError ErrOut;
    handleAllErrors(ToDOrErr.takeError(),
                    [&ErrOut](const ImportError &E) { ErrOut = E; });
    setImportDeclError(FromD, ErrOut);
    // Set the error for the mapped to Decl, which is in the "to" context.
    if (Pos != ImportedDecls.end())
      SharedState->setImportDeclError(Pos->second, ErrOut);

    // Set the error for all nodes which have been created before we
    // recognized the error.
    for (const auto &Path : SavedImportPaths[FromD])
      for (Decl *FromDi : Path) {
        setImportDeclError(FromDi, ErrOut);
        //FIXME Should we remove these Decls from ImportedDecls?
        // Set the error for the mapped to Decl, which is in the "to" context.
        auto Ii = ImportedDecls.find(FromDi);
        if (Ii != ImportedDecls.end())
          SharedState->setImportDeclError(Ii->second, ErrOut);
          // FIXME Should we remove these Decls from the LookupTable,
          // and from ImportedFromDecls?
      }
    SavedImportPaths[FromD].clear();

    // Do not return ToDOrErr, error was taken out of it.
    return make_error<ImportError>(ErrOut);
  }

  ToD = *ToDOrErr;

  // FIXME: Handle the "already imported with error" case. We can get here
  // nullptr only if GetImportedOrCreateDecl returned nullptr (after a
  // previously failed create was requested).
  // Later GetImportedOrCreateDecl can be updated to return the error.
  if (!ToD) {
    auto Err = getImportDeclErrorIfAny(FromD);
    assert(Err);
    return make_error<ImportError>(*Err);
  }

  // We could import from the current TU without error.  But previously we
  // already had imported a Decl as `ToD` from another TU (with another
  // ASTImporter object) and with an error.
  if (auto Error = SharedState->getImportDeclErrorIfAny(ToD)) {
    setImportDeclError(FromD, *Error);
    return make_error<ImportError>(*Error);
  }

  // Make sure that ImportImpl registered the imported decl.
  assert(ImportedDecls.count(FromD) != 0 && "Missing call to MapImported?");

  // Notify subclasses.
  Imported(FromD, ToD);

  updateFlags(FromD, ToD);
  SavedImportPaths[FromD].clear();
  return ToDOrErr;
}

Expected<DeclContext *> ASTImporter::ImportContext(DeclContext *FromDC) {
  if (!FromDC)
    return FromDC;

  ExpectedDecl ToDCOrErr = Import(cast<Decl>(FromDC));
  if (!ToDCOrErr)
    return ToDCOrErr.takeError();
  auto *ToDC = cast<DeclContext>(*ToDCOrErr);

  // When we're using a record/enum/Objective-C class/protocol as a context, we
  // need it to have a definition.
  if (auto *ToRecord = dyn_cast<RecordDecl>(ToDC)) {
    auto *FromRecord = cast<RecordDecl>(FromDC);
    if (ToRecord->isCompleteDefinition()) {
      // Do nothing.
    } else if (FromRecord->isCompleteDefinition()) {
      if (Error Err = ASTNodeImporter(*this).ImportDefinition(
          FromRecord, ToRecord, ASTNodeImporter::IDK_Basic))
        return std::move(Err);
    } else {
      CompleteDecl(ToRecord);
    }
  } else if (auto *ToEnum = dyn_cast<EnumDecl>(ToDC)) {
    auto *FromEnum = cast<EnumDecl>(FromDC);
    if (ToEnum->isCompleteDefinition()) {
      // Do nothing.
    } else if (FromEnum->isCompleteDefinition()) {
      if (Error Err = ASTNodeImporter(*this).ImportDefinition(
          FromEnum, ToEnum, ASTNodeImporter::IDK_Basic))
        return std::move(Err);
    } else {
      CompleteDecl(ToEnum);
    }
  } else if (auto *ToClass = dyn_cast<ObjCInterfaceDecl>(ToDC)) {
    auto *FromClass = cast<ObjCInterfaceDecl>(FromDC);
    if (ToClass->getDefinition()) {
      // Do nothing.
    } else if (ObjCInterfaceDecl *FromDef = FromClass->getDefinition()) {
      if (Error Err = ASTNodeImporter(*this).ImportDefinition(
          FromDef, ToClass, ASTNodeImporter::IDK_Basic))
        return std::move(Err);
    } else {
      CompleteDecl(ToClass);
    }
  } else if (auto *ToProto = dyn_cast<ObjCProtocolDecl>(ToDC)) {
    auto *FromProto = cast<ObjCProtocolDecl>(FromDC);
    if (ToProto->getDefinition()) {
      // Do nothing.
    } else if (ObjCProtocolDecl *FromDef = FromProto->getDefinition()) {
      if (Error Err = ASTNodeImporter(*this).ImportDefinition(
          FromDef, ToProto, ASTNodeImporter::IDK_Basic))
        return std::move(Err);
    } else {
      CompleteDecl(ToProto);
    }
  }

  return ToDC;
}

Expected<Expr *> ASTImporter::Import(Expr *FromE) {
  if (ExpectedStmt ToSOrErr = Import(cast_or_null<Stmt>(FromE)))
    return cast_or_null<Expr>(*ToSOrErr);
  else
    return ToSOrErr.takeError();
}

Expected<Stmt *> ASTImporter::Import(Stmt *FromS) {
  if (!FromS)
    return nullptr;

  // Check whether we've already imported this statement.
  llvm::DenseMap<Stmt *, Stmt *>::iterator Pos = ImportedStmts.find(FromS);
  if (Pos != ImportedStmts.end())
    return Pos->second;

  // Import the statement.
  ASTNodeImporter Importer(*this);
  ExpectedStmt ToSOrErr = Importer.Visit(FromS);
  if (!ToSOrErr)
    return ToSOrErr;

  if (auto *ToE = dyn_cast<Expr>(*ToSOrErr)) {
    auto *FromE = cast<Expr>(FromS);
    // Copy ExprBitfields, which may not be handled in Expr subclasses
    // constructors.
    ToE->setValueKind(FromE->getValueKind());
    ToE->setObjectKind(FromE->getObjectKind());
    ToE->setTypeDependent(FromE->isTypeDependent());
    ToE->setValueDependent(FromE->isValueDependent());
    ToE->setInstantiationDependent(FromE->isInstantiationDependent());
    ToE->setContainsUnexpandedParameterPack(
        FromE->containsUnexpandedParameterPack());
  }

  // Record the imported statement object.
  ImportedStmts[FromS] = *ToSOrErr;
  return ToSOrErr;
}

Expected<NestedNameSpecifier *>
ASTImporter::Import(NestedNameSpecifier *FromNNS) {
  if (!FromNNS)
    return nullptr;

  NestedNameSpecifier *Prefix = nullptr;
  if (Error Err = importInto(Prefix, FromNNS->getPrefix()))
    return std::move(Err);

  switch (FromNNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    assert(FromNNS->getAsIdentifier() && "NNS should contain identifier.");
    return NestedNameSpecifier::Create(ToContext, Prefix,
                                       Import(FromNNS->getAsIdentifier()));

  case NestedNameSpecifier::Namespace:
    if (ExpectedDecl NSOrErr = Import(FromNNS->getAsNamespace())) {
      return NestedNameSpecifier::Create(ToContext, Prefix,
                                         cast<NamespaceDecl>(*NSOrErr));
    } else
      return NSOrErr.takeError();

  case NestedNameSpecifier::NamespaceAlias:
    if (ExpectedDecl NSADOrErr = Import(FromNNS->getAsNamespaceAlias()))
      return NestedNameSpecifier::Create(ToContext, Prefix,
                                         cast<NamespaceAliasDecl>(*NSADOrErr));
    else
      return NSADOrErr.takeError();

  case NestedNameSpecifier::Global:
    return NestedNameSpecifier::GlobalSpecifier(ToContext);

  case NestedNameSpecifier::Super:
    if (ExpectedDecl RDOrErr = Import(FromNNS->getAsRecordDecl()))
      return NestedNameSpecifier::SuperSpecifier(ToContext,
                                                 cast<CXXRecordDecl>(*RDOrErr));
    else
      return RDOrErr.takeError();

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    if (Expected<QualType> TyOrErr =
            Import(QualType(FromNNS->getAsType(), 0u))) {
      bool TSTemplate =
          FromNNS->getKind() == NestedNameSpecifier::TypeSpecWithTemplate;
      return NestedNameSpecifier::Create(ToContext, Prefix, TSTemplate,
                                         TyOrErr->getTypePtr());
    } else {
      return TyOrErr.takeError();
    }
  }

  llvm_unreachable("Invalid nested name specifier kind");
}

Expected<NestedNameSpecifierLoc>
ASTImporter::Import(NestedNameSpecifierLoc FromNNS) {
  // Copied from NestedNameSpecifier mostly.
  SmallVector<NestedNameSpecifierLoc , 8> NestedNames;
  NestedNameSpecifierLoc NNS = FromNNS;

  // Push each of the nested-name-specifiers's onto a stack for
  // serialization in reverse order.
  while (NNS) {
    NestedNames.push_back(NNS);
    NNS = NNS.getPrefix();
  }

  NestedNameSpecifierLocBuilder Builder;

  while (!NestedNames.empty()) {
    NNS = NestedNames.pop_back_val();
    NestedNameSpecifier *Spec = nullptr;
    if (Error Err = importInto(Spec, NNS.getNestedNameSpecifier()))
      return std::move(Err);

    NestedNameSpecifier::SpecifierKind Kind = Spec->getKind();

    SourceLocation ToLocalBeginLoc, ToLocalEndLoc;
    if (Kind != NestedNameSpecifier::Super) {
      if (Error Err = importInto(ToLocalBeginLoc, NNS.getLocalBeginLoc()))
        return std::move(Err);

      if (Kind != NestedNameSpecifier::Global)
        if (Error Err = importInto(ToLocalEndLoc, NNS.getLocalEndLoc()))
          return std::move(Err);
    }

    switch (Kind) {
    case NestedNameSpecifier::Identifier:
      Builder.Extend(getToContext(), Spec->getAsIdentifier(), ToLocalBeginLoc,
                     ToLocalEndLoc);
      break;

    case NestedNameSpecifier::Namespace:
      Builder.Extend(getToContext(), Spec->getAsNamespace(), ToLocalBeginLoc,
                     ToLocalEndLoc);
      break;

    case NestedNameSpecifier::NamespaceAlias:
      Builder.Extend(getToContext(), Spec->getAsNamespaceAlias(),
                     ToLocalBeginLoc, ToLocalEndLoc);
      break;

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate: {
      SourceLocation ToTLoc;
      if (Error Err = importInto(ToTLoc, NNS.getTypeLoc().getBeginLoc()))
        return std::move(Err);
      TypeSourceInfo *TSI = getToContext().getTrivialTypeSourceInfo(
            QualType(Spec->getAsType(), 0), ToTLoc);
      if (Kind == NestedNameSpecifier::TypeSpecWithTemplate)
        // ToLocalBeginLoc is here the location of the 'template' keyword.
        Builder.Extend(getToContext(), ToLocalBeginLoc, TSI->getTypeLoc(),
                       ToLocalEndLoc);
      else
        // No location for 'template' keyword here.
        Builder.Extend(getToContext(), SourceLocation{}, TSI->getTypeLoc(),
                       ToLocalEndLoc);
      break;
    }

    case NestedNameSpecifier::Global:
      Builder.MakeGlobal(getToContext(), ToLocalBeginLoc);
      break;

    case NestedNameSpecifier::Super: {
      auto ToSourceRangeOrErr = Import(NNS.getSourceRange());
      if (!ToSourceRangeOrErr)
        return ToSourceRangeOrErr.takeError();

      Builder.MakeSuper(getToContext(), Spec->getAsRecordDecl(),
                        ToSourceRangeOrErr->getBegin(),
                        ToSourceRangeOrErr->getEnd());
    }
  }
  }

  return Builder.getWithLocInContext(getToContext());
}

Expected<TemplateName> ASTImporter::Import(TemplateName From) {
  switch (From.getKind()) {
  case TemplateName::Template:
    if (ExpectedDecl ToTemplateOrErr = Import(From.getAsTemplateDecl()))
      return TemplateName(cast<TemplateDecl>(*ToTemplateOrErr));
    else
      return ToTemplateOrErr.takeError();

  case TemplateName::OverloadedTemplate: {
    OverloadedTemplateStorage *FromStorage = From.getAsOverloadedTemplate();
    UnresolvedSet<2> ToTemplates;
    for (auto *I : *FromStorage) {
      if (auto ToOrErr = Import(I))
        ToTemplates.addDecl(cast<NamedDecl>(*ToOrErr));
      else
        return ToOrErr.takeError();
    }
    return ToContext.getOverloadedTemplateName(ToTemplates.begin(),
                                               ToTemplates.end());
  }

  case TemplateName::AssumedTemplate: {
    AssumedTemplateStorage *FromStorage = From.getAsAssumedTemplateName();
    auto DeclNameOrErr = Import(FromStorage->getDeclName());
    if (!DeclNameOrErr)
      return DeclNameOrErr.takeError();
    return ToContext.getAssumedTemplateName(*DeclNameOrErr);
  }

  case TemplateName::QualifiedTemplate: {
    QualifiedTemplateName *QTN = From.getAsQualifiedTemplateName();
    auto QualifierOrErr = Import(QTN->getQualifier());
    if (!QualifierOrErr)
      return QualifierOrErr.takeError();

    if (ExpectedDecl ToTemplateOrErr = Import(From.getAsTemplateDecl()))
      return ToContext.getQualifiedTemplateName(
          *QualifierOrErr, QTN->hasTemplateKeyword(),
          cast<TemplateDecl>(*ToTemplateOrErr));
    else
      return ToTemplateOrErr.takeError();
  }

  case TemplateName::DependentTemplate: {
    DependentTemplateName *DTN = From.getAsDependentTemplateName();
    auto QualifierOrErr = Import(DTN->getQualifier());
    if (!QualifierOrErr)
      return QualifierOrErr.takeError();

    if (DTN->isIdentifier()) {
      return ToContext.getDependentTemplateName(*QualifierOrErr,
                                                Import(DTN->getIdentifier()));
    }

    return ToContext.getDependentTemplateName(*QualifierOrErr,
                                              DTN->getOperator());
  }

  case TemplateName::SubstTemplateTemplateParm: {
    SubstTemplateTemplateParmStorage *Subst =
        From.getAsSubstTemplateTemplateParm();
    ExpectedDecl ParamOrErr = Import(Subst->getParameter());
    if (!ParamOrErr)
      return ParamOrErr.takeError();

    auto ReplacementOrErr = Import(Subst->getReplacement());
    if (!ReplacementOrErr)
      return ReplacementOrErr.takeError();

    return ToContext.getSubstTemplateTemplateParm(
        cast<TemplateTemplateParmDecl>(*ParamOrErr), *ReplacementOrErr);
  }

  case TemplateName::SubstTemplateTemplateParmPack: {
    SubstTemplateTemplateParmPackStorage *SubstPack
      = From.getAsSubstTemplateTemplateParmPack();
    ExpectedDecl ParamOrErr = Import(SubstPack->getParameterPack());
    if (!ParamOrErr)
      return ParamOrErr.takeError();

    ASTNodeImporter Importer(*this);
    auto ArgPackOrErr =
        Importer.ImportTemplateArgument(SubstPack->getArgumentPack());
    if (!ArgPackOrErr)
      return ArgPackOrErr.takeError();

    return ToContext.getSubstTemplateTemplateParmPack(
        cast<TemplateTemplateParmDecl>(*ParamOrErr), *ArgPackOrErr);
  }
  }

  llvm_unreachable("Invalid template name kind");
}

Expected<SourceLocation> ASTImporter::Import(SourceLocation FromLoc) {
  if (FromLoc.isInvalid())
    return SourceLocation{};

  SourceManager &FromSM = FromContext.getSourceManager();
  bool IsBuiltin = FromSM.isWrittenInBuiltinFile(FromLoc);

  std::pair<FileID, unsigned> Decomposed = FromSM.getDecomposedLoc(FromLoc);
  Expected<FileID> ToFileIDOrErr = Import(Decomposed.first, IsBuiltin);
  if (!ToFileIDOrErr)
    return ToFileIDOrErr.takeError();
  SourceManager &ToSM = ToContext.getSourceManager();
  return ToSM.getComposedLoc(*ToFileIDOrErr, Decomposed.second);
}

Expected<SourceRange> ASTImporter::Import(SourceRange FromRange) {
  SourceLocation ToBegin, ToEnd;
  if (Error Err = importInto(ToBegin, FromRange.getBegin()))
    return std::move(Err);
  if (Error Err = importInto(ToEnd, FromRange.getEnd()))
    return std::move(Err);

  return SourceRange(ToBegin, ToEnd);
}

Expected<FileID> ASTImporter::Import(FileID FromID, bool IsBuiltin) {
  llvm::DenseMap<FileID, FileID>::iterator Pos = ImportedFileIDs.find(FromID);
  if (Pos != ImportedFileIDs.end())
    return Pos->second;

  SourceManager &FromSM = FromContext.getSourceManager();
  SourceManager &ToSM = ToContext.getSourceManager();
  const SrcMgr::SLocEntry &FromSLoc = FromSM.getSLocEntry(FromID);

  // Map the FromID to the "to" source manager.
  FileID ToID;
  if (FromSLoc.isExpansion()) {
    const SrcMgr::ExpansionInfo &FromEx = FromSLoc.getExpansion();
    ExpectedSLoc ToSpLoc = Import(FromEx.getSpellingLoc());
    if (!ToSpLoc)
      return ToSpLoc.takeError();
    ExpectedSLoc ToExLocS = Import(FromEx.getExpansionLocStart());
    if (!ToExLocS)
      return ToExLocS.takeError();
    unsigned TokenLen = FromSM.getFileIDSize(FromID);
    SourceLocation MLoc;
    if (FromEx.isMacroArgExpansion()) {
      MLoc = ToSM.createMacroArgExpansionLoc(*ToSpLoc, *ToExLocS, TokenLen);
    } else {
      if (ExpectedSLoc ToExLocE = Import(FromEx.getExpansionLocEnd()))
        MLoc = ToSM.createExpansionLoc(*ToSpLoc, *ToExLocS, *ToExLocE, TokenLen,
                                       FromEx.isExpansionTokenRange());
      else
        return ToExLocE.takeError();
    }
    ToID = ToSM.getFileID(MLoc);
  } else {
    const SrcMgr::ContentCache *Cache = FromSLoc.getFile().getContentCache();

    if (!IsBuiltin) {
      // Include location of this file.
      ExpectedSLoc ToIncludeLoc = Import(FromSLoc.getFile().getIncludeLoc());
      if (!ToIncludeLoc)
        return ToIncludeLoc.takeError();

      if (Cache->OrigEntry && Cache->OrigEntry->getDir()) {
        // FIXME: We probably want to use getVirtualFile(), so we don't hit the
        // disk again
        // FIXME: We definitely want to re-use the existing MemoryBuffer, rather
        // than mmap the files several times.
        auto Entry =
            ToFileManager.getFile(Cache->OrigEntry->getName());
        // FIXME: The filename may be a virtual name that does probably not
        // point to a valid file and we get no Entry here. In this case try with
        // the memory buffer below.
        if (Entry)
          ToID = ToSM.createFileID(*Entry, *ToIncludeLoc,
                                   FromSLoc.getFile().getFileCharacteristic());
      }
    }

    if (ToID.isInvalid() || IsBuiltin) {
      // FIXME: We want to re-use the existing MemoryBuffer!
      bool Invalid = true;
      const llvm::MemoryBuffer *FromBuf =
          Cache->getBuffer(FromContext.getDiagnostics(),
                           FromSM.getFileManager(), SourceLocation{}, &Invalid);
      if (!FromBuf || Invalid)
        // FIXME: Use a new error kind?
        return llvm::make_error<ImportError>(ImportError::Unknown);

      std::unique_ptr<llvm::MemoryBuffer> ToBuf =
          llvm::MemoryBuffer::getMemBufferCopy(FromBuf->getBuffer(),
                                               FromBuf->getBufferIdentifier());
      ToID = ToSM.createFileID(std::move(ToBuf),
                               FromSLoc.getFile().getFileCharacteristic());
    }
  }

  assert(ToID.isValid() && "Unexpected invalid fileID was created.");

  ImportedFileIDs[FromID] = ToID;

  if (FileIDImportHandler)
    FileIDImportHandler(ToID, FromID);

  return ToID;
}

Expected<CXXCtorInitializer *> ASTImporter::Import(CXXCtorInitializer *From) {
  ExpectedExpr ToExprOrErr = Import(From->getInit());
  if (!ToExprOrErr)
    return ToExprOrErr.takeError();

  auto LParenLocOrErr = Import(From->getLParenLoc());
  if (!LParenLocOrErr)
    return LParenLocOrErr.takeError();

  auto RParenLocOrErr = Import(From->getRParenLoc());
  if (!RParenLocOrErr)
    return RParenLocOrErr.takeError();

  if (From->isBaseInitializer()) {
    auto ToTInfoOrErr = Import(From->getTypeSourceInfo());
    if (!ToTInfoOrErr)
      return ToTInfoOrErr.takeError();

    SourceLocation EllipsisLoc;
    if (From->isPackExpansion())
      if (Error Err = importInto(EllipsisLoc, From->getEllipsisLoc()))
        return std::move(Err);

    return new (ToContext) CXXCtorInitializer(
        ToContext, *ToTInfoOrErr, From->isBaseVirtual(), *LParenLocOrErr,
        *ToExprOrErr, *RParenLocOrErr, EllipsisLoc);
  } else if (From->isMemberInitializer()) {
    ExpectedDecl ToFieldOrErr = Import(From->getMember());
    if (!ToFieldOrErr)
      return ToFieldOrErr.takeError();

    auto MemberLocOrErr = Import(From->getMemberLocation());
    if (!MemberLocOrErr)
      return MemberLocOrErr.takeError();

    return new (ToContext) CXXCtorInitializer(
        ToContext, cast_or_null<FieldDecl>(*ToFieldOrErr), *MemberLocOrErr,
        *LParenLocOrErr, *ToExprOrErr, *RParenLocOrErr);
  } else if (From->isIndirectMemberInitializer()) {
    ExpectedDecl ToIFieldOrErr = Import(From->getIndirectMember());
    if (!ToIFieldOrErr)
      return ToIFieldOrErr.takeError();

    auto MemberLocOrErr = Import(From->getMemberLocation());
    if (!MemberLocOrErr)
      return MemberLocOrErr.takeError();

    return new (ToContext) CXXCtorInitializer(
        ToContext, cast_or_null<IndirectFieldDecl>(*ToIFieldOrErr),
        *MemberLocOrErr, *LParenLocOrErr, *ToExprOrErr, *RParenLocOrErr);
  } else if (From->isDelegatingInitializer()) {
    auto ToTInfoOrErr = Import(From->getTypeSourceInfo());
    if (!ToTInfoOrErr)
      return ToTInfoOrErr.takeError();

    return new (ToContext)
        CXXCtorInitializer(ToContext, *ToTInfoOrErr, *LParenLocOrErr,
                           *ToExprOrErr, *RParenLocOrErr);
  } else {
    // FIXME: assert?
    return make_error<ImportError>();
  }
}

Expected<CXXBaseSpecifier *>
ASTImporter::Import(const CXXBaseSpecifier *BaseSpec) {
  auto Pos = ImportedCXXBaseSpecifiers.find(BaseSpec);
  if (Pos != ImportedCXXBaseSpecifiers.end())
    return Pos->second;

  Expected<SourceRange> ToSourceRange = Import(BaseSpec->getSourceRange());
  if (!ToSourceRange)
    return ToSourceRange.takeError();
  Expected<TypeSourceInfo *> ToTSI = Import(BaseSpec->getTypeSourceInfo());
  if (!ToTSI)
    return ToTSI.takeError();
  ExpectedSLoc ToEllipsisLoc = Import(BaseSpec->getEllipsisLoc());
  if (!ToEllipsisLoc)
    return ToEllipsisLoc.takeError();
  CXXBaseSpecifier *Imported = new (ToContext) CXXBaseSpecifier(
      *ToSourceRange, BaseSpec->isVirtual(), BaseSpec->isBaseOfClass(),
      BaseSpec->getAccessSpecifierAsWritten(), *ToTSI, *ToEllipsisLoc);
  ImportedCXXBaseSpecifiers[BaseSpec] = Imported;
  return Imported;
}

Error ASTImporter::ImportDefinition(Decl *From) {
  ExpectedDecl ToOrErr = Import(From);
  if (!ToOrErr)
    return ToOrErr.takeError();
  Decl *To = *ToOrErr;

  auto *FromDC = cast<DeclContext>(From);
  ASTNodeImporter Importer(*this);

  if (auto *ToRecord = dyn_cast<RecordDecl>(To)) {
    if (!ToRecord->getDefinition()) {
      return Importer.ImportDefinition(
          cast<RecordDecl>(FromDC), ToRecord,
          ASTNodeImporter::IDK_Everything);
    }
  }

  if (auto *ToEnum = dyn_cast<EnumDecl>(To)) {
    if (!ToEnum->getDefinition()) {
      return Importer.ImportDefinition(
          cast<EnumDecl>(FromDC), ToEnum, ASTNodeImporter::IDK_Everything);
    }
  }

  if (auto *ToIFace = dyn_cast<ObjCInterfaceDecl>(To)) {
    if (!ToIFace->getDefinition()) {
      return Importer.ImportDefinition(
          cast<ObjCInterfaceDecl>(FromDC), ToIFace,
          ASTNodeImporter::IDK_Everything);
    }
  }

  if (auto *ToProto = dyn_cast<ObjCProtocolDecl>(To)) {
    if (!ToProto->getDefinition()) {
      return Importer.ImportDefinition(
          cast<ObjCProtocolDecl>(FromDC), ToProto,
          ASTNodeImporter::IDK_Everything);
    }
  }

  return Importer.ImportDeclContext(FromDC, true);
}

Expected<DeclarationName> ASTImporter::Import(DeclarationName FromName) {
  if (!FromName)
    return DeclarationName{};

  switch (FromName.getNameKind()) {
  case DeclarationName::Identifier:
    return DeclarationName(Import(FromName.getAsIdentifierInfo()));

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    if (auto ToSelOrErr = Import(FromName.getObjCSelector()))
      return DeclarationName(*ToSelOrErr);
    else
      return ToSelOrErr.takeError();

  case DeclarationName::CXXConstructorName: {
    if (auto ToTyOrErr = Import(FromName.getCXXNameType()))
      return ToContext.DeclarationNames.getCXXConstructorName(
          ToContext.getCanonicalType(*ToTyOrErr));
    else
      return ToTyOrErr.takeError();
  }

  case DeclarationName::CXXDestructorName: {
    if (auto ToTyOrErr = Import(FromName.getCXXNameType()))
      return ToContext.DeclarationNames.getCXXDestructorName(
          ToContext.getCanonicalType(*ToTyOrErr));
    else
      return ToTyOrErr.takeError();
  }

  case DeclarationName::CXXDeductionGuideName: {
    if (auto ToTemplateOrErr = Import(FromName.getCXXDeductionGuideTemplate()))
      return ToContext.DeclarationNames.getCXXDeductionGuideName(
          cast<TemplateDecl>(*ToTemplateOrErr));
    else
      return ToTemplateOrErr.takeError();
  }

  case DeclarationName::CXXConversionFunctionName: {
    if (auto ToTyOrErr = Import(FromName.getCXXNameType()))
      return ToContext.DeclarationNames.getCXXConversionFunctionName(
          ToContext.getCanonicalType(*ToTyOrErr));
    else
      return ToTyOrErr.takeError();
  }

  case DeclarationName::CXXOperatorName:
    return ToContext.DeclarationNames.getCXXOperatorName(
                                          FromName.getCXXOverloadedOperator());

  case DeclarationName::CXXLiteralOperatorName:
    return ToContext.DeclarationNames.getCXXLiteralOperatorName(
        Import(FromName.getCXXLiteralIdentifier()));

  case DeclarationName::CXXUsingDirective:
    // FIXME: STATICS!
    return DeclarationName::getUsingDirectiveName();
  }

  llvm_unreachable("Invalid DeclarationName Kind!");
}

IdentifierInfo *ASTImporter::Import(const IdentifierInfo *FromId) {
  if (!FromId)
    return nullptr;

  IdentifierInfo *ToId = &ToContext.Idents.get(FromId->getName());

  if (!ToId->getBuiltinID() && FromId->getBuiltinID())
    ToId->setBuiltinID(FromId->getBuiltinID());

  return ToId;
}

Expected<Selector> ASTImporter::Import(Selector FromSel) {
  if (FromSel.isNull())
    return Selector{};

  SmallVector<IdentifierInfo *, 4> Idents;
  Idents.push_back(Import(FromSel.getIdentifierInfoForSlot(0)));
  for (unsigned I = 1, N = FromSel.getNumArgs(); I < N; ++I)
    Idents.push_back(Import(FromSel.getIdentifierInfoForSlot(I)));
  return ToContext.Selectors.getSelector(FromSel.getNumArgs(), Idents.data());
}

Expected<DeclarationName> ASTImporter::HandleNameConflict(DeclarationName Name,
                                                          DeclContext *DC,
                                                          unsigned IDNS,
                                                          NamedDecl **Decls,
                                                          unsigned NumDecls) {
  if (ODRHandling == ODRHandlingType::Conservative)
    // Report error at any name conflict.
    return make_error<ImportError>(ImportError::NameConflict);
  else
    // Allow to create the new Decl with the same name.
    return Name;
}

DiagnosticBuilder ASTImporter::ToDiag(SourceLocation Loc, unsigned DiagID) {
  if (LastDiagFromFrom)
    ToContext.getDiagnostics().notePriorDiagnosticFrom(
      FromContext.getDiagnostics());
  LastDiagFromFrom = false;
  return ToContext.getDiagnostics().Report(Loc, DiagID);
}

DiagnosticBuilder ASTImporter::FromDiag(SourceLocation Loc, unsigned DiagID) {
  if (!LastDiagFromFrom)
    FromContext.getDiagnostics().notePriorDiagnosticFrom(
      ToContext.getDiagnostics());
  LastDiagFromFrom = true;
  return FromContext.getDiagnostics().Report(Loc, DiagID);
}

void ASTImporter::CompleteDecl (Decl *D) {
  if (auto *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    if (!ID->getDefinition())
      ID->startDefinition();
  }
  else if (auto *PD = dyn_cast<ObjCProtocolDecl>(D)) {
    if (!PD->getDefinition())
      PD->startDefinition();
  }
  else if (auto *TD = dyn_cast<TagDecl>(D)) {
    if (!TD->getDefinition() && !TD->isBeingDefined()) {
      TD->startDefinition();
      TD->setCompleteDefinition(true);
    }
  }
  else {
    assert(0 && "CompleteDecl called on a Decl that can't be completed");
  }
}

Decl *ASTImporter::MapImported(Decl *From, Decl *To) {
  llvm::DenseMap<Decl *, Decl *>::iterator Pos = ImportedDecls.find(From);
  assert((Pos == ImportedDecls.end() || Pos->second == To) &&
      "Try to import an already imported Decl");
  if (Pos != ImportedDecls.end())
    return Pos->second;
  ImportedDecls[From] = To;
  // This mapping should be maintained only in this function. Therefore do not
  // check for additional consistency.
  ImportedFromDecls[To] = From;
  AddToLookupTable(To);
  return To;
}

llvm::Optional<ImportError>
ASTImporter::getImportDeclErrorIfAny(Decl *FromD) const {
  auto Pos = ImportDeclErrors.find(FromD);
  if (Pos != ImportDeclErrors.end())
    return Pos->second;
  else
    return Optional<ImportError>();
}

void ASTImporter::setImportDeclError(Decl *From, ImportError Error) {
  auto InsertRes = ImportDeclErrors.insert({From, Error});
  (void)InsertRes;
  // Either we set the error for the first time, or we already had set one and
  // now we want to set the same error.
  assert(InsertRes.second || InsertRes.first->second.Error == Error.Error);
}

bool ASTImporter::IsStructurallyEquivalent(QualType From, QualType To,
                                           bool Complain) {
  llvm::DenseMap<const Type *, const Type *>::iterator Pos =
      ImportedTypes.find(From.getTypePtr());
  if (Pos != ImportedTypes.end()) {
    if (ExpectedType ToFromOrErr = Import(From)) {
      if (ToContext.hasSameType(*ToFromOrErr, To))
        return true;
    } else {
      llvm::consumeError(ToFromOrErr.takeError());
    }
  }

  StructuralEquivalenceContext Ctx(FromContext, ToContext, NonEquivalentDecls,
                                   getStructuralEquivalenceKind(*this), false,
                                   Complain);
  return Ctx.IsEquivalent(From, To);
}
