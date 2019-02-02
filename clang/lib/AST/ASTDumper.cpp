//===--- ASTDumper.cpp - Dumping implementation for ASTs ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump methods, which dump out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/AttrVisitor.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/LocInfoType.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TemplateArgumentVisitor.h"
#include "clang/AST/TextNodeDumper.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace clang::comments;

//===----------------------------------------------------------------------===//
// ASTDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {

  class ASTDumper
      : public ConstDeclVisitor<ASTDumper>,
        public ConstStmtVisitor<ASTDumper>,
        public ConstCommentVisitor<ASTDumper, void, const FullComment *>,
        public TypeVisitor<ASTDumper>,
        public ConstAttrVisitor<ASTDumper>,
        public ConstTemplateArgumentVisitor<ASTDumper> {

    TextNodeDumper NodeDumper;

    raw_ostream &OS;

    /// Indicates whether we should trigger deserialization of nodes that had
    /// not already been loaded.
    bool Deserialize = false;

    const bool ShowColors;

  public:
    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM)
        : ASTDumper(OS, Traits, SM,
                    SM && SM->getDiagnostics().getShowColors()) {}

    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM, bool ShowColors)
        : ASTDumper(OS, Traits, SM, ShowColors, LangOptions()) {}
    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM, bool ShowColors,
              const PrintingPolicy &PrintPolicy)
        : NodeDumper(OS, ShowColors, SM, PrintPolicy, Traits), OS(OS),
          ShowColors(ShowColors) {}

    void setDeserialize(bool D) { Deserialize = D; }

    void Visit(const Decl *D) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(D);
        if (!D)
          return;

        ConstDeclVisitor<ASTDumper>::Visit(D);

        for (const auto &A : D->attrs())
          Visit(A);

        if (const FullComment *Comment =
                D->getASTContext().getLocalCommentForDeclUncached(D))
          Visit(Comment, Comment);

        // Decls within functions are visited by the body.
        if (!isa<FunctionDecl>(*D) && !isa<ObjCMethodDecl>(*D)) {
          if (const auto *DC = dyn_cast<DeclContext>(D))
            dumpDeclContext(DC);
        }
      });
    }

    void Visit(const Stmt *S, StringRef Label = {}) {
      NodeDumper.AddChild(Label, [=] {
        NodeDumper.Visit(S);

        if (!S) {
          return;
        }

        ConstStmtVisitor<ASTDumper>::Visit(S);

        // Some statements have custom mechanisms for dumping their children.
        if (isa<DeclStmt>(S) || isa<GenericSelectionExpr>(S)) {
          return;
        }

        for (const Stmt *SubStmt : S->children())
          Visit(SubStmt);
      });
    }

    void Visit(QualType T) {
      SplitQualType SQT = T.split();
      if (!SQT.Quals.hasQualifiers())
        return Visit(SQT.Ty);

      NodeDumper.AddChild([=] {
        NodeDumper.Visit(T);
        Visit(T.split().Ty);
      });
    }

    void Visit(const Type *T) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(T);
        if (!T)
          return;
        TypeVisitor<ASTDumper>::Visit(T);

        QualType SingleStepDesugar =
            T->getLocallyUnqualifiedSingleStepDesugaredType();
        if (SingleStepDesugar != QualType(T, 0))
          Visit(SingleStepDesugar);
      });
    }

    void Visit(const Attr *A) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(A);
        ConstAttrVisitor<ASTDumper>::Visit(A);
      });
    }

    void Visit(const CXXCtorInitializer *Init) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(Init);
        Visit(Init->getInit());
      });
    }

    void Visit(const TemplateArgument &A, SourceRange R = {},
               const Decl *From = nullptr, const char *Label = nullptr) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(A, R, From, Label);
        ConstTemplateArgumentVisitor<ASTDumper>::Visit(A);
      });
    }

    void Visit(const BlockDecl::Capture &C) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(C);
        if (C.hasCopyExpr())
          Visit(C.getCopyExpr());
      });
    }

    void Visit(const OMPClause *C) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(C);
        for (const auto *S : C->children())
          Visit(S);
      });
    }

    void Visit(const GenericSelectionExpr::ConstAssociation &A) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(A);
        if (const TypeSourceInfo *TSI = A.getTypeSourceInfo())
          Visit(TSI->getType());
        Visit(A.getAssociationExpr());
      });
    }

    void Visit(const Comment *C, const FullComment *FC) {
      NodeDumper.AddChild([=] {
        NodeDumper.Visit(C, FC);
        if (!C) {
          return;
        }
        ConstCommentVisitor<ASTDumper, void, const FullComment *>::visit(C, FC);
        for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
             I != E; ++I)
          Visit(*I, FC);
      });
    }

    void dumpDeclContext(const DeclContext *DC);
    void dumpLookups(const DeclContext *DC, bool DumpDecls);

    void dumpTemplateParameters(const TemplateParameterList *TPL) {
      if (!TPL)
        return;

      for (const auto &TP : *TPL)
        Visit(TP);
    }

    void dumpTemplateArgumentListInfo(const TemplateArgumentListInfo &TALI) {
      for (const auto &TA : TALI.arguments())
        dumpTemplateArgumentLoc(TA);
    }

    void dumpTemplateArgumentLoc(const TemplateArgumentLoc &A,
                                 const Decl *From = nullptr,
                                 const char *Label = nullptr) {
      Visit(A.getArgument(), A.getSourceRange(), From, Label);
    }

    void dumpTemplateArgumentList(const TemplateArgumentList &TAL) {
      for (unsigned i = 0, e = TAL.size(); i < e; ++i)
        Visit(TAL[i]);
    }

    void dumpObjCTypeParamList(const ObjCTypeParamList *typeParams) {
      if (!typeParams)
        return;

      for (const auto &typeParam : *typeParams) {
        Visit(typeParam);
      }
    }

    void VisitComplexType(const ComplexType *T) { Visit(T->getElementType()); }
    void VisitLocInfoType(const LocInfoType *T) {
      Visit(T->getTypeSourceInfo()->getType());
    }
    void VisitPointerType(const PointerType *T) { Visit(T->getPointeeType()); }
    void VisitBlockPointerType(const BlockPointerType *T) {
      Visit(T->getPointeeType());
    }
    void VisitReferenceType(const ReferenceType *T) {
      Visit(T->getPointeeType());
    }
    void VisitMemberPointerType(const MemberPointerType *T) {
      Visit(T->getClass());
      Visit(T->getPointeeType());
    }
    void VisitArrayType(const ArrayType *T) { Visit(T->getElementType()); }
    void VisitVariableArrayType(const VariableArrayType *T) {
      VisitArrayType(T);
      Visit(T->getSizeExpr());
    }
    void VisitDependentSizedArrayType(const DependentSizedArrayType *T) {
      Visit(T->getElementType());
      Visit(T->getSizeExpr());
    }
    void
    VisitDependentSizedExtVectorType(const DependentSizedExtVectorType *T) {
      Visit(T->getElementType());
      Visit(T->getSizeExpr());
    }
    void VisitVectorType(const VectorType *T) { Visit(T->getElementType()); }
    void VisitFunctionType(const FunctionType *T) { Visit(T->getReturnType()); }
    void VisitFunctionProtoType(const FunctionProtoType *T) {
      VisitFunctionType(T);
      for (const QualType &PT : T->getParamTypes())
        Visit(PT);
    }
    void VisitTypeOfExprType(const TypeOfExprType *T) {
      Visit(T->getUnderlyingExpr());
    }
    void VisitDecltypeType(const DecltypeType *T) {
      Visit(T->getUnderlyingExpr());
    }
    void VisitUnaryTransformType(const UnaryTransformType *T) {
      Visit(T->getBaseType());
    }
    void VisitAttributedType(const AttributedType *T) {
      // FIXME: AttrKind
      Visit(T->getModifiedType());
    }
    void VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *T) {
      Visit(T->getReplacedParameter());
    }
    void
    VisitSubstTemplateTypeParmPackType(const SubstTemplateTypeParmPackType *T) {
      Visit(T->getReplacedParameter());
      Visit(T->getArgumentPack());
    }
    void VisitTemplateSpecializationType(const TemplateSpecializationType *T) {
      for (const auto &Arg : *T)
        Visit(Arg);
      if (T->isTypeAlias())
        Visit(T->getAliasedType());
    }
    void VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
      Visit(T->getPointeeType());
    }
    void VisitAtomicType(const AtomicType *T) { Visit(T->getValueType()); }
    void VisitPipeType(const PipeType *T) { Visit(T->getElementType()); }
    void VisitAdjustedType(const AdjustedType *T) {
      Visit(T->getOriginalType());
    }
    void VisitPackExpansionType(const PackExpansionType *T) {
      if (!T->isSugared())
        Visit(T->getPattern());
    }
    // FIXME: ElaboratedType, DependentNameType,
    // DependentTemplateSpecializationType, ObjCObjectType

    void VisitTypedefDecl(const TypedefDecl *D) {
      Visit(D->getUnderlyingType());
    }

    void VisitEnumConstantDecl(const EnumConstantDecl *D) {
      if (const Expr *Init = D->getInitExpr())
        Visit(Init);
    }

    void VisitFunctionDecl(const FunctionDecl *D) {
      if (const auto *FTSI = D->getTemplateSpecializationInfo())
        dumpTemplateArgumentList(*FTSI->TemplateArguments);

      if (D->param_begin())
        for (const auto *Parameter : D->parameters())
          Visit(Parameter);

      if (const auto *C = dyn_cast<CXXConstructorDecl>(D))
        for (const auto *I : C->inits())
          Visit(I);

      if (D->doesThisDeclarationHaveABody())
        Visit(D->getBody());
    }

    void VisitFieldDecl(const FieldDecl *D) {
      if (D->isBitField())
        Visit(D->getBitWidth());
      if (Expr *Init = D->getInClassInitializer())
        Visit(Init);
    }

    void VisitVarDecl(const VarDecl *D) {
      if (D->hasInit())
        Visit(D->getInit());
    }

    void VisitDecompositionDecl(const DecompositionDecl *D) {
      VisitVarDecl(D);
      for (const auto *B : D->bindings())
        Visit(B);
    }

    void VisitBindingDecl(const BindingDecl *D) {
      if (const auto *E = D->getBinding())
        Visit(E);
    }

    void VisitFileScopeAsmDecl(const FileScopeAsmDecl *D) {
      Visit(D->getAsmString());
    }

    void VisitCapturedDecl(const CapturedDecl *D) { Visit(D->getBody()); }

    void VisitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *D) {
      for (const auto *E : D->varlists())
        Visit(E);
    }

    void VisitOMPDeclareReductionDecl(const OMPDeclareReductionDecl *D) {
      Visit(D->getCombiner());
      if (const auto *Initializer = D->getInitializer())
        Visit(Initializer);
    }

    void VisitOMPDeclareMapperDecl(const OMPDeclareMapperDecl *D) {
      for (const auto *C : D->clauselists())
        Visit(C);
    }

    void VisitOMPCapturedExprDecl(const OMPCapturedExprDecl *D) {
      Visit(D->getInit());
    }

    template <typename SpecializationDecl>
    void dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                        bool DumpExplicitInst,
                                        bool DumpRefOnly);
    template <typename TemplateDecl>
    void dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst);

    void VisitTypeAliasDecl(const TypeAliasDecl *D) {
      Visit(D->getUnderlyingType());
    }

    void VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D) {
      dumpTemplateParameters(D->getTemplateParameters());
      Visit(D->getTemplatedDecl());
    }

    void VisitStaticAssertDecl(const StaticAssertDecl *D) {
      Visit(D->getAssertExpr());
      Visit(D->getMessage());
    }
    void VisitFunctionTemplateDecl(const FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(const ClassTemplateDecl *D);

    void VisitClassTemplateSpecializationDecl(
        const ClassTemplateSpecializationDecl *D) {
      dumpTemplateArgumentList(D->getTemplateArgs());
    }

    void VisitClassTemplatePartialSpecializationDecl(
        const ClassTemplatePartialSpecializationDecl *D) {
      VisitClassTemplateSpecializationDecl(D);
      dumpTemplateParameters(D->getTemplateParameters());
    }

    void VisitClassScopeFunctionSpecializationDecl(
        const ClassScopeFunctionSpecializationDecl *D) {
      Visit(D->getSpecialization());
      if (D->hasExplicitTemplateArgs())
        dumpTemplateArgumentListInfo(D->templateArgs());
    }
    void VisitVarTemplateDecl(const VarTemplateDecl *D);

    void VisitBuiltinTemplateDecl(const BuiltinTemplateDecl *D) {
      dumpTemplateParameters(D->getTemplateParameters());
    }

    void
    VisitVarTemplateSpecializationDecl(const VarTemplateSpecializationDecl *D) {
      dumpTemplateArgumentList(D->getTemplateArgs());
      VisitVarDecl(D);
    }

    void VisitVarTemplatePartialSpecializationDecl(
        const VarTemplatePartialSpecializationDecl *D) {
      dumpTemplateParameters(D->getTemplateParameters());
      VisitVarTemplateSpecializationDecl(D);
    }

    void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D) {
      if (D->hasDefaultArgument())
        Visit(D->getDefaultArgument(), SourceRange(),
              D->getDefaultArgStorage().getInheritedFrom(),
              D->defaultArgumentWasInherited() ? "inherited from" : "previous");
    }

    void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D) {
      if (D->hasDefaultArgument())
        Visit(D->getDefaultArgument(), SourceRange(),
              D->getDefaultArgStorage().getInheritedFrom(),
              D->defaultArgumentWasInherited() ? "inherited from" : "previous");
    }

    void VisitTemplateTemplateParmDecl(const TemplateTemplateParmDecl *D) {
      dumpTemplateParameters(D->getTemplateParameters());
      if (D->hasDefaultArgument())
        dumpTemplateArgumentLoc(
            D->getDefaultArgument(),
            D->getDefaultArgStorage().getInheritedFrom(),
            D->defaultArgumentWasInherited() ? "inherited from" : "previous");
    }

    void VisitUsingShadowDecl(const UsingShadowDecl *D) {
      if (auto *TD = dyn_cast<TypeDecl>(D->getUnderlyingDecl()))
        Visit(TD->getTypeForDecl());
    }

    void VisitFriendDecl(const FriendDecl *D) {
      if (!D->getFriendType())
        Visit(D->getFriendDecl());
    }

    void VisitObjCMethodDecl(const ObjCMethodDecl *D) {
      if (D->isThisDeclarationADefinition())
        dumpDeclContext(D);
      else
        for (const ParmVarDecl *Parameter : D->parameters())
          Visit(Parameter);

      if (D->hasBody())
        Visit(D->getBody());
    }

    void VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
      dumpObjCTypeParamList(D->getTypeParamList());
    }

    void VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
      dumpObjCTypeParamList(D->getTypeParamListAsWritten());
    }

    void VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
      for (const auto &I : D->inits())
        Visit(I);
    }

    void VisitBlockDecl(const BlockDecl *D) {
      for (const auto &I : D->parameters())
        Visit(I);

      for (const auto &I : D->captures())
        Visit(I);
      Visit(D->getBody());
    }

    void VisitDeclStmt(const DeclStmt *Node) {
      for (const auto &D : Node->decls())
        Visit(D);
    }

    void VisitAttributedStmt(const AttributedStmt *Node) {
      for (const auto *A : Node->getAttrs())
        Visit(A);
    }

    void VisitCXXCatchStmt(const CXXCatchStmt *Node) {
      Visit(Node->getExceptionDecl());
    }

    void VisitCapturedStmt(const CapturedStmt *Node) {
      Visit(Node->getCapturedDecl());
    }

    void VisitOMPExecutableDirective(const OMPExecutableDirective *Node) {
      for (const auto *C : Node->clauses())
        Visit(C);
    }

    void VisitInitListExpr(const InitListExpr *ILE) {
      if (auto *Filler = ILE->getArrayFiller()) {
        Visit(Filler, "array_filler");
      }
    }

    void VisitBlockExpr(const BlockExpr *Node) { Visit(Node->getBlockDecl()); }

    void VisitOpaqueValueExpr(const OpaqueValueExpr *Node) {
      if (Expr *Source = Node->getSourceExpr())
        Visit(Source);
    }

    void VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
      Visit(E->getControllingExpr());
      Visit(E->getControllingExpr()->getType()); // FIXME: remove

      for (const auto &Assoc : E->associations()) {
        Visit(Assoc);
      }
    }

    void VisitLambdaExpr(const LambdaExpr *Node) {
      Visit(Node->getLambdaClass());
    }

    void VisitSizeOfPackExpr(const SizeOfPackExpr *Node) {
      if (Node->isPartiallySubstituted())
        for (const auto &A : Node->getPartialArguments())
          Visit(A);
    }

    void VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node) {
      if (const VarDecl *CatchParam = Node->getCatchParamDecl())
        Visit(CatchParam);
    }

    void VisitExpressionTemplateArgument(const TemplateArgument &TA) {
      Visit(TA.getAsExpr());
    }
    void VisitPackTemplateArgument(const TemplateArgument &TA) {
      for (const auto &TArg : TA.pack_elements())
        Visit(TArg);
    }

    // Implements Visit methods for Attrs.
#include "clang/AST/AttrNodeTraverse.inc"
  };
} // namespace

void ASTDumper::dumpDeclContext(const DeclContext *DC) {
  if (!DC)
    return;

  for (const auto *D : (Deserialize ? DC->decls() : DC->noload_decls()))
    Visit(D);
}

void ASTDumper::dumpLookups(const DeclContext *DC, bool DumpDecls) {
  NodeDumper.AddChild([=] {
    OS << "StoredDeclsMap ";
    NodeDumper.dumpBareDeclRef(cast<Decl>(DC));

    const DeclContext *Primary = DC->getPrimaryContext();
    if (Primary != DC) {
      OS << " primary";
      NodeDumper.dumpPointer(cast<Decl>(Primary));
    }

    bool HasUndeserializedLookups = Primary->hasExternalVisibleStorage();

    auto Range = Deserialize
                     ? Primary->lookups()
                     : Primary->noload_lookups(/*PreserveInternalState=*/true);
    for (auto I = Range.begin(), E = Range.end(); I != E; ++I) {
      DeclarationName Name = I.getLookupName();
      DeclContextLookupResult R = *I;

      NodeDumper.AddChild([=] {
        OS << "DeclarationName ";
        {
          ColorScope Color(OS, ShowColors, DeclNameColor);
          OS << '\'' << Name << '\'';
        }

        for (DeclContextLookupResult::iterator RI = R.begin(), RE = R.end();
             RI != RE; ++RI) {
          NodeDumper.AddChild([=] {
            NodeDumper.dumpBareDeclRef(*RI);

            if ((*RI)->isHidden())
              OS << " hidden";

            // If requested, dump the redecl chain for this lookup.
            if (DumpDecls) {
              // Dump earliest decl first.
              std::function<void(Decl *)> DumpWithPrev = [&](Decl *D) {
                if (Decl *Prev = D->getPreviousDecl())
                  DumpWithPrev(Prev);
                Visit(D);
              };
              DumpWithPrev(*RI);
            }
          });
        }
      });
    }

    if (HasUndeserializedLookups) {
      NodeDumper.AddChild([=] {
        ColorScope Color(OS, ShowColors, UndeserializedColor);
        OS << "<undeserialized lookups>";
      });
    }
  });
}

template <typename SpecializationDecl>
void ASTDumper::dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                               bool DumpExplicitInst,
                                               bool DumpRefOnly) {
  bool DumpedAny = false;
  for (const auto *RedeclWithBadType : D->redecls()) {
    // FIXME: The redecls() range sometimes has elements of a less-specific
    // type. (In particular, ClassTemplateSpecializationDecl::redecls() gives
    // us TagDecls, and should give CXXRecordDecls).
    auto *Redecl = dyn_cast<SpecializationDecl>(RedeclWithBadType);
    if (!Redecl) {
      // Found the injected-class-name for a class template. This will be dumped
      // as part of its surrounding class so we don't need to dump it here.
      assert(isa<CXXRecordDecl>(RedeclWithBadType) &&
             "expected an injected-class-name");
      continue;
    }

    switch (Redecl->getTemplateSpecializationKind()) {
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      if (!DumpExplicitInst)
        break;
      LLVM_FALLTHROUGH;
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      if (DumpRefOnly)
        NodeDumper.dumpDeclRef(Redecl);
      else
        Visit(Redecl);
      DumpedAny = true;
      break;
    case TSK_ExplicitSpecialization:
      break;
    }
  }

  // Ensure we dump at least one decl for each specialization.
  if (!DumpedAny)
    NodeDumper.dumpDeclRef(D);
}

template <typename TemplateDecl>
void ASTDumper::dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst) {
  dumpTemplateParameters(D->getTemplateParameters());

  Visit(D->getTemplatedDecl());

  for (const auto *Child : D->specializations())
    dumpTemplateDeclSpecialization(Child, DumpExplicitInst,
                                   !D->isCanonicalDecl());
}

void ASTDumper::VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
  // FIXME: We don't add a declaration of a function template specialization
  // to its context when it's explicitly instantiated, so dump explicit
  // instantiations when we dump the template itself.
  dumpTemplateDecl(D, true);
}

void ASTDumper::VisitClassTemplateDecl(const ClassTemplateDecl *D) {
  dumpTemplateDecl(D, false);
}

void ASTDumper::VisitVarTemplateDecl(const VarTemplateDecl *D) {
  dumpTemplateDecl(D, false);
}

//===----------------------------------------------------------------------===//
// Type method implementations
//===----------------------------------------------------------------------===//

void QualType::dump(const char *msg) const {
  if (msg)
    llvm::errs() << msg << ": ";
  dump();
}

LLVM_DUMP_METHOD void QualType::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void QualType::dump(llvm::raw_ostream &OS) const {
  ASTDumper Dumper(OS, nullptr, nullptr);
  Dumper.Visit(*this);
}

LLVM_DUMP_METHOD void Type::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Type::dump(llvm::raw_ostream &OS) const {
  QualType(this, 0).dump(OS);
}

//===----------------------------------------------------------------------===//
// Decl method implementations
//===----------------------------------------------------------------------===//

LLVM_DUMP_METHOD void Decl::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Decl::dump(raw_ostream &OS, bool Deserialize) const {
  const ASTContext &Ctx = getASTContext();
  const SourceManager &SM = Ctx.getSourceManager();
  ASTDumper P(OS, &Ctx.getCommentCommandTraits(), &SM,
              SM.getDiagnostics().getShowColors(), Ctx.getPrintingPolicy());
  P.setDeserialize(Deserialize);
  P.Visit(this);
}

LLVM_DUMP_METHOD void Decl::dumpColor() const {
  const ASTContext &Ctx = getASTContext();
  ASTDumper P(llvm::errs(), &Ctx.getCommentCommandTraits(),
              &Ctx.getSourceManager(), /*ShowColors*/ true,
              Ctx.getPrintingPolicy());
  P.Visit(this);
}

LLVM_DUMP_METHOD void DeclContext::dumpLookups() const {
  dumpLookups(llvm::errs());
}

LLVM_DUMP_METHOD void DeclContext::dumpLookups(raw_ostream &OS,
                                               bool DumpDecls,
                                               bool Deserialize) const {
  const DeclContext *DC = this;
  while (!DC->isTranslationUnit())
    DC = DC->getParent();
  ASTContext &Ctx = cast<TranslationUnitDecl>(DC)->getASTContext();
  const SourceManager &SM = Ctx.getSourceManager();
  ASTDumper P(OS, &Ctx.getCommentCommandTraits(), &Ctx.getSourceManager(),
              SM.getDiagnostics().getShowColors(), Ctx.getPrintingPolicy());
  P.setDeserialize(Deserialize);
  P.dumpLookups(this, DumpDecls);
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

LLVM_DUMP_METHOD void Stmt::dump(SourceManager &SM) const {
  dump(llvm::errs(), SM);
}

LLVM_DUMP_METHOD void Stmt::dump(raw_ostream &OS, SourceManager &SM) const {
  ASTDumper P(OS, nullptr, &SM);
  P.Visit(this);
}

LLVM_DUMP_METHOD void Stmt::dump(raw_ostream &OS) const {
  ASTDumper P(OS, nullptr, nullptr);
  P.Visit(this);
}

LLVM_DUMP_METHOD void Stmt::dump() const {
  ASTDumper P(llvm::errs(), nullptr, nullptr);
  P.Visit(this);
}

LLVM_DUMP_METHOD void Stmt::dumpColor() const {
  ASTDumper P(llvm::errs(), nullptr, nullptr, /*ShowColors*/true);
  P.Visit(this);
}

//===----------------------------------------------------------------------===//
// Comment method implementations
//===----------------------------------------------------------------------===//

LLVM_DUMP_METHOD void Comment::dump() const {
  dump(llvm::errs(), nullptr, nullptr);
}

LLVM_DUMP_METHOD void Comment::dump(const ASTContext &Context) const {
  dump(llvm::errs(), &Context.getCommentCommandTraits(),
       &Context.getSourceManager());
}

void Comment::dump(raw_ostream &OS, const CommandTraits *Traits,
                   const SourceManager *SM) const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  if (!FC)
    return;
  ASTDumper D(OS, Traits, SM);
  D.Visit(FC, FC);
}

LLVM_DUMP_METHOD void Comment::dumpColor() const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  if (!FC)
    return;
  ASTDumper D(llvm::errs(), nullptr, nullptr, /*ShowColors*/true);
  D.Visit(FC, FC);
}
