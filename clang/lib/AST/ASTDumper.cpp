//===--- ASTDumper.cpp - Dumping implementation for ASTs ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/LocInfoType.h"
#include "clang/AST/StmtVisitor.h"
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
        public TypeVisitor<ASTDumper> {

    TextTreeStructure TreeStructure;
    TextNodeDumper NodeDumper;

    raw_ostream &OS;

    /// The policy to use for printing; can be defaulted.
    PrintingPolicy PrintPolicy;

    /// Indicates whether we should trigger deserialization of nodes that had
    /// not already been loaded.
    bool Deserialize = false;

    const bool ShowColors;

    /// Dump a child of the current node.
    template<typename Fn> void dumpChild(Fn doDumpChild) {
      TreeStructure.addChild(doDumpChild);
    }

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
        : TreeStructure(OS, ShowColors),
          NodeDumper(OS, ShowColors, SM, PrintPolicy, Traits), OS(OS),
          PrintPolicy(PrintPolicy), ShowColors(ShowColors) {}

    void setDeserialize(bool D) { Deserialize = D; }

    void dumpDecl(const Decl *D);
    void dumpStmt(const Stmt *S);

    // Utilities
    void dumpType(QualType T) { NodeDumper.dumpType(T); }
    void dumpTypeAsChild(QualType T);
    void dumpTypeAsChild(const Type *T);
    void dumpDeclRef(const Decl *Node, const char *Label = nullptr);
    void dumpBareDeclRef(const Decl *Node) { NodeDumper.dumpBareDeclRef(Node); }
    void dumpDeclContext(const DeclContext *DC);
    void dumpLookups(const DeclContext *DC, bool DumpDecls);
    void dumpAttr(const Attr *A);

    // C++ Utilities
    void dumpCXXCtorInitializer(const CXXCtorInitializer *Init);
    void dumpTemplateParameters(const TemplateParameterList *TPL);
    void dumpTemplateArgumentListInfo(const TemplateArgumentListInfo &TALI);
    void dumpTemplateArgumentLoc(const TemplateArgumentLoc &A,
                                 const Decl *From = nullptr,
                                 const char *Label = nullptr);
    void dumpTemplateArgumentList(const TemplateArgumentList &TAL);
    void dumpTemplateArgument(const TemplateArgument &A,
                              SourceRange R = SourceRange(),
                              const Decl *From = nullptr,
                              const char *Label = nullptr);
    template <typename SpecializationDecl>
    void dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                        bool DumpExplicitInst,
                                        bool DumpRefOnly);
    template <typename TemplateDecl>
    void dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst);

    // Objective-C utilities.
    void dumpObjCTypeParamList(const ObjCTypeParamList *typeParams);

    // Types
    void VisitComplexType(const ComplexType *T) {
      dumpTypeAsChild(T->getElementType());
    }
    void VisitPointerType(const PointerType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitBlockPointerType(const BlockPointerType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitReferenceType(const ReferenceType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitRValueReferenceType(const ReferenceType *T) {
      if (T->isSpelledAsLValue())
        OS << " written as lvalue reference";
      VisitReferenceType(T);
    }
    void VisitMemberPointerType(const MemberPointerType *T) {
      dumpTypeAsChild(T->getClass());
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitArrayType(const ArrayType *T) {
      switch (T->getSizeModifier()) {
        case ArrayType::Normal: break;
        case ArrayType::Static: OS << " static"; break;
        case ArrayType::Star: OS << " *"; break;
      }
      OS << " " << T->getIndexTypeQualifiers().getAsString();
      dumpTypeAsChild(T->getElementType());
    }
    void VisitConstantArrayType(const ConstantArrayType *T) {
      OS << " " << T->getSize();
      VisitArrayType(T);
    }
    void VisitVariableArrayType(const VariableArrayType *T) {
      OS << " ";
      NodeDumper.dumpSourceRange(T->getBracketsRange());
      VisitArrayType(T);
      dumpStmt(T->getSizeExpr());
    }
    void VisitDependentSizedArrayType(const DependentSizedArrayType *T) {
      switch (T->getSizeModifier()) {
        case ArrayType::Normal: break;
        case ArrayType::Static: OS << " static"; break;
        case ArrayType::Star: OS << " *"; break;
      }
      OS << " " << T->getIndexTypeQualifiers().getAsString();
      OS << " ";
      NodeDumper.dumpSourceRange(T->getBracketsRange());
      dumpTypeAsChild(T->getElementType());
      dumpStmt(T->getSizeExpr());
    }
    void VisitDependentSizedExtVectorType(
        const DependentSizedExtVectorType *T) {
      OS << " ";
      NodeDumper.dumpLocation(T->getAttributeLoc());
      dumpTypeAsChild(T->getElementType());
      dumpStmt(T->getSizeExpr());
    }
    void VisitVectorType(const VectorType *T) {
      switch (T->getVectorKind()) {
        case VectorType::GenericVector: break;
        case VectorType::AltiVecVector: OS << " altivec"; break;
        case VectorType::AltiVecPixel: OS << " altivec pixel"; break;
        case VectorType::AltiVecBool: OS << " altivec bool"; break;
        case VectorType::NeonVector: OS << " neon"; break;
        case VectorType::NeonPolyVector: OS << " neon poly"; break;
      }
      OS << " " << T->getNumElements();
      dumpTypeAsChild(T->getElementType());
    }
    void VisitFunctionType(const FunctionType *T) {
      auto EI = T->getExtInfo();
      if (EI.getNoReturn()) OS << " noreturn";
      if (EI.getProducesResult()) OS << " produces_result";
      if (EI.getHasRegParm()) OS << " regparm " << EI.getRegParm();
      OS << " " << FunctionType::getNameForCallConv(EI.getCC());
      dumpTypeAsChild(T->getReturnType());
    }
    void VisitFunctionProtoType(const FunctionProtoType *T) {
      auto EPI = T->getExtProtoInfo();
      if (EPI.HasTrailingReturn) OS << " trailing_return";

      if (!T->getTypeQuals().empty())
        OS << " " << T->getTypeQuals().getAsString();

      switch (EPI.RefQualifier) {
        case RQ_None: break;
        case RQ_LValue: OS << " &"; break;
        case RQ_RValue: OS << " &&"; break;
      }
      // FIXME: Exception specification.
      // FIXME: Consumed parameters.
      VisitFunctionType(T);
      for (QualType PT : T->getParamTypes())
        dumpTypeAsChild(PT);
      if (EPI.Variadic)
        dumpChild([=] { OS << "..."; });
    }
    void VisitUnresolvedUsingType(const UnresolvedUsingType *T) {
      dumpDeclRef(T->getDecl());
    }
    void VisitTypedefType(const TypedefType *T) {
      dumpDeclRef(T->getDecl());
    }
    void VisitTypeOfExprType(const TypeOfExprType *T) {
      dumpStmt(T->getUnderlyingExpr());
    }
    void VisitDecltypeType(const DecltypeType *T) {
      dumpStmt(T->getUnderlyingExpr());
    }
    void VisitUnaryTransformType(const UnaryTransformType *T) {
      switch (T->getUTTKind()) {
      case UnaryTransformType::EnumUnderlyingType:
        OS << " underlying_type";
        break;
      }
      dumpTypeAsChild(T->getBaseType());
    }
    void VisitTagType(const TagType *T) {
      dumpDeclRef(T->getDecl());
    }
    void VisitAttributedType(const AttributedType *T) {
      // FIXME: AttrKind
      dumpTypeAsChild(T->getModifiedType());
    }
    void VisitTemplateTypeParmType(const TemplateTypeParmType *T) {
      OS << " depth " << T->getDepth() << " index " << T->getIndex();
      if (T->isParameterPack()) OS << " pack";
      dumpDeclRef(T->getDecl());
    }
    void VisitSubstTemplateTypeParmType(const SubstTemplateTypeParmType *T) {
      dumpTypeAsChild(T->getReplacedParameter());
    }
    void VisitSubstTemplateTypeParmPackType(
        const SubstTemplateTypeParmPackType *T) {
      dumpTypeAsChild(T->getReplacedParameter());
      dumpTemplateArgument(T->getArgumentPack());
    }
    void VisitAutoType(const AutoType *T) {
      if (T->isDecltypeAuto()) OS << " decltype(auto)";
      if (!T->isDeduced())
        OS << " undeduced";
    }
    void VisitTemplateSpecializationType(const TemplateSpecializationType *T) {
      if (T->isTypeAlias()) OS << " alias";
      OS << " "; T->getTemplateName().dump(OS);
      for (auto &Arg : *T)
        dumpTemplateArgument(Arg);
      if (T->isTypeAlias())
        dumpTypeAsChild(T->getAliasedType());
    }
    void VisitInjectedClassNameType(const InjectedClassNameType *T) {
      dumpDeclRef(T->getDecl());
    }
    void VisitObjCInterfaceType(const ObjCInterfaceType *T) {
      dumpDeclRef(T->getDecl());
    }
    void VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
      dumpTypeAsChild(T->getPointeeType());
    }
    void VisitAtomicType(const AtomicType *T) {
      dumpTypeAsChild(T->getValueType());
    }
    void VisitPipeType(const PipeType *T) {
      dumpTypeAsChild(T->getElementType());
    }
    void VisitAdjustedType(const AdjustedType *T) {
      dumpTypeAsChild(T->getOriginalType());
    }
    void VisitPackExpansionType(const PackExpansionType *T) {
      if (auto N = T->getNumExpansions()) OS << " expansions " << *N;
      if (!T->isSugared())
        dumpTypeAsChild(T->getPattern());
    }
    // FIXME: ElaboratedType, DependentNameType,
    // DependentTemplateSpecializationType, ObjCObjectType

    // Decls
    void VisitLabelDecl(const LabelDecl *D);
    void VisitTypedefDecl(const TypedefDecl *D);
    void VisitEnumDecl(const EnumDecl *D);
    void VisitRecordDecl(const RecordDecl *D);
    void VisitEnumConstantDecl(const EnumConstantDecl *D);
    void VisitIndirectFieldDecl(const IndirectFieldDecl *D);
    void VisitFunctionDecl(const FunctionDecl *D);
    void VisitFieldDecl(const FieldDecl *D);
    void VisitVarDecl(const VarDecl *D);
    void VisitDecompositionDecl(const DecompositionDecl *D);
    void VisitBindingDecl(const BindingDecl *D);
    void VisitFileScopeAsmDecl(const FileScopeAsmDecl *D);
    void VisitImportDecl(const ImportDecl *D);
    void VisitPragmaCommentDecl(const PragmaCommentDecl *D);
    void VisitPragmaDetectMismatchDecl(const PragmaDetectMismatchDecl *D);
    void VisitCapturedDecl(const CapturedDecl *D);

    // OpenMP decls
    void VisitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *D);
    void VisitOMPDeclareReductionDecl(const OMPDeclareReductionDecl *D);
    void VisitOMPRequiresDecl(const OMPRequiresDecl *D);
    void VisitOMPCapturedExprDecl(const OMPCapturedExprDecl *D);

    // C++ Decls
    void VisitNamespaceDecl(const NamespaceDecl *D);
    void VisitUsingDirectiveDecl(const UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(const NamespaceAliasDecl *D);
    void VisitTypeAliasDecl(const TypeAliasDecl *D);
    void VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D);
    void VisitCXXRecordDecl(const CXXRecordDecl *D);
    void VisitStaticAssertDecl(const StaticAssertDecl *D);
    void VisitFunctionTemplateDecl(const FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(const ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(
        const ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
        const ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
        const ClassScopeFunctionSpecializationDecl *D);
    void VisitBuiltinTemplateDecl(const BuiltinTemplateDecl *D);
    void VisitVarTemplateDecl(const VarTemplateDecl *D);
    void VisitVarTemplateSpecializationDecl(
        const VarTemplateSpecializationDecl *D);
    void VisitVarTemplatePartialSpecializationDecl(
        const VarTemplatePartialSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D);
    void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D);
    void VisitTemplateTemplateParmDecl(const TemplateTemplateParmDecl *D);
    void VisitUsingDecl(const UsingDecl *D);
    void VisitUnresolvedUsingTypenameDecl(const UnresolvedUsingTypenameDecl *D);
    void VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D);
    void VisitUsingShadowDecl(const UsingShadowDecl *D);
    void VisitConstructorUsingShadowDecl(const ConstructorUsingShadowDecl *D);
    void VisitLinkageSpecDecl(const LinkageSpecDecl *D);
    void VisitAccessSpecDecl(const AccessSpecDecl *D);
    void VisitFriendDecl(const FriendDecl *D);

    // ObjC Decls
    void VisitObjCIvarDecl(const ObjCIvarDecl *D);
    void VisitObjCMethodDecl(const ObjCMethodDecl *D);
    void VisitObjCTypeParamDecl(const ObjCTypeParamDecl *D);
    void VisitObjCCategoryDecl(const ObjCCategoryDecl *D);
    void VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D);
    void VisitObjCProtocolDecl(const ObjCProtocolDecl *D);
    void VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D);
    void VisitObjCImplementationDecl(const ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(const ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D);
    void VisitBlockDecl(const BlockDecl *D);

    // Stmts.
    void VisitDeclStmt(const DeclStmt *Node);
    void VisitAttributedStmt(const AttributedStmt *Node);
    void VisitIfStmt(const IfStmt *Node);
    void VisitSwitchStmt(const SwitchStmt *Node);
    void VisitWhileStmt(const WhileStmt *Node);
    void VisitLabelStmt(const LabelStmt *Node);
    void VisitGotoStmt(const GotoStmt *Node);
    void VisitCXXCatchStmt(const CXXCatchStmt *Node);
    void VisitCaseStmt(const CaseStmt *Node);
    void VisitCapturedStmt(const CapturedStmt *Node);

    // OpenMP
    void VisitOMPExecutableDirective(const OMPExecutableDirective *Node);

    // Exprs
    void VisitCallExpr(const CallExpr *Node);
    void VisitCastExpr(const CastExpr *Node);
    void VisitImplicitCastExpr(const ImplicitCastExpr *Node);
    void VisitDeclRefExpr(const DeclRefExpr *Node);
    void VisitPredefinedExpr(const PredefinedExpr *Node);
    void VisitCharacterLiteral(const CharacterLiteral *Node);
    void VisitIntegerLiteral(const IntegerLiteral *Node);
    void VisitFixedPointLiteral(const FixedPointLiteral *Node);
    void VisitFloatingLiteral(const FloatingLiteral *Node);
    void VisitStringLiteral(const StringLiteral *Str);
    void VisitInitListExpr(const InitListExpr *ILE);
    void VisitUnaryOperator(const UnaryOperator *Node);
    void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Node);
    void VisitMemberExpr(const MemberExpr *Node);
    void VisitExtVectorElementExpr(const ExtVectorElementExpr *Node);
    void VisitBinaryOperator(const BinaryOperator *Node);
    void VisitCompoundAssignOperator(const CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(const AddrLabelExpr *Node);
    void VisitBlockExpr(const BlockExpr *Node);
    void VisitOpaqueValueExpr(const OpaqueValueExpr *Node);
    void VisitGenericSelectionExpr(const GenericSelectionExpr *E);

    // C++
    void VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(const CXXThisExpr *Node);
    void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *Node);
    void VisitCXXUnresolvedConstructExpr(const CXXUnresolvedConstructExpr *Node);
    void VisitCXXConstructExpr(const CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Node);
    void VisitCXXNewExpr(const CXXNewExpr *Node);
    void VisitCXXDeleteExpr(const CXXDeleteExpr *Node);
    void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Node);
    void VisitExprWithCleanups(const ExprWithCleanups *Node);
    void VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *Node);
    void VisitLambdaExpr(const LambdaExpr *Node) {
      dumpDecl(Node->getLambdaClass());
    }
    void VisitSizeOfPackExpr(const SizeOfPackExpr *Node);
    void
    VisitCXXDependentScopeMemberExpr(const CXXDependentScopeMemberExpr *Node);

    // ObjC
    void VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node);
    void VisitObjCEncodeExpr(const ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(const ObjCMessageExpr *Node);
    void VisitObjCBoxedExpr(const ObjCBoxedExpr *Node);
    void VisitObjCSelectorExpr(const ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(const ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Node);
    void VisitObjCSubscriptRefExpr(const ObjCSubscriptRefExpr *Node);
    void VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Node);
    void VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *Node);

    // Comments.
    void dumpComment(const Comment *C, const FullComment *FC);
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

void ASTDumper::dumpTypeAsChild(QualType T) {
  SplitQualType SQT = T.split();
  if (!SQT.Quals.hasQualifiers())
    return dumpTypeAsChild(SQT.Ty);

  dumpChild([=] {
    OS << "QualType";
    NodeDumper.dumpPointer(T.getAsOpaquePtr());
    OS << " ";
    NodeDumper.dumpBareType(T, false);
    OS << " " << T.split().Quals.getAsString();
    dumpTypeAsChild(T.split().Ty);
  });
}

void ASTDumper::dumpTypeAsChild(const Type *T) {
  dumpChild([=] {
    if (!T) {
      ColorScope Color(OS, ShowColors, NullColor);
      OS << "<<<NULL>>>";
      return;
    }
    if (const LocInfoType *LIT = llvm::dyn_cast<LocInfoType>(T)) {
      {
        ColorScope Color(OS, ShowColors, TypeColor);
        OS << "LocInfo Type";
      }
      NodeDumper.dumpPointer(T);
      dumpTypeAsChild(LIT->getTypeSourceInfo()->getType());
      return;
    }

    {
      ColorScope Color(OS, ShowColors, TypeColor);
      OS << T->getTypeClassName() << "Type";
    }
    NodeDumper.dumpPointer(T);
    OS << " ";
    NodeDumper.dumpBareType(QualType(T, 0), false);

    QualType SingleStepDesugar =
        T->getLocallyUnqualifiedSingleStepDesugaredType();
    if (SingleStepDesugar != QualType(T, 0))
      OS << " sugar";
    if (T->isDependentType())
      OS << " dependent";
    else if (T->isInstantiationDependentType())
      OS << " instantiation_dependent";
    if (T->isVariablyModifiedType())
      OS << " variably_modified";
    if (T->containsUnexpandedParameterPack())
      OS << " contains_unexpanded_pack";
    if (T->isFromAST())
      OS << " imported";

    TypeVisitor<ASTDumper>::Visit(T);

    if (SingleStepDesugar != QualType(T, 0))
      dumpTypeAsChild(SingleStepDesugar);
  });
}

void ASTDumper::dumpDeclRef(const Decl *D, const char *Label) {
  if (!D)
    return;

  dumpChild([=]{
    if (Label)
      OS << Label << ' ';
    dumpBareDeclRef(D);
  });
}

void ASTDumper::dumpDeclContext(const DeclContext *DC) {
  if (!DC)
    return;

  for (auto *D : (Deserialize ? DC->decls() : DC->noload_decls()))
    dumpDecl(D);

  if (DC->hasExternalLexicalStorage()) {
    dumpChild([=] {
      ColorScope Color(OS, ShowColors, UndeserializedColor);
      OS << "<undeserialized declarations>";
    });
  }
}

void ASTDumper::dumpLookups(const DeclContext *DC, bool DumpDecls) {
  dumpChild([=] {
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

      dumpChild([=] {
        OS << "DeclarationName ";
        {
          ColorScope Color(OS, ShowColors, DeclNameColor);
          OS << '\'' << Name << '\'';
        }

        for (DeclContextLookupResult::iterator RI = R.begin(), RE = R.end();
             RI != RE; ++RI) {
          dumpChild([=] {
            NodeDumper.dumpBareDeclRef(*RI);

            if ((*RI)->isHidden())
              OS << " hidden";

            // If requested, dump the redecl chain for this lookup.
            if (DumpDecls) {
              // Dump earliest decl first.
              std::function<void(Decl *)> DumpWithPrev = [&](Decl *D) {
                if (Decl *Prev = D->getPreviousDecl())
                  DumpWithPrev(Prev);
                dumpDecl(D);
              };
              DumpWithPrev(*RI);
            }
          });
        }
      });
    }

    if (HasUndeserializedLookups) {
      dumpChild([=] {
        ColorScope Color(OS, ShowColors, UndeserializedColor);
        OS << "<undeserialized lookups>";
      });
    }
  });
}

void ASTDumper::dumpAttr(const Attr *A) {
  dumpChild([=] {
    {
      ColorScope Color(OS, ShowColors, AttrColor);

      switch (A->getKind()) {
#define ATTR(X) case attr::X: OS << #X; break;
#include "clang/Basic/AttrList.inc"
      }
      OS << "Attr";
    }
    NodeDumper.dumpPointer(A);
    NodeDumper.dumpSourceRange(A->getRange());
    if (A->isInherited())
      OS << " Inherited";
    if (A->isImplicit())
      OS << " Implicit";
#include "clang/AST/AttrDump.inc"
  });
}

static void dumpPreviousDeclImpl(raw_ostream &OS, ...) {}

template<typename T>
static void dumpPreviousDeclImpl(raw_ostream &OS, const Mergeable<T> *D) {
  const T *First = D->getFirstDecl();
  if (First != D)
    OS << " first " << First;
}

template<typename T>
static void dumpPreviousDeclImpl(raw_ostream &OS, const Redeclarable<T> *D) {
  const T *Prev = D->getPreviousDecl();
  if (Prev)
    OS << " prev " << Prev;
}

/// Dump the previous declaration in the redeclaration chain for a declaration,
/// if any.
static void dumpPreviousDecl(raw_ostream &OS, const Decl *D) {
  switch (D->getKind()) {
#define DECL(DERIVED, BASE) \
  case Decl::DERIVED: \
    return dumpPreviousDeclImpl(OS, cast<DERIVED##Decl>(D));
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
  }
  llvm_unreachable("Decl that isn't part of DeclNodes.inc!");
}

//===----------------------------------------------------------------------===//
//  C++ Utilities
//===----------------------------------------------------------------------===//

void ASTDumper::dumpCXXCtorInitializer(const CXXCtorInitializer *Init) {
  dumpChild([=] {
    OS << "CXXCtorInitializer";
    if (Init->isAnyMemberInitializer()) {
      OS << ' ';
      NodeDumper.dumpBareDeclRef(Init->getAnyMember());
    } else if (Init->isBaseInitializer()) {
      NodeDumper.dumpType(QualType(Init->getBaseClass(), 0));
    } else if (Init->isDelegatingInitializer()) {
      NodeDumper.dumpType(Init->getTypeSourceInfo()->getType());
    } else {
      llvm_unreachable("Unknown initializer type");
    }
    dumpStmt(Init->getInit());
  });
}

void ASTDumper::dumpTemplateParameters(const TemplateParameterList *TPL) {
  if (!TPL)
    return;

  for (TemplateParameterList::const_iterator I = TPL->begin(), E = TPL->end();
       I != E; ++I)
    dumpDecl(*I);
}

void ASTDumper::dumpTemplateArgumentListInfo(
    const TemplateArgumentListInfo &TALI) {
  for (unsigned i = 0, e = TALI.size(); i < e; ++i)
    dumpTemplateArgumentLoc(TALI[i]);
}

void ASTDumper::dumpTemplateArgumentLoc(const TemplateArgumentLoc &A,
                                        const Decl *From, const char *Label) {
  dumpTemplateArgument(A.getArgument(), A.getSourceRange(), From, Label);
}

void ASTDumper::dumpTemplateArgumentList(const TemplateArgumentList &TAL) {
  for (unsigned i = 0, e = TAL.size(); i < e; ++i)
    dumpTemplateArgument(TAL[i]);
}

void ASTDumper::dumpTemplateArgument(const TemplateArgument &A, SourceRange R,
                                     const Decl *From, const char *Label) {
  dumpChild([=] {
    OS << "TemplateArgument";
    if (R.isValid())
      NodeDumper.dumpSourceRange(R);

    if (From)
      dumpDeclRef(From, Label);

    switch (A.getKind()) {
    case TemplateArgument::Null:
      OS << " null";
      break;
    case TemplateArgument::Type:
      OS << " type";
      NodeDumper.dumpType(A.getAsType());
      break;
    case TemplateArgument::Declaration:
      OS << " decl";
      dumpDeclRef(A.getAsDecl());
      break;
    case TemplateArgument::NullPtr:
      OS << " nullptr";
      break;
    case TemplateArgument::Integral:
      OS << " integral " << A.getAsIntegral();
      break;
    case TemplateArgument::Template:
      OS << " template ";
      A.getAsTemplate().dump(OS);
      break;
    case TemplateArgument::TemplateExpansion:
      OS << " template expansion ";
      A.getAsTemplateOrTemplatePattern().dump(OS);
      break;
    case TemplateArgument::Expression:
      OS << " expr";
      dumpStmt(A.getAsExpr());
      break;
    case TemplateArgument::Pack:
      OS << " pack";
      for (TemplateArgument::pack_iterator I = A.pack_begin(), E = A.pack_end();
           I != E; ++I)
        dumpTemplateArgument(*I);
      break;
    }
  });
}

//===----------------------------------------------------------------------===//
//  Objective-C Utilities
//===----------------------------------------------------------------------===//
void ASTDumper::dumpObjCTypeParamList(const ObjCTypeParamList *typeParams) {
  if (!typeParams)
    return;

  for (auto typeParam : *typeParams) {
    dumpDecl(typeParam);
  }
}

//===----------------------------------------------------------------------===//
//  Decl dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpDecl(const Decl *D) {
  dumpChild([=] {
    if (!D) {
      ColorScope Color(OS, ShowColors, NullColor);
      OS << "<<<NULL>>>";
      return;
    }

    {
      ColorScope Color(OS, ShowColors, DeclKindNameColor);
      OS << D->getDeclKindName() << "Decl";
    }
    NodeDumper.dumpPointer(D);
    if (D->getLexicalDeclContext() != D->getDeclContext())
      OS << " parent " << cast<Decl>(D->getDeclContext());
    dumpPreviousDecl(OS, D);
    NodeDumper.dumpSourceRange(D->getSourceRange());
    OS << ' ';
    NodeDumper.dumpLocation(D->getLocation());
    if (D->isFromASTFile())
      OS << " imported";
    if (Module *M = D->getOwningModule())
      OS << " in " << M->getFullModuleName();
    if (auto *ND = dyn_cast<NamedDecl>(D))
      for (Module *M : D->getASTContext().getModulesWithMergedDefinition(
               const_cast<NamedDecl *>(ND)))
        dumpChild([=] { OS << "also in " << M->getFullModuleName(); });
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
      if (ND->isHidden())
        OS << " hidden";
    if (D->isImplicit())
      OS << " implicit";
    if (D->isUsed())
      OS << " used";
    else if (D->isThisDeclarationReferenced())
      OS << " referenced";
    if (D->isInvalidDecl())
      OS << " invalid";
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      if (FD->isConstexpr())
        OS << " constexpr";


    ConstDeclVisitor<ASTDumper>::Visit(D);

    for (Decl::attr_iterator I = D->attr_begin(), E = D->attr_end(); I != E;
         ++I)
      dumpAttr(*I);

    if (const FullComment *Comment =
            D->getASTContext().getLocalCommentForDeclUncached(D))
      dumpComment(Comment, Comment);

    // Decls within functions are visited by the body.
    if (!isa<FunctionDecl>(*D) && !isa<ObjCMethodDecl>(*D)) {
      auto DC = dyn_cast<DeclContext>(D);
      if (DC &&
          (DC->hasExternalLexicalStorage() ||
           (Deserialize ? DC->decls_begin() != DC->decls_end()
                        : DC->noload_decls_begin() != DC->noload_decls_end())))
        dumpDeclContext(DC);
    }
  });
}

void ASTDumper::VisitLabelDecl(const LabelDecl *D) { NodeDumper.dumpName(D); }

void ASTDumper::VisitTypedefDecl(const TypedefDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getUnderlyingType());
  if (D->isModulePrivate())
    OS << " __module_private__";
  dumpTypeAsChild(D->getUnderlyingType());
}

void ASTDumper::VisitEnumDecl(const EnumDecl *D) {
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      OS << " class";
    else
      OS << " struct";
  }
  NodeDumper.dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isFixed())
    NodeDumper.dumpType(D->getIntegerType());
}

void ASTDumper::VisitRecordDecl(const RecordDecl *D) {
  OS << ' ' << D->getKindName();
  NodeDumper.dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isCompleteDefinition())
    OS << " definition";
}

void ASTDumper::VisitEnumConstantDecl(const EnumConstantDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  if (const Expr *Init = D->getInitExpr())
    dumpStmt(Init);
}

void ASTDumper::VisitIndirectFieldDecl(const IndirectFieldDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());

  for (auto *Child : D->chain())
    dumpDeclRef(Child);
}

void ASTDumper::VisitFunctionDecl(const FunctionDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());

  StorageClass SC = D->getStorageClass();
  if (SC != SC_None)
    OS << ' ' << VarDecl::getStorageClassSpecifierString(SC);
  if (D->isInlineSpecified())
    OS << " inline";
  if (D->isVirtualAsWritten())
    OS << " virtual";
  if (D->isModulePrivate())
    OS << " __module_private__";

  if (D->isPure())
    OS << " pure";
  if (D->isDefaulted()) {
    OS << " default";
    if (D->isDeleted())
      OS << "_delete";
  }
  if (D->isDeletedAsWritten())
    OS << " delete";
  if (D->isTrivial())
    OS << " trivial";

  if (const FunctionProtoType *FPT = D->getType()->getAs<FunctionProtoType>()) {
    FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
    switch (EPI.ExceptionSpec.Type) {
    default: break;
    case EST_Unevaluated:
      OS << " noexcept-unevaluated " << EPI.ExceptionSpec.SourceDecl;
      break;
    case EST_Uninstantiated:
      OS << " noexcept-uninstantiated " << EPI.ExceptionSpec.SourceTemplate;
      break;
    }
  }

  if (const FunctionTemplateSpecializationInfo *FTSI =
          D->getTemplateSpecializationInfo())
    dumpTemplateArgumentList(*FTSI->TemplateArguments);

  if (!D->param_begin() && D->getNumParams())
    dumpChild([=] { OS << "<<NULL params x " << D->getNumParams() << ">>"; });
  else
    for (const ParmVarDecl *Parameter : D->parameters())
      dumpDecl(Parameter);

  if (const CXXConstructorDecl *C = dyn_cast<CXXConstructorDecl>(D))
    for (CXXConstructorDecl::init_const_iterator I = C->init_begin(),
                                                 E = C->init_end();
         I != E; ++I)
      dumpCXXCtorInitializer(*I);

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (MD->size_overridden_methods() != 0) {
      auto dumpOverride = [=](const CXXMethodDecl *D) {
        SplitQualType T_split = D->getType().split();
        OS << D << " " << D->getParent()->getName()
           << "::" << D->getNameAsString() << " '"
           << QualType::getAsString(T_split, PrintPolicy) << "'";
      };

      dumpChild([=] {
        auto Overrides = MD->overridden_methods();
        OS << "Overrides: [ ";
        dumpOverride(*Overrides.begin());
        for (const auto *Override :
             llvm::make_range(Overrides.begin() + 1, Overrides.end())) {
          OS << ", ";
          dumpOverride(Override);
        }
        OS << " ]";
      });
    }
  }

  if (D->doesThisDeclarationHaveABody())
    dumpStmt(D->getBody());
}

void ASTDumper::VisitFieldDecl(const FieldDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  if (D->isMutable())
    OS << " mutable";
  if (D->isModulePrivate())
    OS << " __module_private__";

  if (D->isBitField())
    dumpStmt(D->getBitWidth());
  if (Expr *Init = D->getInClassInitializer())
    dumpStmt(Init);
}

void ASTDumper::VisitVarDecl(const VarDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  StorageClass SC = D->getStorageClass();
  if (SC != SC_None)
    OS << ' ' << VarDecl::getStorageClassSpecifierString(SC);
  switch (D->getTLSKind()) {
  case VarDecl::TLS_None: break;
  case VarDecl::TLS_Static: OS << " tls"; break;
  case VarDecl::TLS_Dynamic: OS << " tls_dynamic"; break;
  }
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isNRVOVariable())
    OS << " nrvo";
  if (D->isInline())
    OS << " inline";
  if (D->isConstexpr())
    OS << " constexpr";
  if (D->hasInit()) {
    switch (D->getInitStyle()) {
    case VarDecl::CInit: OS << " cinit"; break;
    case VarDecl::CallInit: OS << " callinit"; break;
    case VarDecl::ListInit: OS << " listinit"; break;
    }
    dumpStmt(D->getInit());
  }
}

void ASTDumper::VisitDecompositionDecl(const DecompositionDecl *D) {
  VisitVarDecl(D);
  for (auto *B : D->bindings())
    dumpDecl(B);
}

void ASTDumper::VisitBindingDecl(const BindingDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  if (auto *E = D->getBinding())
    dumpStmt(E);
}

void ASTDumper::VisitFileScopeAsmDecl(const FileScopeAsmDecl *D) {
  dumpStmt(D->getAsmString());
}

void ASTDumper::VisitImportDecl(const ImportDecl *D) {
  OS << ' ' << D->getImportedModule()->getFullModuleName();
}

void ASTDumper::VisitPragmaCommentDecl(const PragmaCommentDecl *D) {
  OS << ' ';
  switch (D->getCommentKind()) {
  case PCK_Unknown:  llvm_unreachable("unexpected pragma comment kind");
  case PCK_Compiler: OS << "compiler"; break;
  case PCK_ExeStr:   OS << "exestr"; break;
  case PCK_Lib:      OS << "lib"; break;
  case PCK_Linker:   OS << "linker"; break;
  case PCK_User:     OS << "user"; break;
  }
  StringRef Arg = D->getArg();
  if (!Arg.empty())
    OS << " \"" << Arg << "\"";
}

void ASTDumper::VisitPragmaDetectMismatchDecl(
    const PragmaDetectMismatchDecl *D) {
  OS << " \"" << D->getName() << "\" \"" << D->getValue() << "\"";
}

void ASTDumper::VisitCapturedDecl(const CapturedDecl *D) {
  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
// OpenMP Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitOMPThreadPrivateDecl(const OMPThreadPrivateDecl *D) {
  for (auto *E : D->varlists())
    dumpStmt(E);
}

void ASTDumper::VisitOMPDeclareReductionDecl(const OMPDeclareReductionDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  OS << " combiner";
  NodeDumper.dumpPointer(D->getCombiner());
  if (const auto *Initializer = D->getInitializer()) {
    OS << " initializer";
    NodeDumper.dumpPointer(Initializer);
    switch (D->getInitializerKind()) {
    case OMPDeclareReductionDecl::DirectInit:
      OS << " omp_priv = ";
      break;
    case OMPDeclareReductionDecl::CopyInit:
      OS << " omp_priv ()";
      break;
    case OMPDeclareReductionDecl::CallInit:
      break;
    }
  }

  dumpStmt(D->getCombiner());
  if (const auto *Initializer = D->getInitializer())
    dumpStmt(Initializer);
}

void ASTDumper::VisitOMPRequiresDecl(const OMPRequiresDecl *D) {
  for (auto *C : D->clauselists()) {
    dumpChild([=] {
      if (!C) {
        ColorScope Color(OS, ShowColors, NullColor);
        OS << "<<<NULL>>> OMPClause";
        return;
      }
      {
        ColorScope Color(OS, ShowColors, AttrColor);
        StringRef ClauseName(getOpenMPClauseName(C->getClauseKind()));
        OS << "OMP" << ClauseName.substr(/*Start=*/0, /*N=*/1).upper()
           << ClauseName.drop_front() << "Clause";
      }
      NodeDumper.dumpPointer(C);
      NodeDumper.dumpSourceRange(SourceRange(C->getBeginLoc(), C->getEndLoc()));
    });
  }
}

void ASTDumper::VisitOMPCapturedExprDecl(const OMPCapturedExprDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  dumpStmt(D->getInit());
}

//===----------------------------------------------------------------------===//
// C++ Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitNamespaceDecl(const NamespaceDecl *D) {
  NodeDumper.dumpName(D);
  if (D->isInline())
    OS << " inline";
  if (!D->isOriginalNamespace())
    dumpDeclRef(D->getOriginalNamespace(), "original");
}

void ASTDumper::VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
  OS << ' ';
  NodeDumper.dumpBareDeclRef(D->getNominatedNamespace());
}

void ASTDumper::VisitNamespaceAliasDecl(const NamespaceAliasDecl *D) {
  NodeDumper.dumpName(D);
  dumpDeclRef(D->getAliasedNamespace());
}

void ASTDumper::VisitTypeAliasDecl(const TypeAliasDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getUnderlyingType());
  dumpTypeAsChild(D->getUnderlyingType());
}

void ASTDumper::VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D) {
  NodeDumper.dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
}

void ASTDumper::VisitCXXRecordDecl(const CXXRecordDecl *D) {
  VisitRecordDecl(D);
  if (!D->isCompleteDefinition())
    return;

  dumpChild([=] {
    {
      ColorScope Color(OS, ShowColors, DeclKindNameColor);
      OS << "DefinitionData";
    }
#define FLAG(fn, name) if (D->fn()) OS << " " #name;
    FLAG(isParsingBaseSpecifiers, parsing_base_specifiers);

    FLAG(isGenericLambda, generic);
    FLAG(isLambda, lambda);

    FLAG(canPassInRegisters, pass_in_registers);
    FLAG(isEmpty, empty);
    FLAG(isAggregate, aggregate);
    FLAG(isStandardLayout, standard_layout);
    FLAG(isTriviallyCopyable, trivially_copyable);
    FLAG(isPOD, pod);
    FLAG(isTrivial, trivial);
    FLAG(isPolymorphic, polymorphic);
    FLAG(isAbstract, abstract);
    FLAG(isLiteral, literal);

    FLAG(hasUserDeclaredConstructor, has_user_declared_ctor);
    FLAG(hasConstexprNonCopyMoveConstructor, has_constexpr_non_copy_move_ctor);
    FLAG(hasMutableFields, has_mutable_fields);
    FLAG(hasVariantMembers, has_variant_members);
    FLAG(allowConstDefaultInit, can_const_default_init);

    dumpChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "DefaultConstructor";
      }
      FLAG(hasDefaultConstructor, exists);
      FLAG(hasTrivialDefaultConstructor, trivial);
      FLAG(hasNonTrivialDefaultConstructor, non_trivial);
      FLAG(hasUserProvidedDefaultConstructor, user_provided);
      FLAG(hasConstexprDefaultConstructor, constexpr);
      FLAG(needsImplicitDefaultConstructor, needs_implicit);
      FLAG(defaultedDefaultConstructorIsConstexpr, defaulted_is_constexpr);
    });

    dumpChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "CopyConstructor";
      }
      FLAG(hasSimpleCopyConstructor, simple);
      FLAG(hasTrivialCopyConstructor, trivial);
      FLAG(hasNonTrivialCopyConstructor, non_trivial);
      FLAG(hasUserDeclaredCopyConstructor, user_declared);
      FLAG(hasCopyConstructorWithConstParam, has_const_param);
      FLAG(needsImplicitCopyConstructor, needs_implicit);
      FLAG(needsOverloadResolutionForCopyConstructor,
           needs_overload_resolution);
      if (!D->needsOverloadResolutionForCopyConstructor())
        FLAG(defaultedCopyConstructorIsDeleted, defaulted_is_deleted);
      FLAG(implicitCopyConstructorHasConstParam, implicit_has_const_param);
    });

    dumpChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "MoveConstructor";
      }
      FLAG(hasMoveConstructor, exists);
      FLAG(hasSimpleMoveConstructor, simple);
      FLAG(hasTrivialMoveConstructor, trivial);
      FLAG(hasNonTrivialMoveConstructor, non_trivial);
      FLAG(hasUserDeclaredMoveConstructor, user_declared);
      FLAG(needsImplicitMoveConstructor, needs_implicit);
      FLAG(needsOverloadResolutionForMoveConstructor,
           needs_overload_resolution);
      if (!D->needsOverloadResolutionForMoveConstructor())
        FLAG(defaultedMoveConstructorIsDeleted, defaulted_is_deleted);
    });

    dumpChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "CopyAssignment";
      }
      FLAG(hasTrivialCopyAssignment, trivial);
      FLAG(hasNonTrivialCopyAssignment, non_trivial);
      FLAG(hasCopyAssignmentWithConstParam, has_const_param);
      FLAG(hasUserDeclaredCopyAssignment, user_declared);
      FLAG(needsImplicitCopyAssignment, needs_implicit);
      FLAG(needsOverloadResolutionForCopyAssignment, needs_overload_resolution);
      FLAG(implicitCopyAssignmentHasConstParam, implicit_has_const_param);
    });

    dumpChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "MoveAssignment";
      }
      FLAG(hasMoveAssignment, exists);
      FLAG(hasSimpleMoveAssignment, simple);
      FLAG(hasTrivialMoveAssignment, trivial);
      FLAG(hasNonTrivialMoveAssignment, non_trivial);
      FLAG(hasUserDeclaredMoveAssignment, user_declared);
      FLAG(needsImplicitMoveAssignment, needs_implicit);
      FLAG(needsOverloadResolutionForMoveAssignment, needs_overload_resolution);
    });

    dumpChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "Destructor";
      }
      FLAG(hasSimpleDestructor, simple);
      FLAG(hasIrrelevantDestructor, irrelevant);
      FLAG(hasTrivialDestructor, trivial);
      FLAG(hasNonTrivialDestructor, non_trivial);
      FLAG(hasUserDeclaredDestructor, user_declared);
      FLAG(needsImplicitDestructor, needs_implicit);
      FLAG(needsOverloadResolutionForDestructor, needs_overload_resolution);
      if (!D->needsOverloadResolutionForDestructor())
        FLAG(defaultedDestructorIsDeleted, defaulted_is_deleted);
    });
  });

  for (const auto &I : D->bases()) {
    dumpChild([=] {
      if (I.isVirtual())
        OS << "virtual ";
      NodeDumper.dumpAccessSpecifier(I.getAccessSpecifier());
      NodeDumper.dumpType(I.getType());
      if (I.isPackExpansion())
        OS << "...";
    });
  }
}

void ASTDumper::VisitStaticAssertDecl(const StaticAssertDecl *D) {
  dumpStmt(D->getAssertExpr());
  dumpStmt(D->getMessage());
}

template <typename SpecializationDecl>
void ASTDumper::dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                               bool DumpExplicitInst,
                                               bool DumpRefOnly) {
  bool DumpedAny = false;
  for (auto *RedeclWithBadType : D->redecls()) {
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
        dumpDeclRef(Redecl);
      else
        dumpDecl(Redecl);
      DumpedAny = true;
      break;
    case TSK_ExplicitSpecialization:
      break;
    }
  }

  // Ensure we dump at least one decl for each specialization.
  if (!DumpedAny)
    dumpDeclRef(D);
}

template <typename TemplateDecl>
void ASTDumper::dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst) {
  NodeDumper.dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());

  dumpDecl(D->getTemplatedDecl());

  for (auto *Child : D->specializations())
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

void ASTDumper::VisitClassTemplateSpecializationDecl(
    const ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);
  dumpTemplateArgumentList(D->getTemplateArgs());
}

void ASTDumper::VisitClassTemplatePartialSpecializationDecl(
    const ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);
  dumpTemplateParameters(D->getTemplateParameters());
}

void ASTDumper::VisitClassScopeFunctionSpecializationDecl(
    const ClassScopeFunctionSpecializationDecl *D) {
  dumpDecl(D->getSpecialization());
  if (D->hasExplicitTemplateArgs())
    dumpTemplateArgumentListInfo(D->templateArgs());
}

void ASTDumper::VisitVarTemplateDecl(const VarTemplateDecl *D) {
  dumpTemplateDecl(D, false);
}

void ASTDumper::VisitBuiltinTemplateDecl(const BuiltinTemplateDecl *D) {
  NodeDumper.dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
}

void ASTDumper::VisitVarTemplateSpecializationDecl(
    const VarTemplateSpecializationDecl *D) {
  dumpTemplateArgumentList(D->getTemplateArgs());
  VisitVarDecl(D);
}

void ASTDumper::VisitVarTemplatePartialSpecializationDecl(
    const VarTemplatePartialSpecializationDecl *D) {
  dumpTemplateParameters(D->getTemplateParameters());
  VisitVarTemplateSpecializationDecl(D);
}

void ASTDumper::VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D) {
  if (D->wasDeclaredWithTypename())
    OS << " typename";
  else
    OS << " class";
  OS << " depth " << D->getDepth() << " index " << D->getIndex();
  if (D->isParameterPack())
    OS << " ...";
  NodeDumper.dumpName(D);
  if (D->hasDefaultArgument())
    dumpTemplateArgument(D->getDefaultArgument(), SourceRange(),
                         D->getDefaultArgStorage().getInheritedFrom(),
                         D->defaultArgumentWasInherited() ? "inherited from"
                                                          : "previous");
}

void ASTDumper::VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D) {
  NodeDumper.dumpType(D->getType());
  OS << " depth " << D->getDepth() << " index " << D->getIndex();
  if (D->isParameterPack())
    OS << " ...";
  NodeDumper.dumpName(D);
  if (D->hasDefaultArgument())
    dumpTemplateArgument(D->getDefaultArgument(), SourceRange(),
                         D->getDefaultArgStorage().getInheritedFrom(),
                         D->defaultArgumentWasInherited() ? "inherited from"
                                                          : "previous");
}

void ASTDumper::VisitTemplateTemplateParmDecl(
    const TemplateTemplateParmDecl *D) {
  OS << " depth " << D->getDepth() << " index " << D->getIndex();
  if (D->isParameterPack())
    OS << " ...";
  NodeDumper.dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  if (D->hasDefaultArgument())
    dumpTemplateArgumentLoc(
        D->getDefaultArgument(), D->getDefaultArgStorage().getInheritedFrom(),
        D->defaultArgumentWasInherited() ? "inherited from" : "previous");
}

void ASTDumper::VisitUsingDecl(const UsingDecl *D) {
  OS << ' ';
  if (D->getQualifier())
    D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void ASTDumper::VisitUnresolvedUsingTypenameDecl(
    const UnresolvedUsingTypenameDecl *D) {
  OS << ' ';
  if (D->getQualifier())
    D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void ASTDumper::VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D) {
  OS << ' ';
  if (D->getQualifier())
    D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
  NodeDumper.dumpType(D->getType());
}

void ASTDumper::VisitUsingShadowDecl(const UsingShadowDecl *D) {
  OS << ' ';
  NodeDumper.dumpBareDeclRef(D->getTargetDecl());
  if (auto *TD = dyn_cast<TypeDecl>(D->getUnderlyingDecl()))
    dumpTypeAsChild(TD->getTypeForDecl());
}

void ASTDumper::VisitConstructorUsingShadowDecl(
    const ConstructorUsingShadowDecl *D) {
  if (D->constructsVirtualBase())
    OS << " virtual";

  dumpChild([=] {
    OS << "target ";
    NodeDumper.dumpBareDeclRef(D->getTargetDecl());
  });

  dumpChild([=] {
    OS << "nominated ";
    NodeDumper.dumpBareDeclRef(D->getNominatedBaseClass());
    OS << ' ';
    NodeDumper.dumpBareDeclRef(D->getNominatedBaseClassShadowDecl());
  });

  dumpChild([=] {
    OS << "constructed ";
    NodeDumper.dumpBareDeclRef(D->getConstructedBaseClass());
    OS << ' ';
    NodeDumper.dumpBareDeclRef(D->getConstructedBaseClassShadowDecl());
  });
}

void ASTDumper::VisitLinkageSpecDecl(const LinkageSpecDecl *D) {
  switch (D->getLanguage()) {
  case LinkageSpecDecl::lang_c: OS << " C"; break;
  case LinkageSpecDecl::lang_cxx: OS << " C++"; break;
  }
}

void ASTDumper::VisitAccessSpecDecl(const AccessSpecDecl *D) {
  OS << ' ';
  NodeDumper.dumpAccessSpecifier(D->getAccess());
}

void ASTDumper::VisitFriendDecl(const FriendDecl *D) {
  if (TypeSourceInfo *T = D->getFriendType())
    NodeDumper.dumpType(T->getType());
  else
    dumpDecl(D->getFriendDecl());
}

//===----------------------------------------------------------------------===//
// Obj-C Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCIvarDecl(const ObjCIvarDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());
  if (D->getSynthesize())
    OS << " synthesize";

  switch (D->getAccessControl()) {
  case ObjCIvarDecl::None:
    OS << " none";
    break;
  case ObjCIvarDecl::Private:
    OS << " private";
    break;
  case ObjCIvarDecl::Protected:
    OS << " protected";
    break;
  case ObjCIvarDecl::Public:
    OS << " public";
    break;
  case ObjCIvarDecl::Package:
    OS << " package";
    break;
  }
}

void ASTDumper::VisitObjCMethodDecl(const ObjCMethodDecl *D) {
  if (D->isInstanceMethod())
    OS << " -";
  else
    OS << " +";
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getReturnType());

  if (D->isThisDeclarationADefinition()) {
    dumpDeclContext(D);
  } else {
    for (const ParmVarDecl *Parameter : D->parameters())
      dumpDecl(Parameter);
  }

  if (D->isVariadic())
    dumpChild([=] { OS << "..."; });

  if (D->hasBody())
    dumpStmt(D->getBody());
}

void ASTDumper::VisitObjCTypeParamDecl(const ObjCTypeParamDecl *D) {
  NodeDumper.dumpName(D);
  switch (D->getVariance()) {
  case ObjCTypeParamVariance::Invariant:
    break;

  case ObjCTypeParamVariance::Covariant:
    OS << " covariant";
    break;

  case ObjCTypeParamVariance::Contravariant:
    OS << " contravariant";
    break;
  }

  if (D->hasExplicitBound())
    OS << " bounded";
  NodeDumper.dumpType(D->getUnderlyingType());
}

void ASTDumper::VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
  NodeDumper.dumpName(D);
  dumpDeclRef(D->getClassInterface());
  dumpObjCTypeParamList(D->getTypeParamList());
  dumpDeclRef(D->getImplementation());
  for (ObjCCategoryDecl::protocol_iterator I = D->protocol_begin(),
                                           E = D->protocol_end();
       I != E; ++I)
    dumpDeclRef(*I);
}

void ASTDumper::VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D) {
  NodeDumper.dumpName(D);
  dumpDeclRef(D->getClassInterface());
  dumpDeclRef(D->getCategoryDecl());
}

void ASTDumper::VisitObjCProtocolDecl(const ObjCProtocolDecl *D) {
  NodeDumper.dumpName(D);

  for (auto *Child : D->protocols())
    dumpDeclRef(Child);
}

void ASTDumper::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
  NodeDumper.dumpName(D);
  dumpObjCTypeParamList(D->getTypeParamListAsWritten());
  dumpDeclRef(D->getSuperClass(), "super");

  dumpDeclRef(D->getImplementation());
  for (auto *Child : D->protocols())
    dumpDeclRef(Child);
}

void ASTDumper::VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
  NodeDumper.dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");
  dumpDeclRef(D->getClassInterface());
  for (ObjCImplementationDecl::init_const_iterator I = D->init_begin(),
                                                   E = D->init_end();
       I != E; ++I)
    dumpCXXCtorInitializer(*I);
}

void ASTDumper::VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *D) {
  NodeDumper.dumpName(D);
  dumpDeclRef(D->getClassInterface());
  OS << " ";
  NodeDumper.dumpLocation(D->getClassInterfaceLoc());
}

void ASTDumper::VisitObjCPropertyDecl(const ObjCPropertyDecl *D) {
  NodeDumper.dumpName(D);
  NodeDumper.dumpType(D->getType());

  if (D->getPropertyImplementation() == ObjCPropertyDecl::Required)
    OS << " required";
  else if (D->getPropertyImplementation() == ObjCPropertyDecl::Optional)
    OS << " optional";

  ObjCPropertyDecl::PropertyAttributeKind Attrs = D->getPropertyAttributes();
  if (Attrs != ObjCPropertyDecl::OBJC_PR_noattr) {
    if (Attrs & ObjCPropertyDecl::OBJC_PR_readonly)
      OS << " readonly";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_assign)
      OS << " assign";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_readwrite)
      OS << " readwrite";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_retain)
      OS << " retain";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_copy)
      OS << " copy";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_nonatomic)
      OS << " nonatomic";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_atomic)
      OS << " atomic";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_weak)
      OS << " weak";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_strong)
      OS << " strong";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_unsafe_unretained)
      OS << " unsafe_unretained";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_class)
      OS << " class";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_getter)
      dumpDeclRef(D->getGetterMethodDecl(), "getter");
    if (Attrs & ObjCPropertyDecl::OBJC_PR_setter)
      dumpDeclRef(D->getSetterMethodDecl(), "setter");
  }
}

void ASTDumper::VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D) {
  NodeDumper.dumpName(D->getPropertyDecl());
  if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    OS << " synthesize";
  else
    OS << " dynamic";
  dumpDeclRef(D->getPropertyDecl());
  dumpDeclRef(D->getPropertyIvarDecl());
}

void ASTDumper::VisitBlockDecl(const BlockDecl *D) {
  for (auto I : D->parameters())
    dumpDecl(I);

  if (D->isVariadic())
    dumpChild([=]{ OS << "..."; });

  if (D->capturesCXXThis())
    dumpChild([=]{ OS << "capture this"; });

  for (const auto &I : D->captures()) {
    dumpChild([=] {
      OS << "capture";
      if (I.isByRef())
        OS << " byref";
      if (I.isNested())
        OS << " nested";
      if (I.getVariable()) {
        OS << ' ';
        NodeDumper.dumpBareDeclRef(I.getVariable());
      }
      if (I.hasCopyExpr())
        dumpStmt(I.getCopyExpr());
    });
  }
  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
//  Stmt dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpStmt(const Stmt *S) {
  dumpChild([=] {
    if (!S) {
      ColorScope Color(OS, ShowColors, NullColor);
      OS << "<<<NULL>>>";
      return;
    }
    {
      ColorScope Color(OS, ShowColors, StmtColor);
      OS << S->getStmtClassName();
    }
    NodeDumper.dumpPointer(S);
    NodeDumper.dumpSourceRange(S->getSourceRange());

    if (const auto *E = dyn_cast<Expr>(S)) {
      NodeDumper.dumpType(E->getType());

      {
        ColorScope Color(OS, ShowColors, ValueKindColor);
        switch (E->getValueKind()) {
        case VK_RValue:
          break;
        case VK_LValue:
          OS << " lvalue";
          break;
        case VK_XValue:
          OS << " xvalue";
          break;
        }
      }

      {
        ColorScope Color(OS, ShowColors, ObjectKindColor);
        switch (E->getObjectKind()) {
        case OK_Ordinary:
          break;
        case OK_BitField:
          OS << " bitfield";
          break;
        case OK_ObjCProperty:
          OS << " objcproperty";
          break;
        case OK_ObjCSubscript:
          OS << " objcsubscript";
          break;
        case OK_VectorComponent:
          OS << " vectorcomponent";
          break;
        }
      }
    }

    ConstStmtVisitor<ASTDumper>::Visit(S);

    // Some statements have custom mechanisms for dumping their children.
    if (isa<DeclStmt>(S) || isa<GenericSelectionExpr>(S)) {
      return;
    }

    for (const Stmt *SubStmt : S->children())
      dumpStmt(SubStmt);
  });
}

void ASTDumper::VisitDeclStmt(const DeclStmt *Node) {
  for (DeclStmt::const_decl_iterator I = Node->decl_begin(),
                                     E = Node->decl_end();
       I != E; ++I)
    dumpDecl(*I);
}

void ASTDumper::VisitAttributedStmt(const AttributedStmt *Node) {
  for (ArrayRef<const Attr *>::iterator I = Node->getAttrs().begin(),
                                        E = Node->getAttrs().end();
       I != E; ++I)
    dumpAttr(*I);
}

void ASTDumper::VisitIfStmt(const IfStmt *Node) {
  if (Node->hasInitStorage())
    OS << " has_init";
  if (Node->hasVarStorage())
    OS << " has_var";
  if (Node->hasElseStorage())
    OS << " has_else";
}

void ASTDumper::VisitSwitchStmt(const SwitchStmt *Node) {
  if (Node->hasInitStorage())
    OS << " has_init";
  if (Node->hasVarStorage())
    OS << " has_var";
}

void ASTDumper::VisitWhileStmt(const WhileStmt *Node) {
  if (Node->hasVarStorage())
    OS << " has_var";
}

void ASTDumper::VisitLabelStmt(const LabelStmt *Node) {
  OS << " '" << Node->getName() << "'";
}

void ASTDumper::VisitGotoStmt(const GotoStmt *Node) {
  OS << " '" << Node->getLabel()->getName() << "'";
  NodeDumper.dumpPointer(Node->getLabel());
}

void ASTDumper::VisitCXXCatchStmt(const CXXCatchStmt *Node) {
  dumpDecl(Node->getExceptionDecl());
}

void ASTDumper::VisitCaseStmt(const CaseStmt *Node) {
  if (Node->caseStmtIsGNURange())
    OS << " gnu_range";
}

void ASTDumper::VisitCapturedStmt(const CapturedStmt *Node) {
  dumpDecl(Node->getCapturedDecl());
}

//===----------------------------------------------------------------------===//
//  OpenMP dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::VisitOMPExecutableDirective(
    const OMPExecutableDirective *Node) {
  for (auto *C : Node->clauses()) {
    dumpChild([=] {
      if (!C) {
        ColorScope Color(OS, ShowColors, NullColor);
        OS << "<<<NULL>>> OMPClause";
        return;
      }
      {
        ColorScope Color(OS, ShowColors, AttrColor);
        StringRef ClauseName(getOpenMPClauseName(C->getClauseKind()));
        OS << "OMP" << ClauseName.substr(/*Start=*/0, /*N=*/1).upper()
           << ClauseName.drop_front() << "Clause";
      }
      NodeDumper.dumpPointer(C);
      NodeDumper.dumpSourceRange(SourceRange(C->getBeginLoc(), C->getEndLoc()));
      if (C->isImplicit())
        OS << " <implicit>";
      for (auto *S : C->children())
        dumpStmt(S);
    });
  }
}

//===----------------------------------------------------------------------===//
//  Expr dumping methods.
//===----------------------------------------------------------------------===//

static void dumpBasePath(raw_ostream &OS, const CastExpr *Node) {
  if (Node->path_empty())
    return;

  OS << " (";
  bool First = true;
  for (CastExpr::path_const_iterator I = Node->path_begin(),
                                     E = Node->path_end();
       I != E; ++I) {
    const CXXBaseSpecifier *Base = *I;
    if (!First)
      OS << " -> ";

    const CXXRecordDecl *RD =
    cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());

    if (Base->isVirtual())
      OS << "virtual ";
    OS << RD->getName();
    First = false;
  }

  OS << ')';
}

void ASTDumper::VisitCallExpr(const CallExpr *Node) {
  if (Node->usesADL())
    OS << " adl";
}

void ASTDumper::VisitCastExpr(const CastExpr *Node) {
  OS << " <";
  {
    ColorScope Color(OS, ShowColors, CastColor);
    OS << Node->getCastKindName();
  }
  dumpBasePath(OS, Node);
  OS << ">";
}

void ASTDumper::VisitImplicitCastExpr(const ImplicitCastExpr *Node) {
  VisitCastExpr(Node);
  if (Node->isPartOfExplicitCast())
    OS << " part_of_explicit_cast";
}

void ASTDumper::VisitDeclRefExpr(const DeclRefExpr *Node) {
  OS << " ";
  NodeDumper.dumpBareDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    OS << " (";
    NodeDumper.dumpBareDeclRef(Node->getFoundDecl());
    OS << ")";
  }
}

void ASTDumper::VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *Node) {
  OS << " (";
  if (!Node->requiresADL())
    OS << "no ";
  OS << "ADL) = '" << Node->getName() << '\'';

  UnresolvedLookupExpr::decls_iterator
    I = Node->decls_begin(), E = Node->decls_end();
  if (I == E)
    OS << " empty";
  for (; I != E; ++I)
    NodeDumper.dumpPointer(*I);
}

void ASTDumper::VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Node) {
  {
    ColorScope Color(OS, ShowColors, DeclKindNameColor);
    OS << " " << Node->getDecl()->getDeclKindName() << "Decl";
  }
  OS << "='" << *Node->getDecl() << "'";
  NodeDumper.dumpPointer(Node->getDecl());
  if (Node->isFreeIvar())
    OS << " isFreeIvar";
}

void ASTDumper::VisitPredefinedExpr(const PredefinedExpr *Node) {
  OS << " " << PredefinedExpr::getIdentKindName(Node->getIdentKind());
}

void ASTDumper::VisitCharacterLiteral(const CharacterLiteral *Node) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValue();
}

void ASTDumper::VisitIntegerLiteral(const IntegerLiteral *Node) {
  bool isSigned = Node->getType()->isSignedIntegerType();
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValue().toString(10, isSigned);
}

void ASTDumper::VisitFixedPointLiteral(const FixedPointLiteral *Node) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValueAsString(/*Radix=*/10);
}

void ASTDumper::VisitFloatingLiteral(const FloatingLiteral *Node) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValueAsApproximateDouble();
}

void ASTDumper::VisitStringLiteral(const StringLiteral *Str) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " ";
  Str->outputString(OS);
}

void ASTDumper::VisitInitListExpr(const InitListExpr *ILE) {
  if (auto *Field = ILE->getInitializedFieldInUnion()) {
    OS << " field ";
    NodeDumper.dumpBareDeclRef(Field);
  }
  if (auto *Filler = ILE->getArrayFiller()) {
    dumpChild([=] {
      OS << "array filler";
      dumpStmt(Filler);
    });
  }
}

void ASTDumper::VisitUnaryOperator(const UnaryOperator *Node) {
  OS << " " << (Node->isPostfix() ? "postfix" : "prefix")
     << " '" << UnaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
  if (!Node->canOverflow())
    OS << " cannot overflow";
}

void ASTDumper::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *Node) {
  switch(Node->getKind()) {
  case UETT_SizeOf:
    OS << " sizeof";
    break;
  case UETT_AlignOf:
    OS << " alignof";
    break;
  case UETT_VecStep:
    OS << " vec_step";
    break;
  case UETT_OpenMPRequiredSimdAlign:
    OS << " __builtin_omp_required_simd_align";
    break;
  case UETT_PreferredAlignOf:
    OS << " __alignof";
    break;
  }
  if (Node->isArgumentType())
    NodeDumper.dumpType(Node->getArgumentType());
}

void ASTDumper::VisitMemberExpr(const MemberExpr *Node) {
  OS << " " << (Node->isArrow() ? "->" : ".") << *Node->getMemberDecl();
  NodeDumper.dumpPointer(Node->getMemberDecl());
}

void ASTDumper::VisitExtVectorElementExpr(const ExtVectorElementExpr *Node) {
  OS << " " << Node->getAccessor().getNameStart();
}

void ASTDumper::VisitBinaryOperator(const BinaryOperator *Node) {
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}

void ASTDumper::VisitCompoundAssignOperator(
    const CompoundAssignOperator *Node) {
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode())
     << "' ComputeLHSTy=";
  NodeDumper.dumpBareType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  NodeDumper.dumpBareType(Node->getComputationResultType());
}

void ASTDumper::VisitBlockExpr(const BlockExpr *Node) {
  dumpDecl(Node->getBlockDecl());
}

void ASTDumper::VisitOpaqueValueExpr(const OpaqueValueExpr *Node) {
  if (Expr *Source = Node->getSourceExpr())
    dumpStmt(Source);
}

void ASTDumper::VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
  if (E->isResultDependent())
    OS << " result_dependent";
  dumpStmt(E->getControllingExpr());
  dumpTypeAsChild(E->getControllingExpr()->getType()); // FIXME: remove

  for (unsigned I = 0, N = E->getNumAssocs(); I != N; ++I) {
    dumpChild([=] {
      if (const TypeSourceInfo *TSI = E->getAssocTypeSourceInfo(I)) {
        OS << "case ";
        NodeDumper.dumpType(TSI->getType());
      } else {
        OS << "default";
      }

      if (!E->isResultDependent() && E->getResultIndex() == I)
        OS << " selected";

      if (const TypeSourceInfo *TSI = E->getAssocTypeSourceInfo(I))
        dumpTypeAsChild(TSI->getType());
      dumpStmt(E->getAssocExpr(I));
    });
  }
}

// GNU extensions.

void ASTDumper::VisitAddrLabelExpr(const AddrLabelExpr *Node) {
  OS << " " << Node->getLabel()->getName();
  NodeDumper.dumpPointer(Node->getLabel());
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node) {
  OS << " " << Node->getCastName()
     << "<" << Node->getTypeAsWritten().getAsString() << ">"
     << " <" << Node->getCastKindName();
  dumpBasePath(OS, Node);
  OS << ">";
}

void ASTDumper::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node) {
  OS << " " << (Node->getValue() ? "true" : "false");
}

void ASTDumper::VisitCXXThisExpr(const CXXThisExpr *Node) {
  OS << " this";
}

void ASTDumper::VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *Node) {
  OS << " functional cast to " << Node->getTypeAsWritten().getAsString()
     << " <" << Node->getCastKindName() << ">";
}

void ASTDumper::VisitCXXUnresolvedConstructExpr(
    const CXXUnresolvedConstructExpr *Node) {
  NodeDumper.dumpType(Node->getTypeAsWritten());
  if (Node->isListInitialization())
    OS << " list";
}

void ASTDumper::VisitCXXConstructExpr(const CXXConstructExpr *Node) {
  CXXConstructorDecl *Ctor = Node->getConstructor();
  NodeDumper.dumpType(Ctor->getType());
  if (Node->isElidable())
    OS << " elidable";
  if (Node->isListInitialization())
    OS << " list";
  if (Node->isStdInitListInitialization())
    OS << " std::initializer_list";
  if (Node->requiresZeroInitialization())
    OS << " zeroing";
}

void ASTDumper::VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Node) {
  OS << " ";
  NodeDumper.dumpCXXTemporary(Node->getTemporary());
}

void ASTDumper::VisitCXXNewExpr(const CXXNewExpr *Node) {
  if (Node->isGlobalNew())
    OS << " global";
  if (Node->isArray())
    OS << " array";
  if (Node->getOperatorNew()) {
    OS << ' ';
    NodeDumper.dumpBareDeclRef(Node->getOperatorNew());
  }
  // We could dump the deallocation function used in case of error, but it's
  // usually not that interesting.
}

void ASTDumper::VisitCXXDeleteExpr(const CXXDeleteExpr *Node) {
  if (Node->isGlobalDelete())
    OS << " global";
  if (Node->isArrayForm())
    OS << " array";
  if (Node->getOperatorDelete()) {
    OS << ' ';
    NodeDumper.dumpBareDeclRef(Node->getOperatorDelete());
  }
}

void
ASTDumper::VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Node) {
  if (const ValueDecl *VD = Node->getExtendingDecl()) {
    OS << " extended by ";
    NodeDumper.dumpBareDeclRef(VD);
  }
}

void ASTDumper::VisitExprWithCleanups(const ExprWithCleanups *Node) {
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i)
    dumpDeclRef(Node->getObject(i), "cleanup");
}

void ASTDumper::VisitSizeOfPackExpr(const SizeOfPackExpr *Node) {
  NodeDumper.dumpPointer(Node->getPack());
  NodeDumper.dumpName(Node->getPack());
  if (Node->isPartiallySubstituted())
    for (const auto &A : Node->getPartialArguments())
      dumpTemplateArgument(A);
}

void ASTDumper::VisitCXXDependentScopeMemberExpr(
    const CXXDependentScopeMemberExpr *Node) {
  OS << " " << (Node->isArrow() ? "->" : ".") << Node->getMember();
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCMessageExpr(const ObjCMessageExpr *Node) {
  OS << " selector=";
  Node->getSelector().print(OS);
  switch (Node->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    break;

  case ObjCMessageExpr::Class:
    OS << " class=";
    NodeDumper.dumpBareType(Node->getClassReceiver());
    break;

  case ObjCMessageExpr::SuperInstance:
    OS << " super (instance)";
    break;

  case ObjCMessageExpr::SuperClass:
    OS << " super (class)";
    break;
  }
}

void ASTDumper::VisitObjCBoxedExpr(const ObjCBoxedExpr *Node) {
  if (auto *BoxingMethod = Node->getBoxingMethod()) {
    OS << " selector=";
    BoxingMethod->getSelector().print(OS);
  }
}

void ASTDumper::VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node) {
  if (const VarDecl *CatchParam = Node->getCatchParamDecl())
    dumpDecl(CatchParam);
  else
    OS << " catch all";
}

void ASTDumper::VisitObjCEncodeExpr(const ObjCEncodeExpr *Node) {
  NodeDumper.dumpType(Node->getEncodedType());
}

void ASTDumper::VisitObjCSelectorExpr(const ObjCSelectorExpr *Node) {
  OS << " ";
  Node->getSelector().print(OS);
}

void ASTDumper::VisitObjCProtocolExpr(const ObjCProtocolExpr *Node) {
  OS << ' ' << *Node->getProtocol();
}

void ASTDumper::VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Node) {
  if (Node->isImplicitProperty()) {
    OS << " Kind=MethodRef Getter=\"";
    if (Node->getImplicitPropertyGetter())
      Node->getImplicitPropertyGetter()->getSelector().print(OS);
    else
      OS << "(null)";

    OS << "\" Setter=\"";
    if (ObjCMethodDecl *Setter = Node->getImplicitPropertySetter())
      Setter->getSelector().print(OS);
    else
      OS << "(null)";
    OS << "\"";
  } else {
    OS << " Kind=PropertyRef Property=\"" << *Node->getExplicitProperty() <<'"';
  }

  if (Node->isSuperReceiver())
    OS << " super";

  OS << " Messaging=";
  if (Node->isMessagingGetter() && Node->isMessagingSetter())
    OS << "Getter&Setter";
  else if (Node->isMessagingGetter())
    OS << "Getter";
  else if (Node->isMessagingSetter())
    OS << "Setter";
}

void ASTDumper::VisitObjCSubscriptRefExpr(const ObjCSubscriptRefExpr *Node) {
  if (Node->isArraySubscriptRefExpr())
    OS << " Kind=ArraySubscript GetterForArray=\"";
  else
    OS << " Kind=DictionarySubscript GetterForDictionary=\"";
  if (Node->getAtIndexMethodDecl())
    Node->getAtIndexMethodDecl()->getSelector().print(OS);
  else
    OS << "(null)";

  if (Node->isArraySubscriptRefExpr())
    OS << "\" SetterForArray=\"";
  else
    OS << "\" SetterForDictionary=\"";
  if (Node->setAtIndexMethodDecl())
    Node->setAtIndexMethodDecl()->getSelector().print(OS);
  else
    OS << "(null)";
}

void ASTDumper::VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *Node) {
  OS << " " << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

//===----------------------------------------------------------------------===//
// Comments
//===----------------------------------------------------------------------===//

void ASTDumper::dumpComment(const Comment *C, const FullComment *FC) {
  dumpChild([=] {
    NodeDumper.Visit(C, FC);
    if (!C) {
      return;
    }
    ConstCommentVisitor<ASTDumper, void, const FullComment *>::visit(C, FC);
    for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
         I != E; ++I)
      dumpComment(*I, FC);
  });
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
  Dumper.dumpTypeAsChild(*this);
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
  P.dumpDecl(this);
}

LLVM_DUMP_METHOD void Decl::dumpColor() const {
  const ASTContext &Ctx = getASTContext();
  ASTDumper P(llvm::errs(), &Ctx.getCommentCommandTraits(),
              &Ctx.getSourceManager(), /*ShowColors*/ true,
              Ctx.getPrintingPolicy());
  P.dumpDecl(this);
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
  P.dumpStmt(this);
}

LLVM_DUMP_METHOD void Stmt::dump(raw_ostream &OS) const {
  ASTDumper P(OS, nullptr, nullptr);
  P.dumpStmt(this);
}

LLVM_DUMP_METHOD void Stmt::dump() const {
  ASTDumper P(llvm::errs(), nullptr, nullptr);
  P.dumpStmt(this);
}

LLVM_DUMP_METHOD void Stmt::dumpColor() const {
  ASTDumper P(llvm::errs(), nullptr, nullptr, /*ShowColors*/true);
  P.dumpStmt(this);
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
  D.dumpComment(FC, FC);
}

LLVM_DUMP_METHOD void Comment::dumpColor() const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  if (!FC)
    return;
  ASTDumper D(llvm::errs(), nullptr, nullptr, /*ShowColors*/true);
  D.dumpComment(FC, FC);
}
