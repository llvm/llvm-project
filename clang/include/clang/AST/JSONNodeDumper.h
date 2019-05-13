//===--- JSONNodeDumper.h - Printing of AST nodes to JSON -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements AST dumping of components of individual AST nodes to
// a JSON.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_JSONNODEDUMPER_H
#define LLVM_CLANG_AST_JSONNODEDUMPER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTNodeTraverser.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/AttrVisitor.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/JSON.h"

namespace clang {

class NodeStreamer {
  bool FirstChild = true;
  bool TopLevel = true;
  llvm::SmallVector<std::function<void(bool IsLastChild)>, 32> Pending;

protected:
  llvm::json::OStream JOS;

public:
  /// Add a child of the current node.  Calls DoAddChild without arguments
  template <typename Fn> void AddChild(Fn DoAddChild) {
    return AddChild("", DoAddChild);
  }

  /// Add a child of the current node with an optional label.
  /// Calls DoAddChild without arguments.
  template <typename Fn> void AddChild(StringRef Label, Fn DoAddChild) {
    // If we're at the top level, there's nothing interesting to do; just
    // run the dumper.
    if (TopLevel) {
      TopLevel = false;
      JOS.objectBegin();

      DoAddChild();

      while (!Pending.empty()) {
        Pending.back()(true);
        Pending.pop_back();
      }

      JOS.objectEnd();
      TopLevel = true;
      return;
    }

    // We need to capture an owning-string in the lambda because the lambda
    // is invoked in a deferred manner.
    std::string LabelStr = !Label.empty() ? Label : "inner";
    bool WasFirstChild = FirstChild;
    auto DumpWithIndent = [=](bool IsLastChild) {
      if (WasFirstChild) {
        JOS.attributeBegin(LabelStr);
        JOS.arrayBegin();
      }

      FirstChild = true;
      unsigned Depth = Pending.size();
      JOS.objectBegin();

      DoAddChild();

      // If any children are left, they're the last at their nesting level.
      // Dump those ones out now.
      while (Depth < Pending.size()) {
        Pending.back()(true);
        this->Pending.pop_back();
      }

      JOS.objectEnd();

      if (IsLastChild) {
        JOS.arrayEnd();
        JOS.attributeEnd();
      }
    };

    if (FirstChild) {
      Pending.push_back(std::move(DumpWithIndent));
    } else {
      Pending.back()(false);
      Pending.back() = std::move(DumpWithIndent);
    }
    FirstChild = false;
  }

  NodeStreamer(raw_ostream &OS) : JOS(OS, 2) {}
};

// Dumps AST nodes in JSON format. There is no implied stability for the
// content or format of the dump between major releases of Clang, other than it
// being valid JSON output. Further, there is no requirement that the
// information dumped is a complete representation of the AST, only that the
// information presented is correct.
class JSONNodeDumper
    : public ConstAttrVisitor<JSONNodeDumper>,
      public ConstTemplateArgumentVisitor<JSONNodeDumper>,
      public ConstStmtVisitor<JSONNodeDumper>,
      public TypeVisitor<JSONNodeDumper>,
      public ConstDeclVisitor<JSONNodeDumper>,
      public NodeStreamer {
  friend class JSONDumper;

  raw_ostream &OS;
  const SourceManager &SM;
  PrintingPolicy PrintPolicy;

  using InnerAttrVisitor = ConstAttrVisitor<JSONNodeDumper>;
  using InnerTemplateArgVisitor = ConstTemplateArgumentVisitor<JSONNodeDumper>;
  using InnerStmtVisitor = ConstStmtVisitor<JSONNodeDumper>;
  using InnerTypeVisitor = TypeVisitor<JSONNodeDumper>;
  using InnerDeclVisitor = ConstDeclVisitor<JSONNodeDumper>;

  void attributeOnlyIfTrue(StringRef Key, bool Value) {
    if (Value)
      JOS.attribute(Key, Value);
  }

  llvm::json::Object createSourceLocation(SourceLocation Loc);
  llvm::json::Object createSourceRange(SourceRange R);
  std::string createPointerRepresentation(const void *Ptr);
  llvm::json::Object createQualType(QualType QT, bool Desugar = true);
  llvm::json::Object createBareDeclRef(const Decl *D);
  llvm::json::Object createCXXRecordDefinitionData(const CXXRecordDecl *RD);
  llvm::json::Object createCXXBaseSpecifier(const CXXBaseSpecifier &BS);
  std::string createAccessSpecifier(AccessSpecifier AS);
  llvm::json::Array createCastPath(const CastExpr *C);

  void writePreviousDeclImpl(...) {}

  template <typename T> void writePreviousDeclImpl(const Mergeable<T> *D) {
    const T *First = D->getFirstDecl();
    if (First != D)
      JOS.attribute("firstRedecl", createPointerRepresentation(First));
  }

  template <typename T> void writePreviousDeclImpl(const Redeclarable<T> *D) {
    const T *Prev = D->getPreviousDecl();
    if (Prev)
      JOS.attribute("previousDecl", createPointerRepresentation(Prev));
  }
  void addPreviousDeclaration(const Decl *D);

public:
  JSONNodeDumper(raw_ostream &OS, const SourceManager &SrcMgr,
                 const PrintingPolicy &PrintPolicy)
      : NodeStreamer(OS), OS(OS), SM(SrcMgr), PrintPolicy(PrintPolicy) {}

  void Visit(const Attr *A);
  void Visit(const Stmt *Node);
  void Visit(const Type *T);
  void Visit(QualType T);
  void Visit(const Decl *D);

  void Visit(const comments::Comment *C, const comments::FullComment *FC);
  void Visit(const TemplateArgument &TA, SourceRange R = {},
             const Decl *From = nullptr, StringRef Label = {});
  void Visit(const CXXCtorInitializer *Init);
  void Visit(const OMPClause *C);
  void Visit(const BlockDecl::Capture &C);
  void Visit(const GenericSelectionExpr::ConstAssociation &A);

  void VisitTypedefType(const TypedefType *TT);
  void VisitFunctionType(const FunctionType *T);
  void VisitFunctionProtoType(const FunctionProtoType *T);

  void VisitNamedDecl(const NamedDecl *ND);
  void VisitTypedefDecl(const TypedefDecl *TD);
  void VisitTypeAliasDecl(const TypeAliasDecl *TAD);
  void VisitNamespaceDecl(const NamespaceDecl *ND);
  void VisitUsingDirectiveDecl(const UsingDirectiveDecl *UDD);
  void VisitNamespaceAliasDecl(const NamespaceAliasDecl *NAD);
  void VisitUsingDecl(const UsingDecl *UD);
  void VisitUsingShadowDecl(const UsingShadowDecl *USD);
  void VisitVarDecl(const VarDecl *VD);
  void VisitFieldDecl(const FieldDecl *FD);
  void VisitFunctionDecl(const FunctionDecl *FD);
  void VisitEnumDecl(const EnumDecl *ED);
  void VisitEnumConstantDecl(const EnumConstantDecl *ECD);
  void VisitRecordDecl(const RecordDecl *RD);
  void VisitCXXRecordDecl(const CXXRecordDecl *RD);
  void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D);
  void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D);
  void VisitTemplateTemplateParmDecl(const TemplateTemplateParmDecl *D);
  void VisitLinkageSpecDecl(const LinkageSpecDecl *LSD);
  void VisitAccessSpecDecl(const AccessSpecDecl *ASD);
  void VisitFriendDecl(const FriendDecl *FD);

  void VisitDeclRefExpr(const DeclRefExpr *DRE);
  void VisitPredefinedExpr(const PredefinedExpr *PE);
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitCompoundAssignOperator(const CompoundAssignOperator *CAO);
  void VisitMemberExpr(const MemberExpr *ME);
  void VisitCXXNewExpr(const CXXNewExpr *NE);
  void VisitCXXDeleteExpr(const CXXDeleteExpr *DE);
  void VisitCXXThisExpr(const CXXThisExpr *TE);
  void VisitCastExpr(const CastExpr *CE);
  void VisitImplicitCastExpr(const ImplicitCastExpr *ICE);
  void VisitCallExpr(const CallExpr *CE);
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *TTE);
  void VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *ULE);
  void VisitAddrLabelExpr(const AddrLabelExpr *ALE);

  void VisitIntegerLiteral(const IntegerLiteral *IL);
  void VisitCharacterLiteral(const CharacterLiteral *CL);
  void VisitFixedPointLiteral(const FixedPointLiteral *FPL);
  void VisitFloatingLiteral(const FloatingLiteral *FL);
  void VisitStringLiteral(const StringLiteral *SL);
  void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *BLE);

  void VisitIfStmt(const IfStmt *IS);
  void VisitSwitchStmt(const SwitchStmt *SS);
  void VisitCaseStmt(const CaseStmt *CS);
  void VisitLabelStmt(const LabelStmt *LS);
  void VisitGotoStmt(const GotoStmt *GS);
  void VisitWhileStmt(const WhileStmt *WS);
};

class JSONDumper : public ASTNodeTraverser<JSONDumper, JSONNodeDumper> {
  JSONNodeDumper NodeDumper;

  template <typename SpecializationDecl>
  void writeTemplateDeclSpecialization(const SpecializationDecl *SD,
                                       bool DumpExplicitInst,
                                       bool DumpRefOnly) {
    bool DumpedAny = false;
    for (const auto *RedeclWithBadType : SD->redecls()) {
      // FIXME: The redecls() range sometimes has elements of a less-specific
      // type. (In particular, ClassTemplateSpecializationDecl::redecls() gives
      // us TagDecls, and should give CXXRecordDecls).
      const auto *Redecl = dyn_cast<SpecializationDecl>(RedeclWithBadType);
      if (!Redecl) {
        // Found the injected-class-name for a class template. This will be
        // dumped as part of its surrounding class so we don't need to dump it
        // here.
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
          NodeDumper.JOS.value(NodeDumper.createBareDeclRef(Redecl));
        else
          // FIXME: this isn't quite right -- we want to call Visit() rather
          // than NodeDumper.Visit() but that causes issues because it attempts
          // to create a new array of child objects due to calling AddChild(),
          // which messes up the JSON creation.
          NodeDumper.JOS.object([this, Redecl] { NodeDumper.Visit(Redecl); });
        DumpedAny = true;
        break;
      case TSK_ExplicitSpecialization:
        break;
      }
    }

    // Ensure we dump at least one decl for each specialization.
    if (!DumpedAny)
      NodeDumper.JOS.value(NodeDumper.createBareDeclRef(SD));
  }

  template <typename TemplateDecl>
  void writeTemplateDecl(const TemplateDecl *TD, bool DumpExplicitInst) {
    if (const TemplateParameterList *TPL = TD->getTemplateParameters()) {
      NodeDumper.JOS.attributeArray("templateParams", [this, TPL] {
        for (const auto &TP : *TPL) {
          NodeDumper.JOS.object([this, TP] { NodeDumper.Visit(TP); });
        }
      });
    }

    Visit(TD->getTemplatedDecl());

    auto spec_range = TD->specializations();
    if (!llvm::empty(spec_range)) {
      NodeDumper.JOS.attributeArray(
          "specializations", [this, spec_range, TD, DumpExplicitInst] {
            for (const auto *Child : spec_range)
              writeTemplateDeclSpecialization(Child, DumpExplicitInst,
                                              !TD->isCanonicalDecl());
          });
    }
  }

public:
  JSONDumper(raw_ostream &OS, const SourceManager &SrcMgr,
             const PrintingPolicy &PrintPolicy)
      : NodeDumper(OS, SrcMgr, PrintPolicy) {}

  JSONNodeDumper &doGetNodeDelegate() { return NodeDumper; }

  void VisitFunctionTemplateDecl(const FunctionTemplateDecl *FTD) {
    writeTemplateDecl(FTD, true);
  }
  void VisitClassTemplateDecl(const ClassTemplateDecl *CTD) {
    writeTemplateDecl(CTD, false);
  }
  void VisitVarTemplateDecl(const VarTemplateDecl *VTD) {
    writeTemplateDecl(VTD, false);
  }
};

} // namespace clang

#endif // LLVM_CLANG_AST_JSONNODEDUMPER_H
