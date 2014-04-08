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
#include "clang/AST/Attr.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace clang::comments;

//===----------------------------------------------------------------------===//
// ASTDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  // Colors used for various parts of the AST dump
  // Do not use bold yellow for any text.  It is hard to read on white screens.

  struct TerminalColor {
    raw_ostream::Colors Color;
    bool Bold;
  };

  // Red           - CastColor
  // Green         - TypeColor
  // Bold Green    - DeclKindNameColor, UndeserializedColor
  // Yellow        - AddressColor, LocationColor
  // Blue          - CommentColor, NullColor, IndentColor
  // Bold Blue     - AttrColor
  // Bold Magenta  - StmtColor
  // Cyan          - ValueKindColor, ObjectKindColor
  // Bold Cyan     - ValueColor, DeclNameColor

  // Decl kind names (VarDecl, FunctionDecl, etc)
  static const TerminalColor DeclKindNameColor = { raw_ostream::GREEN, true };
  // Attr names (CleanupAttr, GuardedByAttr, etc)
  static const TerminalColor AttrColor = { raw_ostream::BLUE, true };
  // Statement names (DeclStmt, ImplicitCastExpr, etc)
  static const TerminalColor StmtColor = { raw_ostream::MAGENTA, true };
  // Comment names (FullComment, ParagraphComment, TextComment, etc)
  static const TerminalColor CommentColor = { raw_ostream::BLUE, false };

  // Type names (int, float, etc, plus user defined types)
  static const TerminalColor TypeColor = { raw_ostream::GREEN, false };

  // Pointer address
  static const TerminalColor AddressColor = { raw_ostream::YELLOW, false };
  // Source locations
  static const TerminalColor LocationColor = { raw_ostream::YELLOW, false };

  // lvalue/xvalue
  static const TerminalColor ValueKindColor = { raw_ostream::CYAN, false };
  // bitfield/objcproperty/objcsubscript/vectorcomponent
  static const TerminalColor ObjectKindColor = { raw_ostream::CYAN, false };

  // Null statements
  static const TerminalColor NullColor = { raw_ostream::BLUE, false };

  // Undeserialized entities
  static const TerminalColor UndeserializedColor = { raw_ostream::GREEN, true };

  // CastKind from CastExpr's
  static const TerminalColor CastColor = { raw_ostream::RED, false };

  // Value of the statement
  static const TerminalColor ValueColor = { raw_ostream::CYAN, true };
  // Decl names
  static const TerminalColor DeclNameColor = { raw_ostream::CYAN, true };

  // Indents ( `, -. | )
  static const TerminalColor IndentColor = { raw_ostream::BLUE, false };

  class ASTDumper
      : public ConstDeclVisitor<ASTDumper>, public ConstStmtVisitor<ASTDumper>,
        public ConstCommentVisitor<ASTDumper> {
    raw_ostream &OS;
    const CommandTraits *Traits;
    const SourceManager *SM;
    bool IsFirstLine;

    // Indicates whether more child are expected at the current tree depth
    enum IndentType { IT_Child, IT_LastChild };

    /// Indents[i] indicates if another child exists at level i.
    /// Used by Indent() to print the tree structure. 
    llvm::SmallVector<IndentType, 32> Indents;

    /// Indicates that more children will be needed at this indent level.
    /// If true, prevents lastChild() from marking the node as the last child.
    /// This is used when there are multiple collections of children to be
    /// dumped as well as during conditional node dumping.
    bool MoreChildren;

    /// Keep track of the last location we print out so that we can
    /// print out deltas from then on out.
    const char *LastLocFilename;
    unsigned LastLocLine;

    /// The \c FullComment parent of the comment being dumped.
    const FullComment *FC;

    bool ShowColors;

    class IndentScope {
      ASTDumper &Dumper;
      // Preserve the Dumper's MoreChildren value from the previous IndentScope
      bool MoreChildren;
    public:
      IndentScope(ASTDumper &Dumper) : Dumper(Dumper) {
        MoreChildren = Dumper.hasMoreChildren();
        Dumper.setMoreChildren(false);
        Dumper.indent();
      }
      ~IndentScope() {
        Dumper.setMoreChildren(MoreChildren);
        Dumper.unindent();
      }
    };

    class ColorScope {
      ASTDumper &Dumper;
    public:
      ColorScope(ASTDumper &Dumper, TerminalColor Color)
        : Dumper(Dumper) {
        if (Dumper.ShowColors)
          Dumper.OS.changeColor(Color.Color, Color.Bold);
      }
      ~ColorScope() {
        if (Dumper.ShowColors)
          Dumper.OS.resetColor();
      }
    };

  public:
    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM)
      : OS(OS), Traits(Traits), SM(SM), IsFirstLine(true), MoreChildren(false),
        LastLocFilename(""), LastLocLine(~0U), FC(0),
        ShowColors(SM && SM->getDiagnostics().getShowColors()) { }

    ASTDumper(raw_ostream &OS, const CommandTraits *Traits,
              const SourceManager *SM, bool ShowColors)
      : OS(OS), Traits(Traits), SM(SM), IsFirstLine(true), MoreChildren(false),
        LastLocFilename(""), LastLocLine(~0U),
        ShowColors(ShowColors) { }

    ~ASTDumper() {
      OS << "\n";
    }

    void dumpDecl(const Decl *D);
    void dumpStmt(const Stmt *S);
    void dumpFullComment(const FullComment *C);

    // Formatting
    void indent();
    void unindent();
    void lastChild();
    bool hasMoreChildren();
    void setMoreChildren(bool Value);

    // Utilities
    void dumpPointer(const void *Ptr);
    void dumpSourceRange(SourceRange R);
    void dumpLocation(SourceLocation Loc);
    void dumpBareType(QualType T);
    void dumpType(QualType T);
    void dumpBareDeclRef(const Decl *Node);
    void dumpDeclRef(const Decl *Node, const char *Label = 0);
    void dumpName(const NamedDecl *D);
    bool hasNodes(const DeclContext *DC);
    void dumpDeclContext(const DeclContext *DC);
    void dumpLookups(const DeclContext *DC);
    void dumpAttr(const Attr *A);

    // C++ Utilities
    void dumpAccessSpecifier(AccessSpecifier AS);
    void dumpCXXCtorInitializer(const CXXCtorInitializer *Init);
    void dumpTemplateParameters(const TemplateParameterList *TPL);
    void dumpTemplateArgumentListInfo(const TemplateArgumentListInfo &TALI);
    void dumpTemplateArgumentLoc(const TemplateArgumentLoc &A);
    void dumpTemplateArgumentList(const TemplateArgumentList &TAL);
    void dumpTemplateArgument(const TemplateArgument &A,
                              SourceRange R = SourceRange());

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
    void VisitFileScopeAsmDecl(const FileScopeAsmDecl *D);
    void VisitImportDecl(const ImportDecl *D);

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
    void VisitLinkageSpecDecl(const LinkageSpecDecl *D);
    void VisitAccessSpecDecl(const AccessSpecDecl *D);
    void VisitFriendDecl(const FriendDecl *D);

    // ObjC Decls
    void VisitObjCIvarDecl(const ObjCIvarDecl *D);
    void VisitObjCMethodDecl(const ObjCMethodDecl *D);
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
    void VisitStmt(const Stmt *Node);
    void VisitDeclStmt(const DeclStmt *Node);
    void VisitAttributedStmt(const AttributedStmt *Node);
    void VisitLabelStmt(const LabelStmt *Node);
    void VisitGotoStmt(const GotoStmt *Node);
    void VisitCXXCatchStmt(const CXXCatchStmt *Node);

    // Exprs
    void VisitExpr(const Expr *Node);
    void VisitCastExpr(const CastExpr *Node);
    void VisitDeclRefExpr(const DeclRefExpr *Node);
    void VisitPredefinedExpr(const PredefinedExpr *Node);
    void VisitCharacterLiteral(const CharacterLiteral *Node);
    void VisitIntegerLiteral(const IntegerLiteral *Node);
    void VisitFloatingLiteral(const FloatingLiteral *Node);
    void VisitStringLiteral(const StringLiteral *Str);
    void VisitUnaryOperator(const UnaryOperator *Node);
    void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Node);
    void VisitMemberExpr(const MemberExpr *Node);
    void VisitExtVectorElementExpr(const ExtVectorElementExpr *Node);
    void VisitBinaryOperator(const BinaryOperator *Node);
    void VisitCompoundAssignOperator(const CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(const AddrLabelExpr *Node);
    void VisitBlockExpr(const BlockExpr *Node);
    void VisitOpaqueValueExpr(const OpaqueValueExpr *Node);

    // C++
    void VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(const CXXThisExpr *Node);
    void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *Node);
    void VisitCXXConstructExpr(const CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Node);
    void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Node);
    void VisitExprWithCleanups(const ExprWithCleanups *Node);
    void VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *Node);
    void dumpCXXTemporary(const CXXTemporary *Temporary);
    void VisitLambdaExpr(const LambdaExpr *Node) {
      VisitExpr(Node);
      dumpDecl(Node->getLambdaClass());
    }

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
    const char *getCommandName(unsigned CommandID);
    void dumpComment(const Comment *C);

    // Inline comments.
    void visitTextComment(const TextComment *C);
    void visitInlineCommandComment(const InlineCommandComment *C);
    void visitHTMLStartTagComment(const HTMLStartTagComment *C);
    void visitHTMLEndTagComment(const HTMLEndTagComment *C);

    // Block comments.
    void visitBlockCommandComment(const BlockCommandComment *C);
    void visitParamCommandComment(const ParamCommandComment *C);
    void visitTParamCommandComment(const TParamCommandComment *C);
    void visitVerbatimBlockComment(const VerbatimBlockComment *C);
    void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
    void visitVerbatimLineComment(const VerbatimLineComment *C);
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

// Print out the appropriate tree structure using the Indents vector.
// Example of tree and the Indents vector at each level.
// A        { }
// |-B      { IT_Child }
// | `-C    { IT_Child,     IT_LastChild }
// `-D      { IT_LastChild }
//   |-E    { IT_LastChild, IT_Child }
//   `-F    { IT_LastChild, IT_LastChild }
// Type            non-last element, last element
// IT_Child        "| "              "|-"
// IT_LastChild    "  "              "`-"
void ASTDumper::indent() {
  if (IsFirstLine)
    IsFirstLine = false;
  else
    OS << "\n";

  ColorScope Color(*this, IndentColor);
  for (SmallVectorImpl<IndentType>::const_iterator I = Indents.begin(),
                                                   E = Indents.end();
       I != E; ++I) {
    switch (*I) {
    case IT_Child:
      if (I == E - 1)
        OS << "|-";
      else
        OS << "| ";
      continue;
    case IT_LastChild:
      if (I == E - 1)
        OS << "`-";
      else
        OS << "  ";
      continue;
    }
    llvm_unreachable("Invalid IndentType");
  }
  Indents.push_back(IT_Child);
}

void ASTDumper::unindent() {
  Indents.pop_back();
}

// Call before each potential last child node is to be dumped.  If MoreChildren
// is false, then this is the last child, otherwise treat as a regular node.
void ASTDumper::lastChild() {
  if (!hasMoreChildren())
    Indents.back() = IT_LastChild;
}

// MoreChildren should be set before calling another function that may print
// additional nodes to prevent conflicting final child nodes.
bool ASTDumper::hasMoreChildren() {
  return MoreChildren;
}

void ASTDumper::setMoreChildren(bool Value) {
  MoreChildren = Value;
}

void ASTDumper::dumpPointer(const void *Ptr) {
  ColorScope Color(*this, AddressColor);
  OS << ' ' << Ptr;
}

void ASTDumper::dumpLocation(SourceLocation Loc) {
  ColorScope Color(*this, LocationColor);
  SourceLocation SpellingLoc = SM->getSpellingLoc(Loc);

  // The general format we print out is filename:line:col, but we drop pieces
  // that haven't changed since the last loc printed.
  PresumedLoc PLoc = SM->getPresumedLoc(SpellingLoc);

  if (PLoc.isInvalid()) {
    OS << "<invalid sloc>";
    return;
  }

  if (strcmp(PLoc.getFilename(), LastLocFilename) != 0) {
    OS << PLoc.getFilename() << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();
    LastLocFilename = PLoc.getFilename();
    LastLocLine = PLoc.getLine();
  } else if (PLoc.getLine() != LastLocLine) {
    OS << "line" << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();
    LastLocLine = PLoc.getLine();
  } else {
    OS << "col" << ':' << PLoc.getColumn();
  }
}

void ASTDumper::dumpSourceRange(SourceRange R) {
  // Can't translate locations if a SourceManager isn't available.
  if (!SM)
    return;

  OS << " <";
  dumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    dumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>

}

void ASTDumper::dumpBareType(QualType T) {
  ColorScope Color(*this, TypeColor);
  
  SplitQualType T_split = T.split();
  OS << "'" << QualType::getAsString(T_split) << "'";

  if (!T.isNull()) {
    // If the type is sugared, also dump a (shallow) desugared type.
    SplitQualType D_split = T.getSplitDesugaredType();
    if (T_split != D_split)
      OS << ":'" << QualType::getAsString(D_split) << "'";
  }
}

void ASTDumper::dumpType(QualType T) {
  OS << ' ';
  dumpBareType(T);
}

void ASTDumper::dumpBareDeclRef(const Decl *D) {
  {
    ColorScope Color(*this, DeclKindNameColor);
    OS << D->getDeclKindName();
  }
  dumpPointer(D);

  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    ColorScope Color(*this, DeclNameColor);
    OS << " '" << ND->getDeclName() << '\'';
  }

  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D))
    dumpType(VD->getType());
}

void ASTDumper::dumpDeclRef(const Decl *D, const char *Label) {
  if (!D)
    return;

  IndentScope Indent(*this);
  if (Label)
    OS << Label << ' ';
  dumpBareDeclRef(D);
}

void ASTDumper::dumpName(const NamedDecl *ND) {
  if (ND->getDeclName()) {
    ColorScope Color(*this, DeclNameColor);
    OS << ' ' << ND->getNameAsString();
  }
}

bool ASTDumper::hasNodes(const DeclContext *DC) {
  if (!DC)
    return false;

  return DC->hasExternalLexicalStorage() ||
         DC->noload_decls_begin() != DC->noload_decls_end();
}

void ASTDumper::dumpDeclContext(const DeclContext *DC) {
  if (!DC)
    return;
  bool HasUndeserializedDecls = DC->hasExternalLexicalStorage();
  for (DeclContext::decl_iterator I = DC->noload_decls_begin(),
                                  E = DC->noload_decls_end();
       I != E; ++I) {
    DeclContext::decl_iterator Next = I;
    ++Next;
    if (Next == E && !HasUndeserializedDecls)
      lastChild();
    dumpDecl(*I);
  }
  if (HasUndeserializedDecls) {
    lastChild();
    IndentScope Indent(*this);
    ColorScope Color(*this, UndeserializedColor);
    OS << "<undeserialized declarations>";
  }
}

void ASTDumper::dumpLookups(const DeclContext *DC) {
  IndentScope Indent(*this);

  OS << "StoredDeclsMap ";
  dumpBareDeclRef(cast<Decl>(DC));

  const DeclContext *Primary = DC->getPrimaryContext();
  if (Primary != DC) {
    OS << " primary";
    dumpPointer(cast<Decl>(Primary));
  }

  bool HasUndeserializedLookups = Primary->hasExternalVisibleStorage();

  DeclContext::all_lookups_iterator I = Primary->noload_lookups_begin(),
                                    E = Primary->noload_lookups_end();
  while (I != E) {
    DeclarationName Name = I.getLookupName();
    DeclContextLookupResult R = *I++;
    if (I == E && !HasUndeserializedLookups)
      lastChild();

    IndentScope Indent(*this);
    OS << "DeclarationName ";
    {
      ColorScope Color(*this, DeclNameColor);
      OS << '\'' << Name << '\'';
    }

    for (DeclContextLookupResult::iterator RI = R.begin(), RE = R.end();
         RI != RE; ++RI) {
      if (RI + 1 == RE)
        lastChild();
      dumpDeclRef(*RI);
      if ((*RI)->isHidden())
        OS << " hidden";
    }
  }

  if (HasUndeserializedLookups) {
    lastChild();
    IndentScope Indent(*this);
    ColorScope Color(*this, UndeserializedColor);
    OS << "<undeserialized lookups>";
  }
}

void ASTDumper::dumpAttr(const Attr *A) {
  IndentScope Indent(*this);
  {
    ColorScope Color(*this, AttrColor);
    switch (A->getKind()) {
#define ATTR(X) case attr::X: OS << #X; break;
#include "clang/Basic/AttrList.inc"
    default: llvm_unreachable("unexpected attribute kind");
    }
    OS << "Attr";
  }
  dumpPointer(A);
  dumpSourceRange(A->getRange());
#include "clang/AST/AttrDump.inc"
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

void ASTDumper::dumpAccessSpecifier(AccessSpecifier AS) {
  switch (AS) {
  case AS_none:
    break;
  case AS_public:
    OS << "public";
    break;
  case AS_protected:
    OS << "protected";
    break;
  case AS_private:
    OS << "private";
    break;
  }
}

void ASTDumper::dumpCXXCtorInitializer(const CXXCtorInitializer *Init) {
  IndentScope Indent(*this);
  OS << "CXXCtorInitializer";
  if (Init->isAnyMemberInitializer()) {
    OS << ' ';
    dumpBareDeclRef(Init->getAnyMember());
  } else {
    dumpType(QualType(Init->getBaseClass(), 0));
  }
  dumpStmt(Init->getInit());
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
  for (unsigned i = 0, e = TALI.size(); i < e; ++i) {
    if (i + 1 == e)
      lastChild();
    dumpTemplateArgumentLoc(TALI[i]);
  }
}

void ASTDumper::dumpTemplateArgumentLoc(const TemplateArgumentLoc &A) {
  dumpTemplateArgument(A.getArgument(), A.getSourceRange());
}

void ASTDumper::dumpTemplateArgumentList(const TemplateArgumentList &TAL) {
  for (unsigned i = 0, e = TAL.size(); i < e; ++i)
    dumpTemplateArgument(TAL[i]);
}

void ASTDumper::dumpTemplateArgument(const TemplateArgument &A, SourceRange R) {
  IndentScope Indent(*this);
  OS << "TemplateArgument";
  if (R.isValid())
    dumpSourceRange(R);

  switch (A.getKind()) {
  case TemplateArgument::Null:
    OS << " null";
    break;
  case TemplateArgument::Type:
    OS << " type";
    lastChild();
    dumpType(A.getAsType());
    break;
  case TemplateArgument::Declaration:
    OS << " decl";
    lastChild();
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
    OS << " template expansion";
    A.getAsTemplateOrTemplatePattern().dump(OS);
    break;
  case TemplateArgument::Expression:
    OS << " expr";
    lastChild();
    dumpStmt(A.getAsExpr());
    break;
  case TemplateArgument::Pack:
    OS << " pack";
    for (TemplateArgument::pack_iterator I = A.pack_begin(), E = A.pack_end();
         I != E; ++I) {
      if (I + 1 == E)
        lastChild();
      dumpTemplateArgument(*I);
    }
    break;
  }
}

//===----------------------------------------------------------------------===//
//  Decl dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpDecl(const Decl *D) {
  IndentScope Indent(*this);

  if (!D) {
    ColorScope Color(*this, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(*this, DeclKindNameColor);
    OS << D->getDeclKindName() << "Decl";
  }
  dumpPointer(D);
  if (D->getLexicalDeclContext() != D->getDeclContext())
    OS << " parent " << cast<Decl>(D->getDeclContext());
  dumpPreviousDecl(OS, D);
  dumpSourceRange(D->getSourceRange());
  if (Module *M = D->getOwningModule())
    OS << " in " << M->getFullModuleName();
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
    if (ND->isHidden())
      OS << " hidden";

  bool HasAttrs = D->attr_begin() != D->attr_end();
  const FullComment *Comment =
      D->getASTContext().getLocalCommentForDeclUncached(D);
  // Decls within functions are visited by the body
  bool HasDeclContext = !isa<FunctionDecl>(*D) && !isa<ObjCMethodDecl>(*D) &&
                         hasNodes(dyn_cast<DeclContext>(D));

  setMoreChildren(HasAttrs || Comment || HasDeclContext);
  ConstDeclVisitor<ASTDumper>::Visit(D);

  setMoreChildren(Comment || HasDeclContext);
  for (Decl::attr_iterator I = D->attr_begin(), E = D->attr_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpAttr(*I);
  }

  setMoreChildren(HasDeclContext);
  lastChild();
  dumpFullComment(Comment);

  if (D->isInvalidDecl())
    OS << " invalid";

  setMoreChildren(false);
  if (HasDeclContext)
    dumpDeclContext(cast<DeclContext>(D));
}

void ASTDumper::VisitLabelDecl(const LabelDecl *D) {
  dumpName(D);
}

void ASTDumper::VisitTypedefDecl(const TypedefDecl *D) {
  dumpName(D);
  dumpType(D->getUnderlyingType());
  if (D->isModulePrivate())
    OS << " __module_private__";
}

void ASTDumper::VisitEnumDecl(const EnumDecl *D) {
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      OS << " class";
    else
      OS << " struct";
  }
  dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isFixed())
    dumpType(D->getIntegerType());
}

void ASTDumper::VisitRecordDecl(const RecordDecl *D) {
  OS << ' ' << D->getKindName();
  dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isCompleteDefinition())
    OS << " definition";
}

void ASTDumper::VisitEnumConstantDecl(const EnumConstantDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (const Expr *Init = D->getInitExpr()) {
    lastChild();
    dumpStmt(Init);
  }
}

void ASTDumper::VisitIndirectFieldDecl(const IndirectFieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  for (IndirectFieldDecl::chain_iterator I = D->chain_begin(),
                                         E = D->chain_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDeclRef(*I);
  }
}

void ASTDumper::VisitFunctionDecl(const FunctionDecl *D) {
  dumpName(D);
  dumpType(D->getType());

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
  else if (D->isDeletedAsWritten())
    OS << " delete";

  if (const FunctionProtoType *FPT = D->getType()->getAs<FunctionProtoType>()) {
    FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
    switch (EPI.ExceptionSpecType) {
    default: break;
    case EST_Unevaluated:
      OS << " noexcept-unevaluated " << EPI.ExceptionSpecDecl;
      break;
    case EST_Uninstantiated:
      OS << " noexcept-uninstantiated " << EPI.ExceptionSpecTemplate;
      break;
    }
  }

  bool OldMoreChildren = hasMoreChildren();
  const FunctionTemplateSpecializationInfo *FTSI =
      D->getTemplateSpecializationInfo();
  bool HasTemplateSpecialization = FTSI;

  bool HasNamedDecls = D->getDeclsInPrototypeScope().begin() !=
                       D->getDeclsInPrototypeScope().end();

  bool HasFunctionDecls = D->param_begin() != D->param_end();

  const CXXConstructorDecl *C = dyn_cast<CXXConstructorDecl>(D);
  bool HasCtorInitializers = C && C->init_begin() != C->init_end();

  bool HasDeclarationBody = D->doesThisDeclarationHaveABody();

  setMoreChildren(OldMoreChildren || HasNamedDecls || HasFunctionDecls ||
                  HasCtorInitializers || HasDeclarationBody);
  if (HasTemplateSpecialization) {
    lastChild();
    dumpTemplateArgumentList(*FTSI->TemplateArguments);
  }

  setMoreChildren(OldMoreChildren || HasFunctionDecls ||
                  HasCtorInitializers || HasDeclarationBody);
  for (ArrayRef<NamedDecl *>::iterator
       I = D->getDeclsInPrototypeScope().begin(),
       E = D->getDeclsInPrototypeScope().end(); I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDecl(*I);
  }

  setMoreChildren(OldMoreChildren || HasCtorInitializers || HasDeclarationBody);
  for (FunctionDecl::param_const_iterator I = D->param_begin(),
                                          E = D->param_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDecl(*I);
  }
 
  setMoreChildren(OldMoreChildren || HasDeclarationBody);
  if (HasCtorInitializers)
    for (CXXConstructorDecl::init_const_iterator I = C->init_begin(),
                                                 E = C->init_end();
         I != E; ++I) {
      if (I + 1 == E)
        lastChild();
      dumpCXXCtorInitializer(*I);
  }

  setMoreChildren(OldMoreChildren);
  if (HasDeclarationBody) {
    lastChild();
    dumpStmt(D->getBody());
  }
}

void ASTDumper::VisitFieldDecl(const FieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (D->isMutable())
    OS << " mutable";
  if (D->isModulePrivate())
    OS << " __module_private__";

  bool OldMoreChildren = hasMoreChildren();
  bool IsBitField = D->isBitField();
  Expr *Init = D->getInClassInitializer();
  bool HasInit = Init;

  setMoreChildren(OldMoreChildren || HasInit);
  if (IsBitField) {
    lastChild();
    dumpStmt(D->getBitWidth());
  }
  setMoreChildren(OldMoreChildren);
  if (HasInit) {
    lastChild();
    dumpStmt(Init);
  }
}

void ASTDumper::VisitVarDecl(const VarDecl *D) {
  dumpName(D);
  dumpType(D->getType());
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
  if (D->hasInit()) {
    lastChild();
    dumpStmt(D->getInit());
  }
}

void ASTDumper::VisitFileScopeAsmDecl(const FileScopeAsmDecl *D) {
  lastChild();
  dumpStmt(D->getAsmString());
}

void ASTDumper::VisitImportDecl(const ImportDecl *D) {
  OS << ' ' << D->getImportedModule()->getFullModuleName();
}

//===----------------------------------------------------------------------===//
// C++ Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitNamespaceDecl(const NamespaceDecl *D) {
  dumpName(D);
  if (D->isInline())
    OS << " inline";
  if (!D->isOriginalNamespace())
    dumpDeclRef(D->getOriginalNamespace(), "original");
}

void ASTDumper::VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
  OS << ' ';
  dumpBareDeclRef(D->getNominatedNamespace());
}

void ASTDumper::VisitNamespaceAliasDecl(const NamespaceAliasDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getAliasedNamespace());
}

void ASTDumper::VisitTypeAliasDecl(const TypeAliasDecl *D) {
  dumpName(D);
  dumpType(D->getUnderlyingType());
}

void ASTDumper::VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
}

void ASTDumper::VisitCXXRecordDecl(const CXXRecordDecl *D) {
  VisitRecordDecl(D);
  if (!D->isCompleteDefinition())
    return;

  for (CXXRecordDecl::base_class_const_iterator I = D->bases_begin(),
                                                E = D->bases_end();
       I != E; ++I) {
    IndentScope Indent(*this);
    if (I->isVirtual())
      OS << "virtual ";
    dumpAccessSpecifier(I->getAccessSpecifier());
    dumpType(I->getType());
    if (I->isPackExpansion())
      OS << "...";
  }
}

void ASTDumper::VisitStaticAssertDecl(const StaticAssertDecl *D) {
  dumpStmt(D->getAssertExpr());
  lastChild();
  dumpStmt(D->getMessage());
}

void ASTDumper::VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
  for (FunctionTemplateDecl::spec_iterator I = D->spec_begin(),
                                           E = D->spec_end();
       I != E; ++I) {
    FunctionTemplateDecl::spec_iterator Next = I;
    ++Next;
    if (Next == E)
      lastChild();
    switch (I->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      if (D == D->getCanonicalDecl())
        dumpDecl(*I);
      else
        dumpDeclRef(*I);
      break;
    case TSK_ExplicitSpecialization:
      dumpDeclRef(*I);
      break;
    }
  }
}

void ASTDumper::VisitClassTemplateDecl(const ClassTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());

  ClassTemplateDecl::spec_iterator I = D->spec_begin();
  ClassTemplateDecl::spec_iterator E = D->spec_end();
  if (I == E)
    lastChild();
  dumpDecl(D->getTemplatedDecl());
  for (; I != E; ++I) {
    ClassTemplateDecl::spec_iterator Next = I;
    ++Next;
    if (Next == E)
      lastChild();
    switch (I->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      if (D == D->getCanonicalDecl())
        dumpDecl(*I);
      else
        dumpDeclRef(*I);
      break;
    case TSK_ExplicitSpecialization:
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      dumpDeclRef(*I);
      break;
    }
  }
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
  dumpDeclRef(D->getSpecialization());
  if (D->hasExplicitTemplateArgs())
    dumpTemplateArgumentListInfo(D->templateArgs());
}

void ASTDumper::VisitVarTemplateDecl(const VarTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());

  VarTemplateDecl::spec_iterator I = D->spec_begin();
  VarTemplateDecl::spec_iterator E = D->spec_end();
  if (I == E)
    lastChild();
  dumpDecl(D->getTemplatedDecl());
  for (; I != E; ++I) {
    VarTemplateDecl::spec_iterator Next = I;
    ++Next;
    if (Next == E)
      lastChild();
    switch (I->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      if (D == D->getCanonicalDecl())
        dumpDecl(*I);
      else
        dumpDeclRef(*I);
      break;
    case TSK_ExplicitSpecialization:
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      dumpDeclRef(*I);
      break;
    }
  }
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
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
  if (D->hasDefaultArgument())
    dumpType(D->getDefaultArgument());
}

void ASTDumper::VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *D) {
  dumpType(D->getType());
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
  if (D->hasDefaultArgument())
    dumpStmt(D->getDefaultArgument());
}

void ASTDumper::VisitTemplateTemplateParmDecl(
    const TemplateTemplateParmDecl *D) {
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  if (D->hasDefaultArgument())
    dumpTemplateArgumentLoc(D->getDefaultArgument());
}

void ASTDumper::VisitUsingDecl(const UsingDecl *D) {
  OS << ' ';
  D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void ASTDumper::VisitUnresolvedUsingTypenameDecl(
    const UnresolvedUsingTypenameDecl *D) {
  OS << ' ';
  D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void ASTDumper::VisitUnresolvedUsingValueDecl(const UnresolvedUsingValueDecl *D) {
  OS << ' ';
  D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
  dumpType(D->getType());
}

void ASTDumper::VisitUsingShadowDecl(const UsingShadowDecl *D) {
  OS << ' ';
  dumpBareDeclRef(D->getTargetDecl());
}

void ASTDumper::VisitLinkageSpecDecl(const LinkageSpecDecl *D) {
  switch (D->getLanguage()) {
  case LinkageSpecDecl::lang_c: OS << " C"; break;
  case LinkageSpecDecl::lang_cxx: OS << " C++"; break;
  }
}

void ASTDumper::VisitAccessSpecDecl(const AccessSpecDecl *D) {
  OS << ' ';
  dumpAccessSpecifier(D->getAccess());
}

void ASTDumper::VisitFriendDecl(const FriendDecl *D) {
  lastChild();
  if (TypeSourceInfo *T = D->getFriendType())
    dumpType(T->getType());
  else
    dumpDecl(D->getFriendDecl());
}

//===----------------------------------------------------------------------===//
// Obj-C Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCIvarDecl(const ObjCIvarDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (D->getSynthesize())
    OS << " synthesize";
  if (D->getBackingIvarReferencedInAccessor())
    OS << " BackingIvarReferencedInAccessor";

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
  dumpName(D);
  dumpType(D->getResultType());

  bool OldMoreChildren = hasMoreChildren();
  bool IsVariadic = D->isVariadic();
  bool HasBody = D->hasBody();

  setMoreChildren(OldMoreChildren || IsVariadic || HasBody);
  if (D->isThisDeclarationADefinition()) {
    lastChild();
    dumpDeclContext(D);
  } else {
    for (ObjCMethodDecl::param_const_iterator I = D->param_begin(),
                                              E = D->param_end();
         I != E; ++I) {
      if (I + 1 == E)
        lastChild();
      dumpDecl(*I);
    }
  }

  setMoreChildren(OldMoreChildren || HasBody);
  if (IsVariadic) {
    lastChild();
    IndentScope Indent(*this);
    OS << "...";
  }

  setMoreChildren(OldMoreChildren);
  if (HasBody) {
    lastChild();
    dumpStmt(D->getBody());
  }
}

void ASTDumper::VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
  if (D->protocol_begin() == D->protocol_end())
    lastChild();
  dumpDeclRef(D->getImplementation());
  for (ObjCCategoryDecl::protocol_iterator I = D->protocol_begin(),
                                           E = D->protocol_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDeclRef(*I);
  }
}

void ASTDumper::VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
  lastChild();
  dumpDeclRef(D->getCategoryDecl());
}

void ASTDumper::VisitObjCProtocolDecl(const ObjCProtocolDecl *D) {
  dumpName(D);
  for (ObjCProtocolDecl::protocol_iterator I = D->protocol_begin(),
                                           E = D->protocol_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDeclRef(*I);
  }
}

void ASTDumper::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");
  if (D->protocol_begin() == D->protocol_end())
    lastChild();
  dumpDeclRef(D->getImplementation());
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
                                            E = D->protocol_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDeclRef(*I);
  }
}

void ASTDumper::VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");
  if (D->init_begin() == D->init_end())
    lastChild();
  dumpDeclRef(D->getClassInterface());
  for (ObjCImplementationDecl::init_const_iterator I = D->init_begin(),
                                                   E = D->init_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpCXXCtorInitializer(*I);
  }
}

void ASTDumper::VisitObjCCompatibleAliasDecl(const ObjCCompatibleAliasDecl *D) {
  dumpName(D);
  lastChild();
  dumpDeclRef(D->getClassInterface());
}

void ASTDumper::VisitObjCPropertyDecl(const ObjCPropertyDecl *D) {
  dumpName(D);
  dumpType(D->getType());

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
    if (Attrs & ObjCPropertyDecl::OBJC_PR_getter) {
      if (!(Attrs & ObjCPropertyDecl::OBJC_PR_setter))
        lastChild();
      dumpDeclRef(D->getGetterMethodDecl(), "getter");
    }
    if (Attrs & ObjCPropertyDecl::OBJC_PR_setter) {
      lastChild();
      dumpDeclRef(D->getSetterMethodDecl(), "setter");
    }
  }
}

void ASTDumper::VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D) {
  dumpName(D->getPropertyDecl());
  if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    OS << " synthesize";
  else
    OS << " dynamic";
  dumpDeclRef(D->getPropertyDecl());
  lastChild();
  dumpDeclRef(D->getPropertyIvarDecl());
}

void ASTDumper::VisitBlockDecl(const BlockDecl *D) {
  for (BlockDecl::param_const_iterator I = D->param_begin(), E = D->param_end();
       I != E; ++I)
    dumpDecl(*I);

  if (D->isVariadic()) {
    IndentScope Indent(*this);
    OS << "...";
  }

  if (D->capturesCXXThis()) {
    IndentScope Indent(*this);
    OS << "capture this";
  }
  for (BlockDecl::capture_iterator I = D->capture_begin(), E = D->capture_end();
       I != E; ++I) {
    IndentScope Indent(*this);
    OS << "capture";
    if (I->isByRef())
      OS << " byref";
    if (I->isNested())
      OS << " nested";
    if (I->getVariable()) {
      OS << ' ';
      dumpBareDeclRef(I->getVariable());
    }
    if (I->hasCopyExpr())
      dumpStmt(I->getCopyExpr());
  }
  lastChild();
  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
//  Stmt dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpStmt(const Stmt *S) {
  IndentScope Indent(*this);

  if (!S) {
    ColorScope Color(*this, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  if (const DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    VisitDeclStmt(DS);
    return;
  }

  setMoreChildren(!S->children().empty());
  ConstStmtVisitor<ASTDumper>::Visit(S);
  setMoreChildren(false);
  for (Stmt::const_child_range CI = S->children(); CI; ++CI) {
    Stmt::const_child_range Next = CI;
    ++Next;
    if (!Next)
      lastChild();
    dumpStmt(*CI);
  }
}

void ASTDumper::VisitStmt(const Stmt *Node) {
  {   
    ColorScope Color(*this, StmtColor);
    OS << Node->getStmtClassName();
  }
  dumpPointer(Node);
  dumpSourceRange(Node->getSourceRange());
}

void ASTDumper::VisitDeclStmt(const DeclStmt *Node) {
  VisitStmt(Node);
  for (DeclStmt::const_decl_iterator I = Node->decl_begin(),
                                     E = Node->decl_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpDecl(*I);
  }
}

void ASTDumper::VisitAttributedStmt(const AttributedStmt *Node) {
  VisitStmt(Node);
  for (ArrayRef<const Attr *>::iterator I = Node->getAttrs().begin(),
                                        E = Node->getAttrs().end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpAttr(*I);
  }
}

void ASTDumper::VisitLabelStmt(const LabelStmt *Node) {
  VisitStmt(Node);
  OS << " '" << Node->getName() << "'";
}

void ASTDumper::VisitGotoStmt(const GotoStmt *Node) {
  VisitStmt(Node);
  OS << " '" << Node->getLabel()->getName() << "'";
  dumpPointer(Node->getLabel());
}

void ASTDumper::VisitCXXCatchStmt(const CXXCatchStmt *Node) {
  VisitStmt(Node);
  dumpDecl(Node->getExceptionDecl());
}

//===----------------------------------------------------------------------===//
//  Expr dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::VisitExpr(const Expr *Node) {
  VisitStmt(Node);
  dumpType(Node->getType());

  {
    ColorScope Color(*this, ValueKindColor);
    switch (Node->getValueKind()) {
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
    ColorScope Color(*this, ObjectKindColor);
    switch (Node->getObjectKind()) {
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

void ASTDumper::VisitCastExpr(const CastExpr *Node) {
  VisitExpr(Node);
  OS << " <";
  {
    ColorScope Color(*this, CastColor);
    OS << Node->getCastKindName();
  }
  dumpBasePath(OS, Node);
  OS << ">";
}

void ASTDumper::VisitDeclRefExpr(const DeclRefExpr *Node) {
  VisitExpr(Node);

  OS << " ";
  dumpBareDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    OS << " (";
    dumpBareDeclRef(Node->getFoundDecl());
    OS << ")";
  }
}

void ASTDumper::VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *Node) {
  VisitExpr(Node);
  OS << " (";
  if (!Node->requiresADL())
    OS << "no ";
  OS << "ADL) = '" << Node->getName() << '\'';

  UnresolvedLookupExpr::decls_iterator
    I = Node->decls_begin(), E = Node->decls_end();
  if (I == E)
    OS << " empty";
  for (; I != E; ++I)
    dumpPointer(*I);
}

void ASTDumper::VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Node) {
  VisitExpr(Node);

  {
    ColorScope Color(*this, DeclKindNameColor);
    OS << " " << Node->getDecl()->getDeclKindName() << "Decl";
  }
  OS << "='" << *Node->getDecl() << "'";
  dumpPointer(Node->getDecl());
  if (Node->isFreeIvar())
    OS << " isFreeIvar";
}

void ASTDumper::VisitPredefinedExpr(const PredefinedExpr *Node) {
  VisitExpr(Node);
  switch (Node->getIdentType()) {
  default: llvm_unreachable("unknown case");
  case PredefinedExpr::Func:           OS <<  " __func__"; break;
  case PredefinedExpr::Function:       OS <<  " __FUNCTION__"; break;
  case PredefinedExpr::FuncDName:      OS <<  " __FUNCDNAME__"; break;
  case PredefinedExpr::LFunction:      OS <<  " L__FUNCTION__"; break;
  case PredefinedExpr::PrettyFunction: OS <<  " __PRETTY_FUNCTION__";break;
  }
}

void ASTDumper::VisitCharacterLiteral(const CharacterLiteral *Node) {
  VisitExpr(Node);
  ColorScope Color(*this, ValueColor);
  OS << " " << Node->getValue();
}

void ASTDumper::VisitIntegerLiteral(const IntegerLiteral *Node) {
  VisitExpr(Node);

  bool isSigned = Node->getType()->isSignedIntegerType();
  ColorScope Color(*this, ValueColor);
  OS << " " << Node->getValue().toString(10, isSigned);
}

void ASTDumper::VisitFloatingLiteral(const FloatingLiteral *Node) {
  VisitExpr(Node);
  ColorScope Color(*this, ValueColor);
  OS << " " << Node->getValueAsApproximateDouble();
}

void ASTDumper::VisitStringLiteral(const StringLiteral *Str) {
  VisitExpr(Str);
  ColorScope Color(*this, ValueColor);
  OS << " ";
  Str->outputString(OS);
}

void ASTDumper::VisitUnaryOperator(const UnaryOperator *Node) {
  VisitExpr(Node);
  OS << " " << (Node->isPostfix() ? "postfix" : "prefix")
     << " '" << UnaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}

void ASTDumper::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *Node) {
  VisitExpr(Node);
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
  }
  if (Node->isArgumentType())
    dumpType(Node->getArgumentType());
}

void ASTDumper::VisitMemberExpr(const MemberExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->isArrow() ? "->" : ".") << *Node->getMemberDecl();
  dumpPointer(Node->getMemberDecl());
}

void ASTDumper::VisitExtVectorElementExpr(const ExtVectorElementExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getAccessor().getNameStart();
}

void ASTDumper::VisitBinaryOperator(const BinaryOperator *Node) {
  VisitExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}

void ASTDumper::VisitCompoundAssignOperator(
    const CompoundAssignOperator *Node) {
  VisitExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode())
     << "' ComputeLHSTy=";
  dumpBareType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  dumpBareType(Node->getComputationResultType());
}

void ASTDumper::VisitBlockExpr(const BlockExpr *Node) {
  VisitExpr(Node);
  dumpDecl(Node->getBlockDecl());
}

void ASTDumper::VisitOpaqueValueExpr(const OpaqueValueExpr *Node) {
  VisitExpr(Node);

  if (Expr *Source = Node->getSourceExpr()) {
    lastChild();
    dumpStmt(Source);
  }
}

// GNU extensions.

void ASTDumper::VisitAddrLabelExpr(const AddrLabelExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getLabel()->getName();
  dumpPointer(Node->getLabel());
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getCastName()
     << "<" << Node->getTypeAsWritten().getAsString() << ">"
     << " <" << Node->getCastKindName();
  dumpBasePath(OS, Node);
  OS << ">";
}

void ASTDumper::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->getValue() ? "true" : "false");
}

void ASTDumper::VisitCXXThisExpr(const CXXThisExpr *Node) {
  VisitExpr(Node);
  OS << " this";
}

void ASTDumper::VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *Node) {
  VisitExpr(Node);
  OS << " functional cast to " << Node->getTypeAsWritten().getAsString()
     << " <" << Node->getCastKindName() << ">";
}

void ASTDumper::VisitCXXConstructExpr(const CXXConstructExpr *Node) {
  VisitExpr(Node);
  CXXConstructorDecl *Ctor = Node->getConstructor();
  dumpType(Ctor->getType());
  if (Node->isElidable())
    OS << " elidable";
  if (Node->requiresZeroInitialization())
    OS << " zeroing";
}

void ASTDumper::VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Node) {
  VisitExpr(Node);
  OS << " ";
  dumpCXXTemporary(Node->getTemporary());
}

void
ASTDumper::VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Node) {
  VisitExpr(Node);
  if (const ValueDecl *VD = Node->getExtendingDecl()) {
    OS << " extended by ";
    dumpBareDeclRef(VD);
  }
}

void ASTDumper::VisitExprWithCleanups(const ExprWithCleanups *Node) {
  VisitExpr(Node);
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i)
    dumpDeclRef(Node->getObject(i), "cleanup");
}

void ASTDumper::dumpCXXTemporary(const CXXTemporary *Temporary) {
  OS << "(CXXTemporary";
  dumpPointer(Temporary);
  OS << ")";
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCMessageExpr(const ObjCMessageExpr *Node) {
  VisitExpr(Node);
  OS << " selector=" << Node->getSelector().getAsString();
  switch (Node->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    break;

  case ObjCMessageExpr::Class:
    OS << " class=";
    dumpBareType(Node->getClassReceiver());
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
  VisitExpr(Node);
  OS << " selector=" << Node->getBoxingMethod()->getSelector().getAsString();
}

void ASTDumper::VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node) {
  VisitStmt(Node);
  if (const VarDecl *CatchParam = Node->getCatchParamDecl())
    dumpDecl(CatchParam);
  else
    OS << " catch all";
}

void ASTDumper::VisitObjCEncodeExpr(const ObjCEncodeExpr *Node) {
  VisitExpr(Node);
  dumpType(Node->getEncodedType());
}

void ASTDumper::VisitObjCSelectorExpr(const ObjCSelectorExpr *Node) {
  VisitExpr(Node);

  OS << " " << Node->getSelector().getAsString();
}

void ASTDumper::VisitObjCProtocolExpr(const ObjCProtocolExpr *Node) {
  VisitExpr(Node);

  OS << ' ' << *Node->getProtocol();
}

void ASTDumper::VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Node) {
  VisitExpr(Node);
  if (Node->isImplicitProperty()) {
    OS << " Kind=MethodRef Getter=\"";
    if (Node->getImplicitPropertyGetter())
      OS << Node->getImplicitPropertyGetter()->getSelector().getAsString();
    else
      OS << "(null)";

    OS << "\" Setter=\"";
    if (ObjCMethodDecl *Setter = Node->getImplicitPropertySetter())
      OS << Setter->getSelector().getAsString();
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
  VisitExpr(Node);
  if (Node->isArraySubscriptRefExpr())
    OS << " Kind=ArraySubscript GetterForArray=\"";
  else
    OS << " Kind=DictionarySubscript GetterForDictionary=\"";
  if (Node->getAtIndexMethodDecl())
    OS << Node->getAtIndexMethodDecl()->getSelector().getAsString();
  else
    OS << "(null)";

  if (Node->isArraySubscriptRefExpr())
    OS << "\" SetterForArray=\"";
  else
    OS << "\" SetterForDictionary=\"";
  if (Node->setAtIndexMethodDecl())
    OS << Node->setAtIndexMethodDecl()->getSelector().getAsString();
  else
    OS << "(null)";
}

void ASTDumper::VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

//===----------------------------------------------------------------------===//
// Comments
//===----------------------------------------------------------------------===//

const char *ASTDumper::getCommandName(unsigned CommandID) {
  if (Traits)
    return Traits->getCommandInfo(CommandID)->Name;
  const CommandInfo *Info = CommandTraits::getBuiltinCommandInfo(CommandID);
  if (Info)
    return Info->Name;
  return "<not a builtin command>";
}

void ASTDumper::dumpFullComment(const FullComment *C) {
  if (!C)
    return;

  FC = C;
  dumpComment(C);
  FC = 0;
}

void ASTDumper::dumpComment(const Comment *C) {
  IndentScope Indent(*this);

  if (!C) {
    ColorScope Color(*this, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(*this, CommentColor);
    OS << C->getCommentKindName();
  }
  dumpPointer(C);
  dumpSourceRange(C->getSourceRange());
  ConstCommentVisitor<ASTDumper>::visit(C);
  for (Comment::child_iterator I = C->child_begin(), E = C->child_end();
       I != E; ++I) {
    if (I + 1 == E)
      lastChild();
    dumpComment(*I);
  }
}

void ASTDumper::visitTextComment(const TextComment *C) {
  OS << " Text=\"" << C->getText() << "\"";
}

void ASTDumper::visitInlineCommandComment(const InlineCommandComment *C) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  switch (C->getRenderKind()) {
  case InlineCommandComment::RenderNormal:
    OS << " RenderNormal";
    break;
  case InlineCommandComment::RenderBold:
    OS << " RenderBold";
    break;
  case InlineCommandComment::RenderMonospaced:
    OS << " RenderMonospaced";
    break;
  case InlineCommandComment::RenderEmphasized:
    OS << " RenderEmphasized";
    break;
  }

  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void ASTDumper::visitHTMLStartTagComment(const HTMLStartTagComment *C) {
  OS << " Name=\"" << C->getTagName() << "\"";
  if (C->getNumAttrs() != 0) {
    OS << " Attrs: ";
    for (unsigned i = 0, e = C->getNumAttrs(); i != e; ++i) {
      const HTMLStartTagComment::Attribute &Attr = C->getAttr(i);
      OS << " \"" << Attr.Name << "=\"" << Attr.Value << "\"";
    }
  }
  if (C->isSelfClosing())
    OS << " SelfClosing";
}

void ASTDumper::visitHTMLEndTagComment(const HTMLEndTagComment *C) {
  OS << " Name=\"" << C->getTagName() << "\"";
}

void ASTDumper::visitBlockCommandComment(const BlockCommandComment *C) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void ASTDumper::visitParamCommandComment(const ParamCommandComment *C) {
  OS << " " << ParamCommandComment::getDirectionAsString(C->getDirection());

  if (C->isDirectionExplicit())
    OS << " explicitly";
  else
    OS << " implicitly";

  if (C->hasParamName()) {
    if (C->isParamIndexValid())
      OS << " Param=\"" << C->getParamName(FC) << "\"";
    else
      OS << " Param=\"" << C->getParamNameAsWritten() << "\"";
  }

  if (C->isParamIndexValid())
    OS << " ParamIndex=" << C->getParamIndex();
}

void ASTDumper::visitTParamCommandComment(const TParamCommandComment *C) {
  if (C->hasParamName()) {
    if (C->isPositionValid())
      OS << " Param=\"" << C->getParamName(FC) << "\"";
    else
      OS << " Param=\"" << C->getParamNameAsWritten() << "\"";
  }

  if (C->isPositionValid()) {
    OS << " Position=<";
    for (unsigned i = 0, e = C->getDepth(); i != e; ++i) {
      OS << C->getIndex(i);
      if (i != e - 1)
        OS << ", ";
    }
    OS << ">";
  }
}

void ASTDumper::visitVerbatimBlockComment(const VerbatimBlockComment *C) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\""
        " CloseName=\"" << C->getCloseName() << "\"";
}

void ASTDumper::visitVerbatimBlockLineComment(
    const VerbatimBlockLineComment *C) {
  OS << " Text=\"" << C->getText() << "\"";
}

void ASTDumper::visitVerbatimLineComment(const VerbatimLineComment *C) {
  OS << " Text=\"" << C->getText() << "\"";
}

//===----------------------------------------------------------------------===//
// Decl method implementations
//===----------------------------------------------------------------------===//

void Decl::dump() const {
  dump(llvm::errs());
}

void Decl::dump(raw_ostream &OS) const {
  ASTDumper P(OS, &getASTContext().getCommentCommandTraits(),
              &getASTContext().getSourceManager());
  P.dumpDecl(this);
}

void Decl::dumpColor() const {
  ASTDumper P(llvm::errs(), &getASTContext().getCommentCommandTraits(),
              &getASTContext().getSourceManager(), /*ShowColors*/true);
  P.dumpDecl(this);
}

void DeclContext::dumpLookups() const {
  dumpLookups(llvm::errs());
}

void DeclContext::dumpLookups(raw_ostream &OS) const {
  const DeclContext *DC = this;
  while (!DC->isTranslationUnit())
    DC = DC->getParent();
  ASTContext &Ctx = cast<TranslationUnitDecl>(DC)->getASTContext();
  ASTDumper P(OS, &Ctx.getCommentCommandTraits(), &Ctx.getSourceManager());
  P.dumpLookups(this);
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

void Stmt::dump(SourceManager &SM) const {
  dump(llvm::errs(), SM);
}

void Stmt::dump(raw_ostream &OS, SourceManager &SM) const {
  ASTDumper P(OS, 0, &SM);
  P.dumpStmt(this);
}

void Stmt::dump() const {
  ASTDumper P(llvm::errs(), 0, 0);
  P.dumpStmt(this);
}

void Stmt::dumpColor() const {
  ASTDumper P(llvm::errs(), 0, 0, /*ShowColors*/true);
  P.dumpStmt(this);
}

//===----------------------------------------------------------------------===//
// Comment method implementations
//===----------------------------------------------------------------------===//

void Comment::dump() const {
  dump(llvm::errs(), 0, 0);
}

void Comment::dump(const ASTContext &Context) const {
  dump(llvm::errs(), &Context.getCommentCommandTraits(),
       &Context.getSourceManager());
}

void Comment::dump(raw_ostream &OS, const CommandTraits *Traits,
                   const SourceManager *SM) const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  ASTDumper D(OS, Traits, SM);
  D.dumpFullComment(FC);
}

void Comment::dumpColor() const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  ASTDumper D(llvm::errs(), 0, 0, /*ShowColors*/true);
  D.dumpFullComment(FC);
}
