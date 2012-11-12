//===--- StmtDumper.cpp - Dumping implementation for Stmt ASTs ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dump method, which dumps out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// StmtDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class StmtDumper : public StmtVisitor<StmtDumper> {
    SourceManager *SM;
    raw_ostream &OS;
    unsigned IndentLevel;
    bool IsFirstLine;

    /// MaxDepth - When doing a normal dump (not dumpAll) we only want to dump
    /// the first few levels of an AST.  This keeps track of how many ast levels
    /// are left.
    unsigned MaxDepth;

    /// LastLocFilename/LastLocLine - Keep track of the last location we print
    /// out so that we can print out deltas from then on out.
    const char *LastLocFilename;
    unsigned LastLocLine;

    class IndentScope {
      StmtDumper &Dumper;
    public:
      IndentScope(StmtDumper &Dumper) : Dumper(Dumper) {
        Dumper.indent();
      }
      ~IndentScope() {
        Dumper.unindent();
      }
    };

  public:
    StmtDumper(SourceManager *sm, raw_ostream &os, unsigned maxDepth)
      : SM(sm), OS(os), IndentLevel(0), IsFirstLine(true), MaxDepth(maxDepth) {
      LastLocFilename = "";
      LastLocLine = ~0U;
    }

    ~StmtDumper() {
      OS << "\n";
    }

    void DumpSubTree(Stmt *S) {
      // Prune the recursion if not using dump all.
      if (MaxDepth == 0) return;

      IndentScope Indent(*this);

      if (!S) {
        OS << "<<<NULL>>>";
        return;
      }

      if (DeclStmt* DS = dyn_cast<DeclStmt>(S)) {
        VisitDeclStmt(DS);
        return;
      }

      Visit(S);
      for (Stmt::child_range CI = S->children(); CI; CI++)
        DumpSubTree(*CI);
    }

    void DumpDeclarator(Decl *D);

    void indent() {
      if (IsFirstLine)
        IsFirstLine = false;
      else
        OS << "\n";
      OS.indent(IndentLevel * 2);
      OS << "(";
      IndentLevel++;
    }

    void unindent() {
      OS << ")";
      IndentLevel--;
    }

    void DumpType(QualType T) {
      SplitQualType T_split = T.split();
      OS << "'" << QualType::getAsString(T_split) << "'";

      if (!T.isNull()) {
        // If the type is sugared, also dump a (shallow) desugared type.
        SplitQualType D_split = T.getSplitDesugaredType();
        if (T_split != D_split)
          OS << ":'" << QualType::getAsString(D_split) << "'";
      }
    }
    void DumpDeclRef(Decl *node);
    void DumpStmt(const Stmt *Node) {
      OS << Node->getStmtClassName()
         << " " << (const void*)Node;
      DumpSourceRange(Node);
    }
    void DumpValueKind(ExprValueKind K) {
      switch (K) {
      case VK_RValue: break;
      case VK_LValue: OS << " lvalue"; break;
      case VK_XValue: OS << " xvalue"; break;
      }
    }
    void DumpObjectKind(ExprObjectKind K) {
      switch (K) {
      case OK_Ordinary: break;
      case OK_BitField: OS << " bitfield"; break;
      case OK_ObjCProperty: OS << " objcproperty"; break;
      case OK_ObjCSubscript: OS << " objcsubscript"; break;
      case OK_VectorComponent: OS << " vectorcomponent"; break;
      }
    }
    void DumpExpr(const Expr *Node) {
      DumpStmt(Node);
      OS << ' ';
      DumpType(Node->getType());
      DumpValueKind(Node->getValueKind());
      DumpObjectKind(Node->getObjectKind());
    }
    void DumpSourceRange(const Stmt *Node);
    void DumpLocation(SourceLocation Loc);

    // Stmts.
    void VisitStmt(Stmt *Node);
    void VisitDeclStmt(DeclStmt *Node);
    void VisitLabelStmt(LabelStmt *Node);
    void VisitGotoStmt(GotoStmt *Node);

    // Exprs
    void VisitExpr(Expr *Node);
    void VisitCastExpr(CastExpr *Node);
    void VisitDeclRefExpr(DeclRefExpr *Node);
    void VisitPredefinedExpr(PredefinedExpr *Node);
    void VisitCharacterLiteral(CharacterLiteral *Node);
    void VisitIntegerLiteral(IntegerLiteral *Node);
    void VisitFloatingLiteral(FloatingLiteral *Node);
    void VisitStringLiteral(StringLiteral *Str);
    void VisitUnaryOperator(UnaryOperator *Node);
    void VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node);
    void VisitMemberExpr(MemberExpr *Node);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *Node);
    void VisitBinaryOperator(BinaryOperator *Node);
    void VisitCompoundAssignOperator(CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(AddrLabelExpr *Node);
    void VisitBlockExpr(BlockExpr *Node);
    void VisitOpaqueValueExpr(OpaqueValueExpr *Node);

    // C++
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(CXXThisExpr *Node);
    void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node);
    void VisitCXXConstructExpr(CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node);
    void VisitExprWithCleanups(ExprWithCleanups *Node);
    void VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node);
    void DumpCXXTemporary(CXXTemporary *Temporary);

    // ObjC
    void VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node);
    void VisitObjCEncodeExpr(ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(ObjCMessageExpr* Node);
    void VisitObjCBoxedExpr(ObjCBoxedExpr* Node);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node);
    void VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node);
    void VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node);
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

void StmtDumper::DumpLocation(SourceLocation Loc) {
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

void StmtDumper::DumpSourceRange(const Stmt *Node) {
  // Can't translate locations if a SourceManager isn't available.
  if (SM == 0) return;

  // TODO: If the parent expression is available, we can print a delta vs its
  // location.
  SourceRange R = Node->getSourceRange();

  OS << " <";
  DumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    DumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>

}


//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitStmt(Stmt *Node) {
  DumpStmt(Node);
}

void StmtDumper::DumpDeclarator(Decl *D) {
  // FIXME: Need to complete/beautify this... this code simply shows the
  // nodes are where they need to be.
  if (TypedefDecl *localType = dyn_cast<TypedefDecl>(D)) {
    OS << "\"typedef " << localType->getUnderlyingType().getAsString()
       << ' ' << *localType << '"';
  } else if (TypeAliasDecl *localType = dyn_cast<TypeAliasDecl>(D)) {
    OS << "\"using " << *localType << " = "
       << localType->getUnderlyingType().getAsString() << '"';
  } else if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    OS << "\"";
    // Emit storage class for vardecls.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getStorageClass() != SC_None)
        OS << VarDecl::getStorageClassSpecifierString(V->getStorageClass())
           << " ";
    }

    std::string Name = VD->getNameAsString();
    VD->getType().getAsStringInternal(Name,
                          PrintingPolicy(VD->getASTContext().getLangOpts()));
    OS << Name;

    // If this is a vardecl with an initializer, emit it.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getInit()) {
        OS << " =";
        DumpSubTree(V->getInit());
      }
    }
    OS << '"';
  } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // print a free standing tag decl (e.g. "struct x;").
    const char *tagname;
    if (const IdentifierInfo *II = TD->getIdentifier())
      tagname = II->getNameStart();
    else
      tagname = "<anonymous>";
    OS << '"' << TD->getKindName() << ' ' << tagname << ";\"";
    // FIXME: print tag bodies.
  } else if (UsingDirectiveDecl *UD = dyn_cast<UsingDirectiveDecl>(D)) {
    // print using-directive decl (e.g. "using namespace x;")
    const char *ns;
    if (const IdentifierInfo *II = UD->getNominatedNamespace()->getIdentifier())
      ns = II->getNameStart();
    else
      ns = "<anonymous>";
    OS << '"' << UD->getDeclKindName() << ns << ";\"";
  } else if (UsingDecl *UD = dyn_cast<UsingDecl>(D)) {
    // print using decl (e.g. "using std::string;")
    const char *tn = UD->isTypeName() ? "typename " : "";
    OS << '"' << UD->getDeclKindName() << tn;
    UD->getQualifier()->print(OS,
                        PrintingPolicy(UD->getASTContext().getLangOpts()));
    OS << ";\"";
  } else if (LabelDecl *LD = dyn_cast<LabelDecl>(D)) {
    OS << "label " << *LD;
  } else if (StaticAssertDecl *SAD = dyn_cast<StaticAssertDecl>(D)) {
    OS << "\"static_assert(";
    DumpSubTree(SAD->getAssertExpr());
    OS << ",";
    DumpSubTree(SAD->getMessage());
    OS << ");\"";
  } else {
    llvm_unreachable("Unexpected decl");
  }
}

void StmtDumper::VisitDeclStmt(DeclStmt *Node) {
  DumpStmt(Node);
  for (DeclStmt::decl_iterator DI = Node->decl_begin(), DE = Node->decl_end();
       DI != DE; ++DI) {
    IndentScope Indent(*this);
    Decl* D = *DI;
    OS << (void*) D << " ";
    DumpDeclarator(D);
  }
}

void StmtDumper::VisitLabelStmt(LabelStmt *Node) {
  DumpStmt(Node);
  OS << " '" << Node->getName() << "'";
}

void StmtDumper::VisitGotoStmt(GotoStmt *Node) {
  DumpStmt(Node);
  OS << " '" << Node->getLabel()->getName()
     << "':" << (void*)Node->getLabel();
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitExpr(Expr *Node) {
  DumpExpr(Node);
}

static void DumpBasePath(raw_ostream &OS, CastExpr *Node) {
  if (Node->path_empty())
    return;

  OS << " (";
  bool First = true;
  for (CastExpr::path_iterator
         I = Node->path_begin(), E = Node->path_end(); I != E; ++I) {
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

void StmtDumper::VisitCastExpr(CastExpr *Node) {
  DumpExpr(Node);
  OS << " <" << Node->getCastKindName();
  DumpBasePath(OS, Node);
  OS << ">";
}

void StmtDumper::VisitDeclRefExpr(DeclRefExpr *Node) {
  DumpExpr(Node);

  OS << " ";
  DumpDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    OS << " (";
    DumpDeclRef(Node->getFoundDecl());
    OS << ")";
  }
}

void StmtDumper::DumpDeclRef(Decl *d) {
  OS << d->getDeclKindName() << ' ' << (void*) d;

  if (NamedDecl *nd = dyn_cast<NamedDecl>(d)) {
    OS << " '";
    nd->getDeclName().printName(OS);
    OS << "'";
  }

  if (ValueDecl *vd = dyn_cast<ValueDecl>(d)) {
    OS << ' '; DumpType(vd->getType());
  }
}

void StmtDumper::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
  DumpExpr(Node);
  OS << " (";
  if (!Node->requiresADL()) OS << "no ";
  OS << "ADL) = '" << Node->getName() << '\'';

  UnresolvedLookupExpr::decls_iterator
    I = Node->decls_begin(), E = Node->decls_end();
  if (I == E) OS << " empty";
  for (; I != E; ++I)
    OS << " " << (void*) *I;
}

void StmtDumper::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  DumpExpr(Node);

  OS << " " << Node->getDecl()->getDeclKindName()
     << "Decl='" << *Node->getDecl()
     << "' " << (void*)Node->getDecl();
  if (Node->isFreeIvar())
    OS << " isFreeIvar";
}

void StmtDumper::VisitPredefinedExpr(PredefinedExpr *Node) {
  DumpExpr(Node);
  switch (Node->getIdentType()) {
  default: llvm_unreachable("unknown case");
  case PredefinedExpr::Func:           OS <<  " __func__"; break;
  case PredefinedExpr::Function:       OS <<  " __FUNCTION__"; break;
  case PredefinedExpr::LFunction:       OS <<  " L__FUNCTION__"; break;
  case PredefinedExpr::PrettyFunction: OS <<  " __PRETTY_FUNCTION__";break;
  }
}

void StmtDumper::VisitCharacterLiteral(CharacterLiteral *Node) {
  DumpExpr(Node);
  OS << " " << Node->getValue();
}

void StmtDumper::VisitIntegerLiteral(IntegerLiteral *Node) {
  DumpExpr(Node);

  bool isSigned = Node->getType()->isSignedIntegerType();
  OS << " " << Node->getValue().toString(10, isSigned);
}
void StmtDumper::VisitFloatingLiteral(FloatingLiteral *Node) {
  DumpExpr(Node);
  OS << " " << Node->getValueAsApproximateDouble();
}

void StmtDumper::VisitStringLiteral(StringLiteral *Str) {
  DumpExpr(Str);
  OS << " ";
  Str->outputString(OS);
}

void StmtDumper::VisitUnaryOperator(UnaryOperator *Node) {
  DumpExpr(Node);
  OS << " " << (Node->isPostfix() ? "postfix" : "prefix")
     << " '" << UnaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}
void StmtDumper::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) {
  DumpExpr(Node);
  switch(Node->getKind()) {
  case UETT_SizeOf:
    OS << " sizeof ";
    break;
  case UETT_AlignOf:
    OS << " alignof ";
    break;
  case UETT_VecStep:
    OS << " vec_step ";
    break;
  }
  if (Node->isArgumentType())
    DumpType(Node->getArgumentType());
}

void StmtDumper::VisitMemberExpr(MemberExpr *Node) {
  DumpExpr(Node);
  OS << " " << (Node->isArrow() ? "->" : ".")
     << *Node->getMemberDecl() << ' '
     << (void*)Node->getMemberDecl();
}
void StmtDumper::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  DumpExpr(Node);
  OS << " " << Node->getAccessor().getNameStart();
}
void StmtDumper::VisitBinaryOperator(BinaryOperator *Node) {
  DumpExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}
void StmtDumper::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  DumpExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode())
     << "' ComputeLHSTy=";
  DumpType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  DumpType(Node->getComputationResultType());
}

void StmtDumper::VisitBlockExpr(BlockExpr *Node) {
  DumpExpr(Node);

  BlockDecl *block = Node->getBlockDecl();
  OS << " decl=" << block;

  if (block->capturesCXXThis()) {
    IndentScope Indent(*this);
    OS << "capture this";
  }
  for (BlockDecl::capture_iterator
         i = block->capture_begin(), e = block->capture_end(); i != e; ++i) {
    IndentScope Indent(*this);
    OS << "capture ";
    if (i->isByRef()) OS << "byref ";
    if (i->isNested()) OS << "nested ";
    if (i->getVariable())
      DumpDeclRef(i->getVariable());
    if (i->hasCopyExpr()) DumpSubTree(i->getCopyExpr());
  }

  DumpSubTree(block->getBody());
}

void StmtDumper::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
  DumpExpr(Node);

  if (Expr *Source = Node->getSourceExpr())
    DumpSubTree(Source);
}

// GNU extensions.

void StmtDumper::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  DumpExpr(Node);
  OS << " " << Node->getLabel()->getName()
     << " " << (void*)Node->getLabel();
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void StmtDumper::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  DumpExpr(Node);
  OS << " " << Node->getCastName() 
     << "<" << Node->getTypeAsWritten().getAsString() << ">"
     << " <" << Node->getCastKindName();
  DumpBasePath(OS, Node);
  OS << ">";
}

void StmtDumper::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  DumpExpr(Node);
  OS << " " << (Node->getValue() ? "true" : "false");
}

void StmtDumper::VisitCXXThisExpr(CXXThisExpr *Node) {
  DumpExpr(Node);
  OS << " this";
}

void StmtDumper::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  DumpExpr(Node);
  OS << " functional cast to " << Node->getTypeAsWritten().getAsString()
     << " <" << Node->getCastKindName() << ">";
}

void StmtDumper::VisitCXXConstructExpr(CXXConstructExpr *Node) {
  DumpExpr(Node);
  CXXConstructorDecl *Ctor = Node->getConstructor();
  DumpType(Ctor->getType());
  if (Node->isElidable())
    OS << " elidable";
  if (Node->requiresZeroInitialization())
    OS << " zeroing";
}

void StmtDumper::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
  DumpExpr(Node);
  OS << " ";
  DumpCXXTemporary(Node->getTemporary());
}

void StmtDumper::VisitExprWithCleanups(ExprWithCleanups *Node) {
  DumpExpr(Node);
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i) {
    IndentScope Indent(*this);
    OS << "cleanup ";
    DumpDeclRef(Node->getObject(i));
  }
}

void StmtDumper::DumpCXXTemporary(CXXTemporary *Temporary) {
  OS << "(CXXTemporary " << (void *)Temporary << ")";
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void StmtDumper::VisitObjCMessageExpr(ObjCMessageExpr* Node) {
  DumpExpr(Node);
  OS << " selector=" << Node->getSelector().getAsString();
  switch (Node->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    break;

  case ObjCMessageExpr::Class:
    OS << " class=";
    DumpType(Node->getClassReceiver());
    break;

  case ObjCMessageExpr::SuperInstance:
    OS << " super (instance)";
    break;

  case ObjCMessageExpr::SuperClass:
    OS << " super (class)";
    break;
  }
}

void StmtDumper::VisitObjCBoxedExpr(ObjCBoxedExpr* Node) {
  DumpExpr(Node);
  OS << " selector=" << Node->getBoxingMethod()->getSelector().getAsString();
}

void StmtDumper::VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node) {
  DumpStmt(Node);
  if (VarDecl *CatchParam = Node->getCatchParamDecl()) {
    OS << " catch parm = ";
    DumpDeclarator(CatchParam);
  } else {
    OS << " catch all";
  }
}

void StmtDumper::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  DumpExpr(Node);
  OS << " ";
  DumpType(Node->getEncodedType());
}

void StmtDumper::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  DumpExpr(Node);

  OS << " " << Node->getSelector().getAsString();
}

void StmtDumper::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  DumpExpr(Node);

  OS << ' ' <<* Node->getProtocol();
}

void StmtDumper::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  DumpExpr(Node);
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

void StmtDumper::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node) {
  DumpExpr(Node);
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

void StmtDumper::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node) {
  DumpExpr(Node);
  OS << " " << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump(SourceManager &SM) const {
  dump(llvm::errs(), SM);
}

void Stmt::dump(raw_ostream &OS, SourceManager &SM) const {
  StmtDumper P(&SM, OS, 4);
  P.DumpSubTree(const_cast<Stmt*>(this));
}

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump() const {
  StmtDumper P(0, llvm::errs(), 4);
  P.DumpSubTree(const_cast<Stmt*>(this));
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll(SourceManager &SM) const {
  StmtDumper P(&SM, llvm::errs(), ~0U);
  P.DumpSubTree(const_cast<Stmt*>(this));
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll() const {
  StmtDumper P(0, llvm::errs(), ~0U);
  P.DumpSubTree(const_cast<Stmt*>(this));
}
