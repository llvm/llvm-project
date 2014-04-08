//===--- StmtPrinter.cpp - Printing implementation for Stmt ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dumpPretty/Stmt::printPretty methods, which
// pretty print the AST back out to C code.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// StmtPrinter Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class StmtPrinter : public StmtVisitor<StmtPrinter> {
    raw_ostream &OS;
    unsigned IndentLevel;
    clang::PrinterHelper* Helper;
    PrintingPolicy Policy;

  public:
    StmtPrinter(raw_ostream &os, PrinterHelper* helper,
                const PrintingPolicy &Policy,
                unsigned Indentation = 0)
      : OS(os), IndentLevel(Indentation), Helper(helper), Policy(Policy) {}

    void PrintStmt(Stmt *S) {
      PrintStmt(S, Policy.Indentation);
    }

    void PrintStmt(Stmt *S, int SubIndent) {
      IndentLevel += SubIndent;
      if (S && isa<Expr>(S)) {
        // If this is an expr used in a stmt context, indent and newline it.
        Indent();
        Visit(S);
        OS << ";\n";
      } else if (S) {
        Visit(S);
      } else {
        Indent() << "<<<NULL STATEMENT>>>\n";
      }
      IndentLevel -= SubIndent;
    }

    void PrintRawCompoundStmt(CompoundStmt *S);
    void PrintRawDecl(Decl *D);
    void PrintRawDeclStmt(const DeclStmt *S);
    void PrintRawIfStmt(IfStmt *If);
    void PrintRawCXXCatchStmt(CXXCatchStmt *Catch);
    void PrintCallArgs(CallExpr *E);
    void PrintRawSEHExceptHandler(SEHExceptStmt *S);
    void PrintRawSEHFinallyStmt(SEHFinallyStmt *S);

    void PrintExpr(Expr *E) {
      if (E)
        Visit(E);
      else
        OS << "<null expr>";
    }

    raw_ostream &Indent(int Delta = 0) {
      for (int i = 0, e = IndentLevel+Delta; i < e; ++i)
        OS << "  ";
      return OS;
    }

    void Visit(Stmt* S) {
      if (Helper && Helper->handledStmt(S,OS))
          return;
      else StmtVisitor<StmtPrinter>::Visit(S);
    }
    
    void VisitStmt(Stmt *Node) LLVM_ATTRIBUTE_UNUSED {
      Indent() << "<<unknown stmt type>>\n";
    }
    void VisitExpr(Expr *Node) LLVM_ATTRIBUTE_UNUSED {
      OS << "<<unknown expr type>>";
    }
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);

#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT) \
    void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.inc"
  };
}

//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void StmtPrinter::PrintRawCompoundStmt(CompoundStmt *Node) {
  OS << "{\n";
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ++I)
    PrintStmt(*I);

  Indent() << "}";
}

void StmtPrinter::PrintRawDecl(Decl *D) {
  D->print(OS, Policy, IndentLevel);
}

void StmtPrinter::PrintRawDeclStmt(const DeclStmt *S) {
  DeclStmt::const_decl_iterator Begin = S->decl_begin(), End = S->decl_end();
  SmallVector<Decl*, 2> Decls;
  for ( ; Begin != End; ++Begin)
    Decls.push_back(*Begin);

  Decl::printGroup(Decls.data(), Decls.size(), OS, Policy, IndentLevel);
}

void StmtPrinter::VisitNullStmt(NullStmt *Node) {
  Indent() << ";\n";
}

void StmtPrinter::VisitDeclStmt(DeclStmt *Node) {
  Indent();
  PrintRawDeclStmt(Node);
  OS << ";\n";
}

void StmtPrinter::VisitCompoundStmt(CompoundStmt *Node) {
  Indent();
  PrintRawCompoundStmt(Node);
  OS << "\n";
}

void StmtPrinter::VisitCaseStmt(CaseStmt *Node) {
  Indent(-1) << "case ";
  PrintExpr(Node->getLHS());
  if (Node->getRHS()) {
    OS << " ... ";
    PrintExpr(Node->getRHS());
  }
  OS << ":\n";

  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::VisitDefaultStmt(DefaultStmt *Node) {
  Indent(-1) << "default:\n";
  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::VisitLabelStmt(LabelStmt *Node) {
  Indent(-1) << Node->getName() << ":\n";
  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::VisitAttributedStmt(AttributedStmt *Node) {
  OS << "[[";
  bool first = true;
  for (ArrayRef<const Attr*>::iterator it = Node->getAttrs().begin(),
                                       end = Node->getAttrs().end();
                                       it != end; ++it) {
    if (!first) {
      OS << ", ";
      first = false;
    }
    // TODO: check this
    (*it)->printPretty(OS, Policy);
  }
  OS << "]] ";
  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::PrintRawIfStmt(IfStmt *If) {
  OS << "if (";
  if (const DeclStmt *DS = If->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else
    PrintExpr(If->getCond());
  OS << ')';

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen())) {
    OS << ' ';
    PrintRawCompoundStmt(CS);
    OS << (If->getElse() ? ' ' : '\n');
  } else {
    OS << '\n';
    PrintStmt(If->getThen());
    if (If->getElse()) Indent();
  }

  if (Stmt *Else = If->getElse()) {
    OS << "else";

    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
      OS << ' ';
      PrintRawCompoundStmt(CS);
      OS << '\n';
    } else if (IfStmt *ElseIf = dyn_cast<IfStmt>(Else)) {
      OS << ' ';
      PrintRawIfStmt(ElseIf);
    } else {
      OS << '\n';
      PrintStmt(If->getElse());
    }
  }
}

void StmtPrinter::VisitIfStmt(IfStmt *If) {
  Indent();
  PrintRawIfStmt(If);
}

void StmtPrinter::VisitSwitchStmt(SwitchStmt *Node) {
  Indent() << "switch (";
  if (const DeclStmt *DS = Node->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else
    PrintExpr(Node->getCond());
  OS << ")";

  // Pretty print compoundstmt bodies (very common).
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    OS << " ";
    PrintRawCompoundStmt(CS);
    OS << "\n";
  } else {
    OS << "\n";
    PrintStmt(Node->getBody());
  }
}

void StmtPrinter::VisitWhileStmt(WhileStmt *Node) {
  Indent() << "while (";
  if (const DeclStmt *DS = Node->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else
    PrintExpr(Node->getCond());
  OS << ")\n";
  PrintStmt(Node->getBody());
}

void StmtPrinter::VisitDoStmt(DoStmt *Node) {
  Indent() << "do ";
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    PrintRawCompoundStmt(CS);
    OS << " ";
  } else {
    OS << "\n";
    PrintStmt(Node->getBody());
    Indent();
  }

  OS << "while (";
  PrintExpr(Node->getCond());
  OS << ");\n";
}

void StmtPrinter::VisitForStmt(ForStmt *Node) {
  Indent() << "for (";
  if (Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getInit()))
      PrintRawDeclStmt(DS);
    else
      PrintExpr(cast<Expr>(Node->getInit()));
  }
  OS << ";";
  if (Node->getCond()) {
    OS << " ";
    PrintExpr(Node->getCond());
  }
  OS << ";";
  if (Node->getInc()) {
    OS << " ";
    PrintExpr(Node->getInc());
  }
  OS << ") ";

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    PrintRawCompoundStmt(CS);
    OS << "\n";
  } else {
    OS << "\n";
    PrintStmt(Node->getBody());
  }
}

void StmtPrinter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *Node) {
  Indent() << "for (";
  if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getElement()))
    PrintRawDeclStmt(DS);
  else
    PrintExpr(cast<Expr>(Node->getElement()));
  OS << " in ";
  PrintExpr(Node->getCollection());
  OS << ") ";

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    PrintRawCompoundStmt(CS);
    OS << "\n";
  } else {
    OS << "\n";
    PrintStmt(Node->getBody());
  }
}

void StmtPrinter::VisitCXXForRangeStmt(CXXForRangeStmt *Node) {
  Indent() << "for (";
  PrintingPolicy SubPolicy(Policy);
  SubPolicy.SuppressInitializers = true;
  Node->getLoopVariable()->print(OS, SubPolicy, IndentLevel);
  OS << " : ";
  PrintExpr(Node->getRangeInit());
  OS << ") {\n";
  PrintStmt(Node->getBody());
  Indent() << "}\n";
}

void StmtPrinter::VisitMSDependentExistsStmt(MSDependentExistsStmt *Node) {
  Indent();
  if (Node->isIfExists())
    OS << "__if_exists (";
  else
    OS << "__if_not_exists (";
  
  if (NestedNameSpecifier *Qualifier
        = Node->getQualifierLoc().getNestedNameSpecifier())
    Qualifier->print(OS, Policy);
  
  OS << Node->getNameInfo() << ") ";
  
  PrintRawCompoundStmt(Node->getSubStmt());
}

void StmtPrinter::VisitGotoStmt(GotoStmt *Node) {
  Indent() << "goto " << Node->getLabel()->getName() << ";\n";
}

void StmtPrinter::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Indent() << "goto *";
  PrintExpr(Node->getTarget());
  OS << ";\n";
}

void StmtPrinter::VisitContinueStmt(ContinueStmt *Node) {
  Indent() << "continue;\n";
}

void StmtPrinter::VisitBreakStmt(BreakStmt *Node) {
  Indent() << "break;\n";
}


void StmtPrinter::VisitReturnStmt(ReturnStmt *Node) {
  Indent() << "return";
  if (Node->getRetValue()) {
    OS << " ";
    PrintExpr(Node->getRetValue());
  }
  OS << ";\n";
}


void StmtPrinter::VisitGCCAsmStmt(GCCAsmStmt *Node) {
  Indent() << "asm ";

  if (Node->isVolatile())
    OS << "volatile ";

  OS << "(";
  VisitStringLiteral(Node->getAsmString());

  // Outputs
  if (Node->getNumOutputs() != 0 || Node->getNumInputs() != 0 ||
      Node->getNumClobbers() != 0)
    OS << " : ";

  for (unsigned i = 0, e = Node->getNumOutputs(); i != e; ++i) {
    if (i != 0)
      OS << ", ";

    if (!Node->getOutputName(i).empty()) {
      OS << '[';
      OS << Node->getOutputName(i);
      OS << "] ";
    }

    VisitStringLiteral(Node->getOutputConstraintLiteral(i));
    OS << " ";
    Visit(Node->getOutputExpr(i));
  }

  // Inputs
  if (Node->getNumInputs() != 0 || Node->getNumClobbers() != 0)
    OS << " : ";

  for (unsigned i = 0, e = Node->getNumInputs(); i != e; ++i) {
    if (i != 0)
      OS << ", ";

    if (!Node->getInputName(i).empty()) {
      OS << '[';
      OS << Node->getInputName(i);
      OS << "] ";
    }

    VisitStringLiteral(Node->getInputConstraintLiteral(i));
    OS << " ";
    Visit(Node->getInputExpr(i));
  }

  // Clobbers
  if (Node->getNumClobbers() != 0)
    OS << " : ";

  for (unsigned i = 0, e = Node->getNumClobbers(); i != e; ++i) {
    if (i != 0)
      OS << ", ";

    VisitStringLiteral(Node->getClobberStringLiteral(i));
  }

  OS << ");\n";
}

void StmtPrinter::VisitMSAsmStmt(MSAsmStmt *Node) {
  // FIXME: Implement MS style inline asm statement printer.
  Indent() << "__asm ";
  if (Node->hasBraces())
    OS << "{\n";
  OS << Node->getAsmString() << "\n";
  if (Node->hasBraces())
    Indent() << "}\n";
}

void StmtPrinter::VisitCapturedStmt(CapturedStmt *Node) {
  PrintStmt(Node->getCapturedDecl()->getBody());
}

void StmtPrinter::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
  Indent() << "@try";
  if (CompoundStmt *TS = dyn_cast<CompoundStmt>(Node->getTryBody())) {
    PrintRawCompoundStmt(TS);
    OS << "\n";
  }

  for (unsigned I = 0, N = Node->getNumCatchStmts(); I != N; ++I) {
    ObjCAtCatchStmt *catchStmt = Node->getCatchStmt(I);
    Indent() << "@catch(";
    if (catchStmt->getCatchParamDecl()) {
      if (Decl *DS = catchStmt->getCatchParamDecl())
        PrintRawDecl(DS);
    }
    OS << ")";
    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(catchStmt->getCatchBody())) {
      PrintRawCompoundStmt(CS);
      OS << "\n";
    }
  }

  if (ObjCAtFinallyStmt *FS = static_cast<ObjCAtFinallyStmt *>(
        Node->getFinallyStmt())) {
    Indent() << "@finally";
    PrintRawCompoundStmt(dyn_cast<CompoundStmt>(FS->getFinallyBody()));
    OS << "\n";
  }
}

void StmtPrinter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void StmtPrinter::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
  Indent() << "@catch (...) { /* todo */ } \n";
}

void StmtPrinter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
  Indent() << "@throw";
  if (Node->getThrowExpr()) {
    OS << " ";
    PrintExpr(Node->getThrowExpr());
  }
  OS << ";\n";
}

void StmtPrinter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
  Indent() << "@synchronized (";
  PrintExpr(Node->getSynchExpr());
  OS << ")";
  PrintRawCompoundStmt(Node->getSynchBody());
  OS << "\n";
}

void StmtPrinter::VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *Node) {
  Indent() << "@autoreleasepool";
  PrintRawCompoundStmt(dyn_cast<CompoundStmt>(Node->getSubStmt()));
  OS << "\n";
}

void StmtPrinter::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
  OS << "catch (";
  if (Decl *ExDecl = Node->getExceptionDecl())
    PrintRawDecl(ExDecl);
  else
    OS << "...";
  OS << ") ";
  PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void StmtPrinter::VisitCXXCatchStmt(CXXCatchStmt *Node) {
  Indent();
  PrintRawCXXCatchStmt(Node);
  OS << "\n";
}

void StmtPrinter::VisitCXXTryStmt(CXXTryStmt *Node) {
  Indent() << "try ";
  PrintRawCompoundStmt(Node->getTryBlock());
  for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i) {
    OS << " ";
    PrintRawCXXCatchStmt(Node->getHandler(i));
  }
  OS << "\n";
}

void StmtPrinter::VisitSEHTryStmt(SEHTryStmt *Node) {
  Indent() << (Node->getIsCXXTry() ? "try " : "__try ");
  PrintRawCompoundStmt(Node->getTryBlock());
  SEHExceptStmt *E = Node->getExceptHandler();
  SEHFinallyStmt *F = Node->getFinallyHandler();
  if(E)
    PrintRawSEHExceptHandler(E);
  else {
    assert(F && "Must have a finally block...");
    PrintRawSEHFinallyStmt(F);
  }
  OS << "\n";
}

void StmtPrinter::PrintRawSEHFinallyStmt(SEHFinallyStmt *Node) {
  OS << "__finally ";
  PrintRawCompoundStmt(Node->getBlock());
  OS << "\n";
}

void StmtPrinter::PrintRawSEHExceptHandler(SEHExceptStmt *Node) {
  OS << "__except (";
  VisitExpr(Node->getFilterExpr());
  OS << ")\n";
  PrintRawCompoundStmt(Node->getBlock());
  OS << "\n";
}

void StmtPrinter::VisitSEHExceptStmt(SEHExceptStmt *Node) {
  Indent();
  PrintRawSEHExceptHandler(Node);
  OS << "\n";
}

void StmtPrinter::VisitSEHFinallyStmt(SEHFinallyStmt *Node) {
  Indent();
  PrintRawSEHFinallyStmt(Node);
  OS << "\n";
}

//===----------------------------------------------------------------------===//
//  OpenMP clauses printing methods
//===----------------------------------------------------------------------===//

namespace {
class OMPClausePrinter : public OMPClauseVisitor<OMPClausePrinter> {
  raw_ostream &OS;
  /// \brief Process clauses with list of variables.
  template <typename T>
  void VisitOMPClauseList(T *Node, char StartSym);
public:
  OMPClausePrinter(raw_ostream &OS) : OS(OS) { }
#define OPENMP_CLAUSE(Name, Class)                              \
  void Visit##Class(Class *S);
#include "clang/Basic/OpenMPKinds.def"
};

void OMPClausePrinter::VisitOMPDefaultClause(OMPDefaultClause *Node) {
  OS << "default("
     << getOpenMPSimpleClauseTypeName(OMPC_default, Node->getDefaultKind())
     << ")";
}

template<typename T>
void OMPClausePrinter::VisitOMPClauseList(T *Node, char StartSym) {
  for (typename T::varlist_iterator I = Node->varlist_begin(),
                                    E = Node->varlist_end();
         I != E; ++I)
    OS << (I == Node->varlist_begin() ? StartSym : ',')
       << *cast<NamedDecl>(cast<DeclRefExpr>(*I)->getDecl());
}

void OMPClausePrinter::VisitOMPPrivateClause(OMPPrivateClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "private";
    VisitOMPClauseList(Node, '(');
    OS << ")";
  }
}

void OMPClausePrinter::VisitOMPFirstprivateClause(OMPFirstprivateClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "firstprivate";
    VisitOMPClauseList(Node, '(');
    OS << ")";
  }
}

void OMPClausePrinter::VisitOMPSharedClause(OMPSharedClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "shared";
    VisitOMPClauseList(Node, '(');
    OS << ")";
  }
}

}

//===----------------------------------------------------------------------===//
//  OpenMP directives printing methods
//===----------------------------------------------------------------------===//

void StmtPrinter::VisitOMPParallelDirective(OMPParallelDirective *Node) {
  Indent() << "#pragma omp parallel ";

  OMPClausePrinter Printer(OS);
  ArrayRef<OMPClause *> Clauses = Node->clauses();
  for (ArrayRef<OMPClause *>::iterator I = Clauses.begin(), E = Clauses.end();
       I != E; ++I)
    if (*I && !(*I)->isImplicit()) {
      Printer.Visit(*I);
      OS << ' ';
    }
  OS << "\n";
  if (Node->getAssociatedStmt()) {
    assert(isa<CapturedStmt>(Node->getAssociatedStmt()) &&
           "Expected captured statement!");
    Stmt *CS = cast<CapturedStmt>(Node->getAssociatedStmt())->getCapturedStmt();
    PrintStmt(CS);
  }
}
//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void StmtPrinter::VisitDeclRefExpr(DeclRefExpr *Node) {
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(OS, Policy);
  if (Node->hasTemplateKeyword())
    OS << "template ";
  OS << Node->getNameInfo();
  if (Node->hasExplicitTemplateArgs())
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, Node->getTemplateArgs(), Node->getNumTemplateArgs(), Policy);
}

void StmtPrinter::VisitDependentScopeDeclRefExpr(
                                           DependentScopeDeclRefExpr *Node) {
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(OS, Policy);
  if (Node->hasTemplateKeyword())
    OS << "template ";
  OS << Node->getNameInfo();
  if (Node->hasExplicitTemplateArgs())
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, Node->getTemplateArgs(), Node->getNumTemplateArgs(), Policy);
}

void StmtPrinter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
  if (Node->getQualifier())
    Node->getQualifier()->print(OS, Policy);
  if (Node->hasTemplateKeyword())
    OS << "template ";
  OS << Node->getNameInfo();
  if (Node->hasExplicitTemplateArgs())
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, Node->getTemplateArgs(), Node->getNumTemplateArgs(), Policy);
}

void StmtPrinter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  if (Node->getBase()) {
    PrintExpr(Node->getBase());
    OS << (Node->isArrow() ? "->" : ".");
  }
  OS << *Node->getDecl();
}

void StmtPrinter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  if (Node->isSuperReceiver())
    OS << "super.";
  else if (Node->isObjectReceiver() && Node->getBase()) {
    PrintExpr(Node->getBase());
    OS << ".";
  } else if (Node->isClassReceiver() && Node->getClassReceiver()) {
    OS << Node->getClassReceiver()->getName() << ".";
  }

  if (Node->isImplicitProperty())
    OS << Node->getImplicitPropertyGetter()->getSelector().getAsString();
  else
    OS << Node->getExplicitProperty()->getName();
}

void StmtPrinter::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node) {
  
  PrintExpr(Node->getBaseExpr());
  OS << "[";
  PrintExpr(Node->getKeyExpr());
  OS << "]";
}

void StmtPrinter::VisitPredefinedExpr(PredefinedExpr *Node) {
  switch (Node->getIdentType()) {
    default:
      llvm_unreachable("unknown case");
    case PredefinedExpr::Func:
      OS << "__func__";
      break;
    case PredefinedExpr::Function:
      OS << "__FUNCTION__";
      break;
    case PredefinedExpr::FuncDName:
      OS << "__FUNCDNAME__";
      break;
    case PredefinedExpr::LFunction:
      OS << "L__FUNCTION__";
      break;
    case PredefinedExpr::PrettyFunction:
      OS << "__PRETTY_FUNCTION__";
      break;
  }
}

void StmtPrinter::VisitCharacterLiteral(CharacterLiteral *Node) {
  unsigned value = Node->getValue();

  switch (Node->getKind()) {
  case CharacterLiteral::Ascii: break; // no prefix.
  case CharacterLiteral::Wide:  OS << 'L'; break;
  case CharacterLiteral::UTF16: OS << 'u'; break;
  case CharacterLiteral::UTF32: OS << 'U'; break;
  }

  switch (value) {
  case '\\':
    OS << "'\\\\'";
    break;
  case '\'':
    OS << "'\\''";
    break;
  case '\a':
    // TODO: K&R: the meaning of '\\a' is different in traditional C
    OS << "'\\a'";
    break;
  case '\b':
    OS << "'\\b'";
    break;
  // Nonstandard escape sequence.
  /*case '\e':
    OS << "'\\e'";
    break;*/
  case '\f':
    OS << "'\\f'";
    break;
  case '\n':
    OS << "'\\n'";
    break;
  case '\r':
    OS << "'\\r'";
    break;
  case '\t':
    OS << "'\\t'";
    break;
  case '\v':
    OS << "'\\v'";
    break;
  default:
    if (value < 256 && isPrintable((unsigned char)value))
      OS << "'" << (char)value << "'";
    else if (value < 256)
      OS << "'\\x" << llvm::format("%02x", value) << "'";
    else if (value <= 0xFFFF)
      OS << "'\\u" << llvm::format("%04x", value) << "'";
    else
      OS << "'\\U" << llvm::format("%08x", value) << "'";
  }
}

void StmtPrinter::VisitIntegerLiteral(IntegerLiteral *Node) {
  bool isSigned = Node->getType()->isSignedIntegerType();
  OS << Node->getValue().toString(10, isSigned);

  // Emit suffixes.  Integer literals are always a builtin integer type.
  switch (Node->getType()->getAs<BuiltinType>()->getKind()) {
  default: llvm_unreachable("Unexpected type for integer literal!");
  // FIXME: The Short and UShort cases are to handle cases where a short
  // integeral literal is formed during template instantiation.  They should
  // be removed when template instantiation no longer needs integer literals.
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:       break; // no suffix.
  case BuiltinType::UInt:      OS << 'U'; break;
  case BuiltinType::Long:      OS << 'L'; break;
  case BuiltinType::ULong:     OS << "UL"; break;
  case BuiltinType::LongLong:  OS << "LL"; break;
  case BuiltinType::ULongLong: OS << "ULL"; break;
  case BuiltinType::Int128:    OS << "i128"; break;
  case BuiltinType::UInt128:   OS << "Ui128"; break;
  }
}

static void PrintFloatingLiteral(raw_ostream &OS, FloatingLiteral *Node,
                                 bool PrintSuffix) {
  SmallString<16> Str;
  Node->getValue().toString(Str);
  OS << Str;
  if (Str.find_first_not_of("-0123456789") == StringRef::npos)
    OS << '.'; // Trailing dot in order to separate from ints.

  if (!PrintSuffix)
    return;

  // Emit suffixes.  Float literals are always a builtin float type.
  switch (Node->getType()->getAs<BuiltinType>()->getKind()) {
  default: llvm_unreachable("Unexpected type for float literal!");
  case BuiltinType::Half:       break; // FIXME: suffix?
  case BuiltinType::Double:     break; // no suffix.
  case BuiltinType::Float:      OS << 'F'; break;
  case BuiltinType::LongDouble: OS << 'L'; break;
  }
}

void StmtPrinter::VisitFloatingLiteral(FloatingLiteral *Node) {
  PrintFloatingLiteral(OS, Node, /*PrintSuffix=*/true);
}

void StmtPrinter::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  PrintExpr(Node->getSubExpr());
  OS << "i";
}

void StmtPrinter::VisitStringLiteral(StringLiteral *Str) {
  Str->outputString(OS);
}
void StmtPrinter::VisitParenExpr(ParenExpr *Node) {
  OS << "(";
  PrintExpr(Node->getSubExpr());
  OS << ")";
}
void StmtPrinter::VisitUnaryOperator(UnaryOperator *Node) {
  if (!Node->isPostfix()) {
    OS << UnaryOperator::getOpcodeStr(Node->getOpcode());

    // Print a space if this is an "identifier operator" like __real, or if
    // it might be concatenated incorrectly like '+'.
    switch (Node->getOpcode()) {
    default: break;
    case UO_Real:
    case UO_Imag:
    case UO_Extension:
      OS << ' ';
      break;
    case UO_Plus:
    case UO_Minus:
      if (isa<UnaryOperator>(Node->getSubExpr()))
        OS << ' ';
      break;
    }
  }
  PrintExpr(Node->getSubExpr());

  if (Node->isPostfix())
    OS << UnaryOperator::getOpcodeStr(Node->getOpcode());
}

void StmtPrinter::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  OS << "__builtin_offsetof(";
  Node->getTypeSourceInfo()->getType().print(OS, Policy);
  OS << ", ";
  bool PrintedSomething = false;
  for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
    if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
      // Array node
      OS << "[";
      PrintExpr(Node->getIndexExpr(ON.getArrayExprIndex()));
      OS << "]";
      PrintedSomething = true;
      continue;
    }

    // Skip implicit base indirections.
    if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Base)
      continue;

    // Field or identifier node.
    IdentifierInfo *Id = ON.getFieldName();
    if (!Id)
      continue;
    
    if (PrintedSomething)
      OS << ".";
    else
      PrintedSomething = true;
    OS << Id->getName();    
  }
  OS << ")";
}

void StmtPrinter::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node){
  switch(Node->getKind()) {
  case UETT_SizeOf:
    OS << "sizeof";
    break;
  case UETT_AlignOf:
    if (Policy.LangOpts.CPlusPlus)
      OS << "alignof";
    else if (Policy.LangOpts.C11)
      OS << "_Alignof";
    else
      OS << "__alignof";
    break;
  case UETT_VecStep:
    OS << "vec_step";
    break;
  }
  if (Node->isArgumentType()) {
    OS << '(';
    Node->getArgumentType().print(OS, Policy);
    OS << ')';
  } else {
    OS << " ";
    PrintExpr(Node->getArgumentExpr());
  }
}

void StmtPrinter::VisitGenericSelectionExpr(GenericSelectionExpr *Node) {
  OS << "_Generic(";
  PrintExpr(Node->getControllingExpr());
  for (unsigned i = 0; i != Node->getNumAssocs(); ++i) {
    OS << ", ";
    QualType T = Node->getAssocType(i);
    if (T.isNull())
      OS << "default";
    else
      T.print(OS, Policy);
    OS << ": ";
    PrintExpr(Node->getAssocExpr(i));
  }
  OS << ")";
}

void StmtPrinter::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  PrintExpr(Node->getLHS());
  OS << "[";
  PrintExpr(Node->getRHS());
  OS << "]";
}

void StmtPrinter::PrintCallArgs(CallExpr *Call) {
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
      // Don't print any defaulted arguments
      break;
    }

    if (i) OS << ", ";
    PrintExpr(Call->getArg(i));
  }
}

void StmtPrinter::VisitCallExpr(CallExpr *Call) {
  PrintExpr(Call->getCallee());
  OS << "(";
  PrintCallArgs(Call);
  OS << ")";
}
void StmtPrinter::VisitMemberExpr(MemberExpr *Node) {
  // FIXME: Suppress printing implicit bases (like "this")
  PrintExpr(Node->getBase());

  MemberExpr *ParentMember = dyn_cast<MemberExpr>(Node->getBase());
  FieldDecl  *ParentDecl   = ParentMember
    ? dyn_cast<FieldDecl>(ParentMember->getMemberDecl()) : NULL;

  if (!ParentDecl || !ParentDecl->isAnonymousStructOrUnion())
    OS << (Node->isArrow() ? "->" : ".");

  if (FieldDecl *FD = dyn_cast<FieldDecl>(Node->getMemberDecl()))
    if (FD->isAnonymousStructOrUnion())
      return;

  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(OS, Policy);
  if (Node->hasTemplateKeyword())
    OS << "template ";
  OS << Node->getMemberNameInfo();
  if (Node->hasExplicitTemplateArgs())
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, Node->getTemplateArgs(), Node->getNumTemplateArgs(), Policy);
}
void StmtPrinter::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
  PrintExpr(Node->getBase());
  OS << (Node->isArrow() ? "->isa" : ".isa");
}

void StmtPrinter::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  PrintExpr(Node->getBase());
  OS << ".";
  OS << Node->getAccessor().getName();
}
void StmtPrinter::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  OS << '(';
  Node->getTypeAsWritten().print(OS, Policy);
  OS << ')';
  PrintExpr(Node->getSubExpr());
}
void StmtPrinter::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  OS << '(';
  Node->getType().print(OS, Policy);
  OS << ')';
  PrintExpr(Node->getInitializer());
}
void StmtPrinter::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  // No need to print anything, simply forward to the sub expression.
  PrintExpr(Node->getSubExpr());
}
void StmtPrinter::VisitBinaryOperator(BinaryOperator *Node) {
  PrintExpr(Node->getLHS());
  OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
  PrintExpr(Node->getRHS());
}
void StmtPrinter::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  PrintExpr(Node->getLHS());
  OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
  PrintExpr(Node->getRHS());
}
void StmtPrinter::VisitConditionalOperator(ConditionalOperator *Node) {
  PrintExpr(Node->getCond());
  OS << " ? ";
  PrintExpr(Node->getLHS());
  OS << " : ";
  PrintExpr(Node->getRHS());
}

// GNU extensions.

void
StmtPrinter::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
  PrintExpr(Node->getCommon());
  OS << " ?: ";
  PrintExpr(Node->getFalseExpr());
}
void StmtPrinter::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  OS << "&&" << Node->getLabel()->getName();
}

void StmtPrinter::VisitStmtExpr(StmtExpr *E) {
  OS << "(";
  PrintRawCompoundStmt(E->getSubStmt());
  OS << ")";
}

void StmtPrinter::VisitChooseExpr(ChooseExpr *Node) {
  OS << "__builtin_choose_expr(";
  PrintExpr(Node->getCond());
  OS << ", ";
  PrintExpr(Node->getLHS());
  OS << ", ";
  PrintExpr(Node->getRHS());
  OS << ")";
}

void StmtPrinter::VisitGNUNullExpr(GNUNullExpr *) {
  OS << "__null";
}

void StmtPrinter::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
  OS << "__builtin_shufflevector(";
  for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
    if (i) OS << ", ";
    PrintExpr(Node->getExpr(i));
  }
  OS << ")";
}

void StmtPrinter::VisitConvertVectorExpr(ConvertVectorExpr *Node) {
  OS << "__builtin_convertvector(";
  PrintExpr(Node->getSrcExpr());
  OS << ", ";
  Node->getType().print(OS, Policy);
  OS << ")";
}

void StmtPrinter::VisitInitListExpr(InitListExpr* Node) {
  if (Node->getSyntacticForm()) {
    Visit(Node->getSyntacticForm());
    return;
  }

  OS << "{ ";
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (i) OS << ", ";
    if (Node->getInit(i))
      PrintExpr(Node->getInit(i));
    else
      OS << "0";
  }
  OS << " }";
}

void StmtPrinter::VisitParenListExpr(ParenListExpr* Node) {
  OS << "( ";
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    if (i) OS << ", ";
    PrintExpr(Node->getExpr(i));
  }
  OS << " )";
}

void StmtPrinter::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (DesignatedInitExpr::designators_iterator D = Node->designators_begin(),
                      DEnd = Node->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      if (D->getDotLoc().isInvalid())
        OS << D->getFieldName()->getName() << ":";
      else
        OS << "." << D->getFieldName()->getName();
    } else {
      OS << "[";
      if (D->isArrayDesignator()) {
        PrintExpr(Node->getArrayIndex(*D));
      } else {
        PrintExpr(Node->getArrayRangeStart(*D));
        OS << " ... ";
        PrintExpr(Node->getArrayRangeEnd(*D));
      }
      OS << "]";
    }
  }

  OS << " = ";
  PrintExpr(Node->getInit());
}

void StmtPrinter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
  if (Policy.LangOpts.CPlusPlus) {
    OS << "/*implicit*/";
    Node->getType().print(OS, Policy);
    OS << "()";
  } else {
    OS << "/*implicit*/(";
    Node->getType().print(OS, Policy);
    OS << ')';
    if (Node->getType()->isRecordType())
      OS << "{}";
    else
      OS << 0;
  }
}

void StmtPrinter::VisitVAArgExpr(VAArgExpr *Node) {
  OS << "__builtin_va_arg(";
  PrintExpr(Node->getSubExpr());
  OS << ", ";
  Node->getType().print(OS, Policy);
  OS << ")";
}

void StmtPrinter::VisitPseudoObjectExpr(PseudoObjectExpr *Node) {
  PrintExpr(Node->getSyntacticForm());
}

void StmtPrinter::VisitAtomicExpr(AtomicExpr *Node) {
  const char *Name = 0;
  switch (Node->getOp()) {
#define BUILTIN(ID, TYPE, ATTRS)
#define ATOMIC_BUILTIN(ID, TYPE, ATTRS) \
  case AtomicExpr::AO ## ID: \
    Name = #ID "("; \
    break;
#include "clang/Basic/Builtins.def"
  }
  OS << Name;

  // AtomicExpr stores its subexpressions in a permuted order.
  PrintExpr(Node->getPtr());
  if (Node->getOp() != AtomicExpr::AO__c11_atomic_load &&
      Node->getOp() != AtomicExpr::AO__atomic_load_n) {
    OS << ", ";
    PrintExpr(Node->getVal1());
  }
  if (Node->getOp() == AtomicExpr::AO__atomic_exchange ||
      Node->isCmpXChg()) {
    OS << ", ";
    PrintExpr(Node->getVal2());
  }
  if (Node->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
      Node->getOp() == AtomicExpr::AO__atomic_compare_exchange_n) {
    OS << ", ";
    PrintExpr(Node->getWeak());
  }
  if (Node->getOp() != AtomicExpr::AO__c11_atomic_init) {
    OS << ", ";
    PrintExpr(Node->getOrder());
  }
  if (Node->isCmpXChg()) {
    OS << ", ";
    PrintExpr(Node->getOrderFail());
  }
  OS << ")";
}

// C++
void StmtPrinter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
  const char *OpStrings[NUM_OVERLOADED_OPERATORS] = {
    "",
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
    Spelling,
#include "clang/Basic/OperatorKinds.def"
  };

  OverloadedOperatorKind Kind = Node->getOperator();
  if (Kind == OO_PlusPlus || Kind == OO_MinusMinus) {
    if (Node->getNumArgs() == 1) {
      OS << OpStrings[Kind] << ' ';
      PrintExpr(Node->getArg(0));
    } else {
      PrintExpr(Node->getArg(0));
      OS << ' ' << OpStrings[Kind];
    }
  } else if (Kind == OO_Arrow) {
    PrintExpr(Node->getArg(0));
  } else if (Kind == OO_Call) {
    PrintExpr(Node->getArg(0));
    OS << '(';
    for (unsigned ArgIdx = 1; ArgIdx < Node->getNumArgs(); ++ArgIdx) {
      if (ArgIdx > 1)
        OS << ", ";
      if (!isa<CXXDefaultArgExpr>(Node->getArg(ArgIdx)))
        PrintExpr(Node->getArg(ArgIdx));
    }
    OS << ')';
  } else if (Kind == OO_Subscript) {
    PrintExpr(Node->getArg(0));
    OS << '[';
    PrintExpr(Node->getArg(1));
    OS << ']';
  } else if (Node->getNumArgs() == 1) {
    OS << OpStrings[Kind] << ' ';
    PrintExpr(Node->getArg(0));
  } else if (Node->getNumArgs() == 2) {
    PrintExpr(Node->getArg(0));
    OS << ' ' << OpStrings[Kind] << ' ';
    PrintExpr(Node->getArg(1));
  } else {
    llvm_unreachable("unknown overloaded operator");
  }
}

void StmtPrinter::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
  VisitCallExpr(cast<CallExpr>(Node));
}

void StmtPrinter::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
  PrintExpr(Node->getCallee());
  OS << "<<<";
  PrintCallArgs(Node->getConfig());
  OS << ">>>(";
  PrintCallArgs(Node);
  OS << ")";
}

void StmtPrinter::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  OS << Node->getCastName() << '<';
  Node->getTypeAsWritten().print(OS, Policy);
  OS << ">(";
  PrintExpr(Node->getSubExpr());
  OS << ")";
}

void StmtPrinter::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void StmtPrinter::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void StmtPrinter::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void StmtPrinter::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void StmtPrinter::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
  OS << "typeid(";
  if (Node->isTypeOperand()) {
    Node->getTypeOperandSourceInfo()->getType().print(OS, Policy);
  } else {
    PrintExpr(Node->getExprOperand());
  }
  OS << ")";
}

void StmtPrinter::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
  OS << "__uuidof(";
  if (Node->isTypeOperand()) {
    Node->getTypeOperandSourceInfo()->getType().print(OS, Policy);
  } else {
    PrintExpr(Node->getExprOperand());
  }
  OS << ")";
}

void StmtPrinter::VisitMSPropertyRefExpr(MSPropertyRefExpr *Node) {
  PrintExpr(Node->getBaseExpr());
  if (Node->isArrow())
    OS << "->";
  else
    OS << ".";
  if (NestedNameSpecifier *Qualifier =
      Node->getQualifierLoc().getNestedNameSpecifier())
    Qualifier->print(OS, Policy);
  OS << Node->getPropertyDecl()->getDeclName();
}

void StmtPrinter::VisitUserDefinedLiteral(UserDefinedLiteral *Node) {
  switch (Node->getLiteralOperatorKind()) {
  case UserDefinedLiteral::LOK_Raw:
    OS << cast<StringLiteral>(Node->getArg(0)->IgnoreImpCasts())->getString();
    break;
  case UserDefinedLiteral::LOK_Template: {
    DeclRefExpr *DRE = cast<DeclRefExpr>(Node->getCallee()->IgnoreImpCasts());
    const TemplateArgumentList *Args =
      cast<FunctionDecl>(DRE->getDecl())->getTemplateSpecializationArgs();
    assert(Args);
    const TemplateArgument &Pack = Args->get(0);
    for (TemplateArgument::pack_iterator I = Pack.pack_begin(),
                                         E = Pack.pack_end(); I != E; ++I) {
      char C = (char)I->getAsIntegral().getZExtValue();
      OS << C;
    }
    break;
  }
  case UserDefinedLiteral::LOK_Integer: {
    // Print integer literal without suffix.
    IntegerLiteral *Int = cast<IntegerLiteral>(Node->getCookedLiteral());
    OS << Int->getValue().toString(10, /*isSigned*/false);
    break;
  }
  case UserDefinedLiteral::LOK_Floating: {
    // Print floating literal without suffix.
    FloatingLiteral *Float = cast<FloatingLiteral>(Node->getCookedLiteral());
    PrintFloatingLiteral(OS, Float, /*PrintSuffix=*/false);
    break;
  }
  case UserDefinedLiteral::LOK_String:
  case UserDefinedLiteral::LOK_Character:
    PrintExpr(Node->getCookedLiteral());
    break;
  }
  OS << Node->getUDSuffix()->getName();
}

void StmtPrinter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  OS << (Node->getValue() ? "true" : "false");
}

void StmtPrinter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
  OS << "nullptr";
}

void StmtPrinter::VisitCXXThisExpr(CXXThisExpr *Node) {
  OS << "this";
}

void StmtPrinter::VisitCXXThrowExpr(CXXThrowExpr *Node) {
  if (Node->getSubExpr() == 0)
    OS << "throw";
  else {
    OS << "throw ";
    PrintExpr(Node->getSubExpr());
  }
}

void StmtPrinter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
  // Nothing to print: we picked up the default argument.
}

void StmtPrinter::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *Node) {
  // Nothing to print: we picked up the default initializer.
}

void StmtPrinter::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  Node->getType().print(OS, Policy);
  OS << "(";
  PrintExpr(Node->getSubExpr());
  OS << ")";
}

void StmtPrinter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
  PrintExpr(Node->getSubExpr());
}

void StmtPrinter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
  Node->getType().print(OS, Policy);
  OS << "(";
  for (CXXTemporaryObjectExpr::arg_iterator Arg = Node->arg_begin(),
                                         ArgEnd = Node->arg_end();
       Arg != ArgEnd; ++Arg) {
    if (Arg->isDefaultArgument())
      break;
    if (Arg != Node->arg_begin())
      OS << ", ";
    PrintExpr(*Arg);
  }
  OS << ")";
}

void StmtPrinter::VisitLambdaExpr(LambdaExpr *Node) {
  OS << '[';
  bool NeedComma = false;
  switch (Node->getCaptureDefault()) {
  case LCD_None:
    break;

  case LCD_ByCopy:
    OS << '=';
    NeedComma = true;
    break;

  case LCD_ByRef:
    OS << '&';
    NeedComma = true;
    break;
  }
  for (LambdaExpr::capture_iterator C = Node->explicit_capture_begin(),
                                 CEnd = Node->explicit_capture_end();
       C != CEnd;
       ++C) {
    if (NeedComma)
      OS << ", ";
    NeedComma = true;

    switch (C->getCaptureKind()) {
    case LCK_This:
      OS << "this";
      break;

    case LCK_ByRef:
      if (Node->getCaptureDefault() != LCD_ByRef || C->isInitCapture())
        OS << '&';
      OS << C->getCapturedVar()->getName();
      break;

    case LCK_ByCopy:
      OS << C->getCapturedVar()->getName();
      break;
    }

    if (C->isInitCapture())
      PrintExpr(C->getCapturedVar()->getInit());
  }
  OS << ']';

  if (Node->hasExplicitParameters()) {
    OS << " (";
    CXXMethodDecl *Method = Node->getCallOperator();
    NeedComma = false;
    for (CXXMethodDecl::param_iterator P = Method->param_begin(),
                                    PEnd = Method->param_end();
         P != PEnd; ++P) {
      if (NeedComma) {
        OS << ", ";
      } else {
        NeedComma = true;
      }
      std::string ParamStr = (*P)->getNameAsString();
      (*P)->getOriginalType().print(OS, Policy, ParamStr);
    }
    if (Method->isVariadic()) {
      if (NeedComma)
        OS << ", ";
      OS << "...";
    }
    OS << ')';

    if (Node->isMutable())
      OS << " mutable";

    const FunctionProtoType *Proto
      = Method->getType()->getAs<FunctionProtoType>();
    Proto->printExceptionSpecification(OS, Policy);

    // FIXME: Attributes

    // Print the trailing return type if it was specified in the source.
    if (Node->hasExplicitResultType()) {
      OS << " -> ";
      Proto->getResultType().print(OS, Policy);
    }
  }

  // Print the body.
  CompoundStmt *Body = Node->getBody();
  OS << ' ';
  PrintStmt(Body);
}

void StmtPrinter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
  if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
    TSInfo->getType().print(OS, Policy);
  else
    Node->getType().print(OS, Policy);
  OS << "()";
}

void StmtPrinter::VisitCXXNewExpr(CXXNewExpr *E) {
  if (E->isGlobalNew())
    OS << "::";
  OS << "new ";
  unsigned NumPlace = E->getNumPlacementArgs();
  if (NumPlace > 0 && !isa<CXXDefaultArgExpr>(E->getPlacementArg(0))) {
    OS << "(";
    PrintExpr(E->getPlacementArg(0));
    for (unsigned i = 1; i < NumPlace; ++i) {
      if (isa<CXXDefaultArgExpr>(E->getPlacementArg(i)))
        break;
      OS << ", ";
      PrintExpr(E->getPlacementArg(i));
    }
    OS << ") ";
  }
  if (E->isParenTypeId())
    OS << "(";
  std::string TypeS;
  if (Expr *Size = E->getArraySize()) {
    llvm::raw_string_ostream s(TypeS);
    s << '[';
    Size->printPretty(s, Helper, Policy);
    s << ']';
  }
  E->getAllocatedType().print(OS, Policy, TypeS);
  if (E->isParenTypeId())
    OS << ")";

  CXXNewExpr::InitializationStyle InitStyle = E->getInitializationStyle();
  if (InitStyle) {
    if (InitStyle == CXXNewExpr::CallInit)
      OS << "(";
    PrintExpr(E->getInitializer());
    if (InitStyle == CXXNewExpr::CallInit)
      OS << ")";
  }
}

void StmtPrinter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  if (E->isGlobalDelete())
    OS << "::";
  OS << "delete ";
  if (E->isArrayForm())
    OS << "[] ";
  PrintExpr(E->getArgument());
}

void StmtPrinter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  PrintExpr(E->getBase());
  if (E->isArrow())
    OS << "->";
  else
    OS << '.';
  if (E->getQualifier())
    E->getQualifier()->print(OS, Policy);
  OS << "~";

  if (IdentifierInfo *II = E->getDestroyedTypeIdentifier())
    OS << II->getName();
  else
    E->getDestroyedType().print(OS, Policy);
}

void StmtPrinter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  if (E->isListInitialization())
    OS << "{ ";

  for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
    if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
      // Don't print any defaulted arguments
      break;
    }

    if (i) OS << ", ";
    PrintExpr(E->getArg(i));
  }

  if (E->isListInitialization())
    OS << " }";
}

void StmtPrinter::VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
  PrintExpr(E->getSubExpr());
}

void StmtPrinter::VisitExprWithCleanups(ExprWithCleanups *E) {
  // Just forward to the sub expression.
  PrintExpr(E->getSubExpr());
}

void
StmtPrinter::VisitCXXUnresolvedConstructExpr(
                                           CXXUnresolvedConstructExpr *Node) {
  Node->getTypeAsWritten().print(OS, Policy);
  OS << "(";
  for (CXXUnresolvedConstructExpr::arg_iterator Arg = Node->arg_begin(),
                                             ArgEnd = Node->arg_end();
       Arg != ArgEnd; ++Arg) {
    if (Arg != Node->arg_begin())
      OS << ", ";
    PrintExpr(*Arg);
  }
  OS << ")";
}

void StmtPrinter::VisitCXXDependentScopeMemberExpr(
                                         CXXDependentScopeMemberExpr *Node) {
  if (!Node->isImplicitAccess()) {
    PrintExpr(Node->getBase());
    OS << (Node->isArrow() ? "->" : ".");
  }
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(OS, Policy);
  if (Node->hasTemplateKeyword())
    OS << "template ";
  OS << Node->getMemberNameInfo();
  if (Node->hasExplicitTemplateArgs())
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, Node->getTemplateArgs(), Node->getNumTemplateArgs(), Policy);
}

void StmtPrinter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
  if (!Node->isImplicitAccess()) {
    PrintExpr(Node->getBase());
    OS << (Node->isArrow() ? "->" : ".");
  }
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(OS, Policy);
  if (Node->hasTemplateKeyword())
    OS << "template ";
  OS << Node->getMemberNameInfo();
  if (Node->hasExplicitTemplateArgs())
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, Node->getTemplateArgs(), Node->getNumTemplateArgs(), Policy);
}

static const char *getTypeTraitName(UnaryTypeTrait UTT) {
  switch (UTT) {
  case UTT_HasNothrowAssign:      return "__has_nothrow_assign";
  case UTT_HasNothrowMoveAssign:  return "__has_nothrow_move_assign";
  case UTT_HasNothrowConstructor: return "__has_nothrow_constructor";
  case UTT_HasNothrowCopy:          return "__has_nothrow_copy";
  case UTT_HasTrivialAssign:      return "__has_trivial_assign";
  case UTT_HasTrivialMoveAssign:      return "__has_trivial_move_assign";
  case UTT_HasTrivialMoveConstructor: return "__has_trivial_move_constructor";
  case UTT_HasTrivialDefaultConstructor: return "__has_trivial_constructor";
  case UTT_HasTrivialCopy:          return "__has_trivial_copy";
  case UTT_HasTrivialDestructor:  return "__has_trivial_destructor";
  case UTT_HasVirtualDestructor:  return "__has_virtual_destructor";
  case UTT_IsAbstract:            return "__is_abstract";
  case UTT_IsArithmetic:            return "__is_arithmetic";
  case UTT_IsArray:                 return "__is_array";
  case UTT_IsClass:               return "__is_class";
  case UTT_IsCompleteType:          return "__is_complete_type";
  case UTT_IsCompound:              return "__is_compound";
  case UTT_IsConst:                 return "__is_const";
  case UTT_IsEmpty:               return "__is_empty";
  case UTT_IsEnum:                return "__is_enum";
  case UTT_IsFinal:                 return "__is_final";
  case UTT_IsFloatingPoint:         return "__is_floating_point";
  case UTT_IsFunction:              return "__is_function";
  case UTT_IsFundamental:           return "__is_fundamental";
  case UTT_IsIntegral:              return "__is_integral";
  case UTT_IsInterfaceClass:        return "__is_interface_class";
  case UTT_IsLiteral:               return "__is_literal";
  case UTT_IsLvalueReference:       return "__is_lvalue_reference";
  case UTT_IsMemberFunctionPointer: return "__is_member_function_pointer";
  case UTT_IsMemberObjectPointer:   return "__is_member_object_pointer";
  case UTT_IsMemberPointer:         return "__is_member_pointer";
  case UTT_IsObject:                return "__is_object";
  case UTT_IsPOD:                 return "__is_pod";
  case UTT_IsPointer:               return "__is_pointer";
  case UTT_IsPolymorphic:         return "__is_polymorphic";
  case UTT_IsReference:             return "__is_reference";
  case UTT_IsRvalueReference:       return "__is_rvalue_reference";
  case UTT_IsScalar:                return "__is_scalar";
  case UTT_IsSealed:                return "__is_sealed";
  case UTT_IsSigned:                return "__is_signed";
  case UTT_IsStandardLayout:        return "__is_standard_layout";
  case UTT_IsTrivial:               return "__is_trivial";
  case UTT_IsTriviallyCopyable:     return "__is_trivially_copyable";
  case UTT_IsUnion:               return "__is_union";
  case UTT_IsUnsigned:              return "__is_unsigned";
  case UTT_IsVoid:                  return "__is_void";
  case UTT_IsVolatile:              return "__is_volatile";
  }
  llvm_unreachable("Type trait not covered by switch statement");
}

static const char *getTypeTraitName(BinaryTypeTrait BTT) {
  switch (BTT) {
  case BTT_IsBaseOf:              return "__is_base_of";
  case BTT_IsConvertible:         return "__is_convertible";
  case BTT_IsSame:                return "__is_same";
  case BTT_TypeCompatible:        return "__builtin_types_compatible_p";
  case BTT_IsConvertibleTo:       return "__is_convertible_to";
  case BTT_IsTriviallyAssignable: return "__is_trivially_assignable";
  }
  llvm_unreachable("Binary type trait not covered by switch");
}

static const char *getTypeTraitName(TypeTrait TT) {
  switch (TT) {
  case clang::TT_IsTriviallyConstructible:return "__is_trivially_constructible";
  }
  llvm_unreachable("Type trait not covered by switch");
}

static const char *getTypeTraitName(ArrayTypeTrait ATT) {
  switch (ATT) {
  case ATT_ArrayRank:        return "__array_rank";
  case ATT_ArrayExtent:      return "__array_extent";
  }
  llvm_unreachable("Array type trait not covered by switch");
}

static const char *getExpressionTraitName(ExpressionTrait ET) {
  switch (ET) {
  case ET_IsLValueExpr:      return "__is_lvalue_expr";
  case ET_IsRValueExpr:      return "__is_rvalue_expr";
  }
  llvm_unreachable("Expression type trait not covered by switch");
}

void StmtPrinter::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  OS << getTypeTraitName(E->getTrait()) << '(';
  E->getQueriedType().print(OS, Policy);
  OS << ')';
}

void StmtPrinter::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  OS << getTypeTraitName(E->getTrait()) << '(';
  E->getLhsType().print(OS, Policy);
  OS << ',';
  E->getRhsType().print(OS, Policy);
  OS << ')';
}

void StmtPrinter::VisitTypeTraitExpr(TypeTraitExpr *E) {
  OS << getTypeTraitName(E->getTrait()) << "(";
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    if (I > 0)
      OS << ", ";
    E->getArg(I)->getType().print(OS, Policy);
  }
  OS << ")";
}

void StmtPrinter::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  OS << getTypeTraitName(E->getTrait()) << '(';
  E->getQueriedType().print(OS, Policy);
  OS << ')';
}

void StmtPrinter::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  OS << getExpressionTraitName(E->getTrait()) << '(';
  PrintExpr(E->getQueriedExpression());
  OS << ')';
}

void StmtPrinter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  OS << "noexcept(";
  PrintExpr(E->getOperand());
  OS << ")";
}

void StmtPrinter::VisitPackExpansionExpr(PackExpansionExpr *E) {
  PrintExpr(E->getPattern());
  OS << "...";
}

void StmtPrinter::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  OS << "sizeof...(" << *E->getPack() << ")";
}

void StmtPrinter::VisitSubstNonTypeTemplateParmPackExpr(
                                       SubstNonTypeTemplateParmPackExpr *Node) {
  OS << *Node->getParameterPack();
}

void StmtPrinter::VisitSubstNonTypeTemplateParmExpr(
                                       SubstNonTypeTemplateParmExpr *Node) {
  Visit(Node->getReplacement());
}

void StmtPrinter::VisitFunctionParmPackExpr(FunctionParmPackExpr *E) {
  OS << *E->getParameterPack();
}

void StmtPrinter::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *Node){
  PrintExpr(Node->GetTemporaryExpr());
}

// Obj-C

void StmtPrinter::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
  OS << "@";
  VisitStringLiteral(Node->getString());
}

void StmtPrinter::VisitObjCBoxedExpr(ObjCBoxedExpr *E) {
  OS << "@";
  Visit(E->getSubExpr());
}

void StmtPrinter::VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
  OS << "@[ ";
  StmtRange ch = E->children();
  if (ch.first != ch.second) {
    while (1) {
      Visit(*ch.first);
      ++ch.first;
      if (ch.first == ch.second) break;
      OS << ", ";
    }
  }
  OS << " ]";
}

void StmtPrinter::VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
  OS << "@{ ";
  for (unsigned I = 0, N = E->getNumElements(); I != N; ++I) {
    if (I > 0)
      OS << ", ";
    
    ObjCDictionaryElement Element = E->getKeyValueElement(I);
    Visit(Element.Key);
    OS << " : ";
    Visit(Element.Value);
    if (Element.isPackExpansion())
      OS << "...";
  }
  OS << " }";
}

void StmtPrinter::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  OS << "@encode(";
  Node->getEncodedType().print(OS, Policy);
  OS << ')';
}

void StmtPrinter::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  OS << "@selector(" << Node->getSelector().getAsString() << ')';
}

void StmtPrinter::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  OS << "@protocol(" << *Node->getProtocol() << ')';
}

void StmtPrinter::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
  OS << "[";
  switch (Mess->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    PrintExpr(Mess->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class:
    Mess->getClassReceiver().print(OS, Policy);
    break;

  case ObjCMessageExpr::SuperInstance:
  case ObjCMessageExpr::SuperClass:
    OS << "Super";
    break;
  }

  OS << ' ';
  Selector selector = Mess->getSelector();
  if (selector.isUnarySelector()) {
    OS << selector.getNameForSlot(0);
  } else {
    for (unsigned i = 0, e = Mess->getNumArgs(); i != e; ++i) {
      if (i < selector.getNumArgs()) {
        if (i > 0) OS << ' ';
        if (selector.getIdentifierInfoForSlot(i))
          OS << selector.getIdentifierInfoForSlot(i)->getName() << ':';
        else
           OS << ":";
      }
      else OS << ", "; // Handle variadic methods.

      PrintExpr(Mess->getArg(i));
    }
  }
  OS << "]";
}

void StmtPrinter::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node) {
  OS << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

void
StmtPrinter::VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *E) {
  PrintExpr(E->getSubExpr());
}

void
StmtPrinter::VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *E) {
  OS << '(' << E->getBridgeKindName();
  E->getType().print(OS, Policy);
  OS << ')';
  PrintExpr(E->getSubExpr());
}

void StmtPrinter::VisitBlockExpr(BlockExpr *Node) {
  BlockDecl *BD = Node->getBlockDecl();
  OS << "^";

  const FunctionType *AFT = Node->getFunctionType();

  if (isa<FunctionNoProtoType>(AFT)) {
    OS << "()";
  } else if (!BD->param_empty() || cast<FunctionProtoType>(AFT)->isVariadic()) {
    OS << '(';
    for (BlockDecl::param_iterator AI = BD->param_begin(),
         E = BD->param_end(); AI != E; ++AI) {
      if (AI != BD->param_begin()) OS << ", ";
      std::string ParamStr = (*AI)->getNameAsString();
      (*AI)->getType().print(OS, Policy, ParamStr);
    }

    const FunctionProtoType *FT = cast<FunctionProtoType>(AFT);
    if (FT->isVariadic()) {
      if (!BD->param_empty()) OS << ", ";
      OS << "...";
    }
    OS << ')';
  }
  OS << "{ }";
}

void StmtPrinter::VisitOpaqueValueExpr(OpaqueValueExpr *Node) { 
  PrintExpr(Node->getSourceExpr());
}

void StmtPrinter::VisitAsTypeExpr(AsTypeExpr *Node) {
  OS << "__builtin_astype(";
  PrintExpr(Node->getSrcExpr());
  OS << ", ";
  Node->getType().print(OS, Policy);
  OS << ")";
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

void Stmt::dumpPretty(const ASTContext &Context) const {
  printPretty(llvm::errs(), 0, PrintingPolicy(Context.getLangOpts()));
}

void Stmt::printPretty(raw_ostream &OS,
                       PrinterHelper *Helper,
                       const PrintingPolicy &Policy,
                       unsigned Indentation) const {
  if (this == 0) {
    OS << "<NULL>";
    return;
  }

  StmtPrinter P(OS, Helper, Policy, Indentation);
  P.Visit(const_cast<Stmt*>(this));
}

//===----------------------------------------------------------------------===//
// PrinterHelper
//===----------------------------------------------------------------------===//

// Implement virtual destructor.
PrinterHelper::~PrinterHelper() {}
