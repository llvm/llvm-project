//===--- ASTConsumers.cpp - ASTConsumer implementations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumer Implementations.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"
using namespace clang;

//===----------------------------------------------------------------------===//
/// ASTPrinter - Pretty-printer and dumper of ASTs

namespace {
  class ASTPrinter : public ASTConsumer,
                     public RecursiveASTVisitor<ASTPrinter> {
    typedef RecursiveASTVisitor<ASTPrinter> base;

  public:
    ASTPrinter(raw_ostream *Out = NULL, bool Dump = false,
               StringRef FilterString = "")
        : Out(Out ? *Out : llvm::outs()), Dump(Dump),
          FilterString(FilterString) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
      TranslationUnitDecl *D = Context.getTranslationUnitDecl();

      if (FilterString.empty()) {
        if (Dump)
          D->dump(Out);
        else
          D->print(Out, /*Indentation=*/0, /*PrintInstantiation=*/true);
        return;
      }

      TraverseDecl(D);
    }

    bool shouldWalkTypesOfTypeLocs() const { return false; }

    bool TraverseDecl(Decl *D) {
      if (D != NULL && filterMatches(D)) {
        Out.changeColor(llvm::raw_ostream::BLUE) <<
            (Dump ? "Dumping " : "Printing ") << getName(D) << ":\n";
        Out.resetColor();
        if (Dump)
          D->dump(Out);
        else
          D->print(Out, /*Indentation=*/0, /*PrintInstantiation=*/true);
        Out << "\n";
        // Don't traverse child nodes to avoid output duplication.
        return true;
      }
      return base::TraverseDecl(D);
    }

  private:
    std::string getName(Decl *D) {
      if (isa<NamedDecl>(D))
        return cast<NamedDecl>(D)->getQualifiedNameAsString();
      return "";
    }
    bool filterMatches(Decl *D) {
      return getName(D).find(FilterString) != std::string::npos;
    }

    raw_ostream &Out;
    bool Dump;
    std::string FilterString;
  };

  class ASTDeclNodeLister : public ASTConsumer,
                     public RecursiveASTVisitor<ASTDeclNodeLister> {
  public:
    ASTDeclNodeLister(raw_ostream *Out = NULL)
        : Out(Out ? *Out : llvm::outs()) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
      TraverseDecl(Context.getTranslationUnitDecl());
    }

    bool shouldWalkTypesOfTypeLocs() const { return false; }

    virtual bool VisitNamedDecl(NamedDecl *D) {
      Out << D->getQualifiedNameAsString() << "\n";
      return true;
    }

  private:
    raw_ostream &Out;
  };
} // end anonymous namespace

ASTConsumer *clang::CreateASTPrinter(raw_ostream *Out,
                                     StringRef FilterString) {
  return new ASTPrinter(Out, /*Dump=*/ false, FilterString);
}

ASTConsumer *clang::CreateASTDumper(StringRef FilterString) {
  return new ASTPrinter(0, /*Dump=*/ true, FilterString);
}

ASTConsumer *clang::CreateASTDeclNodeLister() {
  return new ASTDeclNodeLister(0);
}

//===----------------------------------------------------------------------===//
/// ASTViewer - AST Visualization

namespace {
  class ASTViewer : public ASTConsumer {
    ASTContext *Context;
  public:
    void Initialize(ASTContext &Context) {
      this->Context = &Context;
    }

    virtual bool HandleTopLevelDecl(DeclGroupRef D) {
      for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
        HandleTopLevelSingleDecl(*I);
      return true;
    }

    void HandleTopLevelSingleDecl(Decl *D);
  };
}

void ASTViewer::HandleTopLevelSingleDecl(Decl *D) {
  if (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)) {
    D->print(llvm::errs());
  
    if (Stmt *Body = D->getBody()) {
      llvm::errs() << '\n';
      Body->viewAST();
      llvm::errs() << '\n';
    }
  }
}


ASTConsumer *clang::CreateASTViewer() { return new ASTViewer(); }

//===----------------------------------------------------------------------===//
/// DeclContextPrinter - Decl and DeclContext Visualization

namespace {

class DeclContextPrinter : public ASTConsumer {
  raw_ostream& Out;
public:
  DeclContextPrinter() : Out(llvm::errs()) {}

  void HandleTranslationUnit(ASTContext &C) {
    PrintDeclContext(C.getTranslationUnitDecl(), 4);
  }

  void PrintDeclContext(const DeclContext* DC, unsigned Indentation);
};
}  // end anonymous namespace

void DeclContextPrinter::PrintDeclContext(const DeclContext* DC,
                                          unsigned Indentation) {
  // Print DeclContext name.
  switch (DC->getDeclKind()) {
  case Decl::TranslationUnit:
    Out << "[translation unit] " << DC;
    break;
  case Decl::Namespace: {
    Out << "[namespace] ";
    const NamespaceDecl* ND = cast<NamespaceDecl>(DC);
    Out << *ND;
    break;
  }
  case Decl::Enum: {
    const EnumDecl* ED = cast<EnumDecl>(DC);
    if (ED->isCompleteDefinition())
      Out << "[enum] ";
    else
      Out << "<enum> ";
    Out << *ED;
    break;
  }
  case Decl::Record: {
    const RecordDecl* RD = cast<RecordDecl>(DC);
    if (RD->isCompleteDefinition())
      Out << "[struct] ";
    else
      Out << "<struct> ";
    Out << *RD;
    break;
  }
  case Decl::CXXRecord: {
    const CXXRecordDecl* RD = cast<CXXRecordDecl>(DC);
    if (RD->isCompleteDefinition())
      Out << "[class] ";
    else
      Out << "<class> ";
    Out << *RD << ' ' << DC;
    break;
  }
  case Decl::ObjCMethod:
    Out << "[objc method]";
    break;
  case Decl::ObjCInterface:
    Out << "[objc interface]";
    break;
  case Decl::ObjCCategory:
    Out << "[objc category]";
    break;
  case Decl::ObjCProtocol:
    Out << "[objc protocol]";
    break;
  case Decl::ObjCImplementation:
    Out << "[objc implementation]";
    break;
  case Decl::ObjCCategoryImpl:
    Out << "[objc categoryimpl]";
    break;
  case Decl::LinkageSpec:
    Out << "[linkage spec]";
    break;
  case Decl::Block:
    Out << "[block]";
    break;
  case Decl::Function: {
    const FunctionDecl* FD = cast<FunctionDecl>(DC);
    if (FD->doesThisDeclarationHaveABody())
      Out << "[function] ";
    else
      Out << "<function> ";
    Out << *FD;
    // Print the parameters.
    Out << "(";
    bool PrintComma = false;
    for (FunctionDecl::param_const_iterator I = FD->param_begin(),
           E = FD->param_end(); I != E; ++I) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << **I;
    }
    Out << ")";
    break;
  }
  case Decl::CXXMethod: {
    const CXXMethodDecl* D = cast<CXXMethodDecl>(DC);
    if (D->isOutOfLine())
      Out << "[c++ method] ";
    else if (D->isImplicit())
      Out << "(c++ method) ";
    else
      Out << "<c++ method> ";
    Out << *D;
    // Print the parameters.
    Out << "(";
    bool PrintComma = false;
    for (FunctionDecl::param_const_iterator I = D->param_begin(),
           E = D->param_end(); I != E; ++I) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << **I;
    }
    Out << ")";

    // Check the semantic DeclContext.
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";

    break;
  }
  case Decl::CXXConstructor: {
    const CXXConstructorDecl* D = cast<CXXConstructorDecl>(DC);
    if (D->isOutOfLine())
      Out << "[c++ ctor] ";
    else if (D->isImplicit())
      Out << "(c++ ctor) ";
    else
      Out << "<c++ ctor> ";
    Out << *D;
    // Print the parameters.
    Out << "(";
    bool PrintComma = false;
    for (FunctionDecl::param_const_iterator I = D->param_begin(),
           E = D->param_end(); I != E; ++I) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << **I;
    }
    Out << ")";

    // Check the semantic DC.
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }
  case Decl::CXXDestructor: {
    const CXXDestructorDecl* D = cast<CXXDestructorDecl>(DC);
    if (D->isOutOfLine())
      Out << "[c++ dtor] ";
    else if (D->isImplicit())
      Out << "(c++ dtor) ";
    else
      Out << "<c++ dtor> ";
    Out << *D;
    // Check the semantic DC.
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }
  case Decl::CXXConversion: {
    const CXXConversionDecl* D = cast<CXXConversionDecl>(DC);
    if (D->isOutOfLine())
      Out << "[c++ conversion] ";
    else if (D->isImplicit())
      Out << "(c++ conversion) ";
    else
      Out << "<c++ conversion> ";
    Out << *D;
    // Check the semantic DC.
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }

  default:
    llvm_unreachable("a decl that inherits DeclContext isn't handled");
  }

  Out << "\n";

  // Print decls in the DeclContext.
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I) {
    for (unsigned i = 0; i < Indentation; ++i)
      Out << "  ";

    Decl::Kind DK = I->getKind();
    switch (DK) {
    case Decl::Namespace:
    case Decl::Enum:
    case Decl::Record:
    case Decl::CXXRecord:
    case Decl::ObjCMethod:
    case Decl::ObjCInterface:
    case Decl::ObjCCategory:
    case Decl::ObjCProtocol:
    case Decl::ObjCImplementation:
    case Decl::ObjCCategoryImpl:
    case Decl::LinkageSpec:
    case Decl::Block:
    case Decl::Function:
    case Decl::CXXMethod:
    case Decl::CXXConstructor:
    case Decl::CXXDestructor:
    case Decl::CXXConversion:
    {
      DeclContext* DC = cast<DeclContext>(*I);
      PrintDeclContext(DC, Indentation+2);
      break;
    }
    case Decl::IndirectField: {
      IndirectFieldDecl* IFD = cast<IndirectFieldDecl>(*I);
      Out << "<IndirectField> " << *IFD << '\n';
      break;
    }
    case Decl::Label: {
      LabelDecl *LD = cast<LabelDecl>(*I);
      Out << "<Label> " << *LD << '\n';
      break;
    }
    case Decl::Field: {
      FieldDecl *FD = cast<FieldDecl>(*I);
      Out << "<field> " << *FD << '\n';
      break;
    }
    case Decl::Typedef:
    case Decl::TypeAlias: {
      TypedefNameDecl* TD = cast<TypedefNameDecl>(*I);
      Out << "<typedef> " << *TD << '\n';
      break;
    }
    case Decl::EnumConstant: {
      EnumConstantDecl* ECD = cast<EnumConstantDecl>(*I);
      Out << "<enum constant> " << *ECD << '\n';
      break;
    }
    case Decl::Var: {
      VarDecl* VD = cast<VarDecl>(*I);
      Out << "<var> " << *VD << '\n';
      break;
    }
    case Decl::ImplicitParam: {
      ImplicitParamDecl* IPD = cast<ImplicitParamDecl>(*I);
      Out << "<implicit parameter> " << *IPD << '\n';
      break;
    }
    case Decl::ParmVar: {
      ParmVarDecl* PVD = cast<ParmVarDecl>(*I);
      Out << "<parameter> " << *PVD << '\n';
      break;
    }
    case Decl::ObjCProperty: {
      ObjCPropertyDecl* OPD = cast<ObjCPropertyDecl>(*I);
      Out << "<objc property> " << *OPD << '\n';
      break;
    }
    case Decl::FunctionTemplate: {
      FunctionTemplateDecl* FTD = cast<FunctionTemplateDecl>(*I);
      Out << "<function template> " << *FTD << '\n';
      break;
    }
    case Decl::FileScopeAsm: {
      Out << "<file-scope asm>\n";
      break;
    }
    case Decl::UsingDirective: {
      Out << "<using directive>\n";
      break;
    }
    case Decl::NamespaceAlias: {
      NamespaceAliasDecl* NAD = cast<NamespaceAliasDecl>(*I);
      Out << "<namespace alias> " << *NAD << '\n';
      break;
    }
    case Decl::ClassTemplate: {
      ClassTemplateDecl *CTD = cast<ClassTemplateDecl>(*I);
      Out << "<class template> " << *CTD << '\n';
      break;
    }
    default:
      Out << "DeclKind: " << DK << '"' << *I << "\"\n";
      llvm_unreachable("decl unhandled");
    }
  }
}
ASTConsumer *clang::CreateDeclContextPrinter() {
  return new DeclContextPrinter();
}

//===----------------------------------------------------------------------===//
/// ASTDumperXML - In-depth XML dumping.

namespace {
class ASTDumpXML : public ASTConsumer {
  raw_ostream &OS;

public:
  ASTDumpXML(raw_ostream &OS) : OS(OS) {}

  void HandleTranslationUnit(ASTContext &C) {
    C.getTranslationUnitDecl()->dumpXML(OS);
  }  
};
}

ASTConsumer *clang::CreateASTDumperXML(raw_ostream &OS) {
  return new ASTDumpXML(OS);
}
