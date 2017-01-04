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
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

//===----------------------------------------------------------------------===//
/// ASTPrinter - Pretty-printer and dumper of ASTs

namespace {
  class ASTPrinter : public ASTConsumer,
                     public RecursiveASTVisitor<ASTPrinter> {
    typedef RecursiveASTVisitor<ASTPrinter> base;

  public:
    ASTPrinter(std::unique_ptr<raw_ostream> Out = nullptr, bool Dump = false,
               StringRef FilterString = "", bool DumpLookups = false)
        : Out(Out ? *Out : llvm::outs()), OwnedOut(std::move(Out)), Dump(Dump),
          FilterString(FilterString), DumpLookups(DumpLookups) {}

    void HandleTranslationUnit(ASTContext &Context) override {
      TranslationUnitDecl *D = Context.getTranslationUnitDecl();

      if (FilterString.empty())
        return print(D);

      TraverseDecl(D);
    }

    bool shouldWalkTypesOfTypeLocs() const { return false; }

    bool TraverseDecl(Decl *D) {
      if (D && filterMatches(D)) {
        bool ShowColors = Out.has_colors();
        if (ShowColors)
          Out.changeColor(raw_ostream::BLUE);
        Out << ((Dump || DumpLookups) ? "Dumping " : "Printing ") << getName(D)
            << ":\n";
        if (ShowColors)
          Out.resetColor();
        print(D);
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
    void print(Decl *D) {
      if (DumpLookups) {
        if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
          if (DC == DC->getPrimaryContext())
            DC->dumpLookups(Out, Dump);
          else
            Out << "Lookup map is in primary DeclContext "
                << DC->getPrimaryContext() << "\n";
        } else
          Out << "Not a DeclContext\n";
      } else if (Dump)
        D->dump(Out);
      else
        D->print(Out, /*Indentation=*/0, /*PrintInstantiation=*/true);
    }

    raw_ostream &Out;
    std::unique_ptr<raw_ostream> OwnedOut;
    bool Dump;
    std::string FilterString;
    bool DumpLookups;
  };

  class ASTDeclNodeLister : public ASTConsumer,
                     public RecursiveASTVisitor<ASTDeclNodeLister> {
  public:
    ASTDeclNodeLister(raw_ostream *Out = nullptr)
        : Out(Out ? *Out : llvm::outs()) {}

    void HandleTranslationUnit(ASTContext &Context) override {
      TraverseDecl(Context.getTranslationUnitDecl());
    }

    bool shouldWalkTypesOfTypeLocs() const { return false; }

    bool VisitNamedDecl(NamedDecl *D) {
      D->printQualifiedName(Out);
      Out << '\n';
      return true;
    }

  private:
    raw_ostream &Out;
  };
} // end anonymous namespace

std::unique_ptr<ASTConsumer>
clang::CreateASTPrinter(std::unique_ptr<raw_ostream> Out,
                        StringRef FilterString) {
  return llvm::make_unique<ASTPrinter>(std::move(Out), /*Dump=*/false,
                                       FilterString);
}

std::unique_ptr<ASTConsumer> clang::CreateASTDumper(StringRef FilterString,
                                                    bool DumpDecls,
                                                    bool DumpLookups) {
  assert((DumpDecls || DumpLookups) && "nothing to dump");
  return llvm::make_unique<ASTPrinter>(nullptr, DumpDecls, FilterString,
                                       DumpLookups);
}

std::unique_ptr<ASTConsumer> clang::CreateASTDeclNodeLister() {
  return llvm::make_unique<ASTDeclNodeLister>(nullptr);
}

//===----------------------------------------------------------------------===//
/// ASTViewer - AST Visualization

namespace {
  class ASTViewer : public ASTConsumer {
    ASTContext *Context;
  public:
    void Initialize(ASTContext &Context) override {
      this->Context = &Context;
    }

    bool HandleTopLevelDecl(DeclGroupRef D) override {
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

std::unique_ptr<ASTConsumer> clang::CreateASTViewer() {
  return llvm::make_unique<ASTViewer>();
}

//===----------------------------------------------------------------------===//
/// DeclContextPrinter - Decl and DeclContext Visualization

namespace {

class DeclContextPrinter : public ASTConsumer {
  raw_ostream& Out;
public:
  DeclContextPrinter() : Out(llvm::errs()) {}

  void HandleTranslationUnit(ASTContext &C) override {
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
    for (auto I : FD->parameters()) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << *I;
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
    for (ParmVarDecl *Parameter : D->parameters()) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << *Parameter;
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
    for (ParmVarDecl *Parameter : D->parameters()) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << *Parameter;
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

  case Decl::ClassTemplateSpecialization: {
    const auto *CTSD = cast<ClassTemplateSpecializationDecl>(DC);
    if (CTSD->isCompleteDefinition())
      Out << "[class template specialization] ";
    else
      Out << "<class template specialization> ";
    Out << *CTSD;
    break;
  }

  case Decl::ClassTemplatePartialSpecialization: {
    const auto *CTPSD = cast<ClassTemplatePartialSpecializationDecl>(DC);
    if (CTPSD->isCompleteDefinition())
      Out << "[class template partial specialization] ";
    else
      Out << "<class template partial specialization> ";
    Out << *CTPSD;
    break;
  }

  default:
    llvm_unreachable("a decl that inherits DeclContext isn't handled");
  }

  Out << "\n";

  // Print decls in the DeclContext.
  for (auto *I : DC->decls()) {
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
    case Decl::ClassTemplateSpecialization:
    case Decl::ClassTemplatePartialSpecialization: {
      DeclContext* DC = cast<DeclContext>(I);
      PrintDeclContext(DC, Indentation+2);
      break;
    }
    case Decl::IndirectField: {
      IndirectFieldDecl* IFD = cast<IndirectFieldDecl>(I);
      Out << "<IndirectField> " << *IFD << '\n';
      break;
    }
    case Decl::Label: {
      LabelDecl *LD = cast<LabelDecl>(I);
      Out << "<Label> " << *LD << '\n';
      break;
    }
    case Decl::Field: {
      FieldDecl *FD = cast<FieldDecl>(I);
      Out << "<field> " << *FD << '\n';
      break;
    }
    case Decl::Typedef:
    case Decl::TypeAlias: {
      TypedefNameDecl* TD = cast<TypedefNameDecl>(I);
      Out << "<typedef> " << *TD << '\n';
      break;
    }
    case Decl::EnumConstant: {
      EnumConstantDecl* ECD = cast<EnumConstantDecl>(I);
      Out << "<enum constant> " << *ECD << '\n';
      break;
    }
    case Decl::Var: {
      VarDecl* VD = cast<VarDecl>(I);
      Out << "<var> " << *VD << '\n';
      break;
    }
    case Decl::ImplicitParam: {
      ImplicitParamDecl* IPD = cast<ImplicitParamDecl>(I);
      Out << "<implicit parameter> " << *IPD << '\n';
      break;
    }
    case Decl::ParmVar: {
      ParmVarDecl* PVD = cast<ParmVarDecl>(I);
      Out << "<parameter> " << *PVD << '\n';
      break;
    }
    case Decl::ObjCProperty: {
      ObjCPropertyDecl* OPD = cast<ObjCPropertyDecl>(I);
      Out << "<objc property> " << *OPD << '\n';
      break;
    }
    case Decl::FunctionTemplate: {
      FunctionTemplateDecl* FTD = cast<FunctionTemplateDecl>(I);
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
      NamespaceAliasDecl* NAD = cast<NamespaceAliasDecl>(I);
      Out << "<namespace alias> " << *NAD << '\n';
      break;
    }
    case Decl::ClassTemplate: {
      ClassTemplateDecl *CTD = cast<ClassTemplateDecl>(I);
      Out << "<class template> " << *CTD << '\n';
      break;
    }
    case Decl::OMPThreadPrivate: {
      Out << "<omp threadprivate> " << '"' << I << "\"\n";
      break;
    }
    case Decl::Friend: {
      Out << "<friend>";
      if (const NamedDecl *ND = cast<FriendDecl>(I)->getFriendDecl())
        Out << ' ' << *ND;
      Out << "\n";
      break;
    }
    case Decl::Using: {
      Out << "<using> " << *cast<UsingDecl>(I) << "\n";
      break;
    }
    case Decl::UsingShadow: {
      Out << "<using shadow> " << *cast<UsingShadowDecl>(I) << "\n";
      break;
    }
    case Decl::Empty: {
      Out << "<empty>\n";
      break;
    }
    case Decl::AccessSpec: {
      Out << "<access specifier>\n";
      break;
    }
    case Decl::VarTemplate: {
      Out << "<var template> " << *cast<VarTemplateDecl>(I) << "\n";
      break;
    }
    case Decl::StaticAssert: {
      Out << "<static assert>\n";
      break;
    }
    default:
      Out << "DeclKind: " << DK << '"' << I << "\"\n";
      llvm_unreachable("decl unhandled");
    }
  }
}
std::unique_ptr<ASTConsumer> clang::CreateDeclContextPrinter() {
  return llvm::make_unique<DeclContextPrinter>();
}
