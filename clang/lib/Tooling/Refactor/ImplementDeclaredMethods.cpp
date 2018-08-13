//===--- ImplementDeclaredMethods.cpp -  ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the "Generate missing method definitions" refactoring
// operation.
//
//===----------------------------------------------------------------------===//

#include "RefactoringContinuations.h"
#include "RefactoringOperations.h"
#include "SourceLocationUtilities.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

using namespace clang;
using namespace clang::tooling;

namespace {

template <typename ClassType, typename MethodType, typename Derived>
class ImplementDeclaredMethodsOperation : public RefactoringOperation {
public:
  ImplementDeclaredMethodsOperation(
      const ClassType *Container, ArrayRef<const MethodType *> SelectedMethods)
      : Container(Container),
        SelectedMethods(SelectedMethods.begin(), SelectedMethods.end()) {}

  const Decl *getTransformedDecl() const override {
    return SelectedMethods.front();
  }

  const Decl *getLastTransformedDecl() const override {
    return SelectedMethods.back();
  }

  static RefactoringOperationResult
  initiate(const ClassType *Container, ArrayRef<const MethodType *> Methods,
           bool CreateOperation) {
    if (Methods.empty())
      return None;

    RefactoringOperationResult Result;
    Result.Initiated = true;
    if (!CreateOperation)
      return Result;
    auto Operation = llvm::make_unique<Derived>(Container, Methods);
    Result.RefactoringOp = std::move(Operation);
    return Result;
  }

  const ClassType *Container;
  llvm::SmallVector<const MethodType *, 8> SelectedMethods;
};

class ImplementDeclaredCXXMethodsOperation
    : public ImplementDeclaredMethodsOperation<
          CXXRecordDecl, CXXMethodDecl, ImplementDeclaredCXXMethodsOperation> {
public:
  ImplementDeclaredCXXMethodsOperation(
      const CXXRecordDecl *Container,
      ArrayRef<const CXXMethodDecl *> SelectedMethods)
      : ImplementDeclaredMethodsOperation(Container, SelectedMethods) {}

  llvm::Expected<RefactoringResult>
  perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex) override;

  static void addInlineBody(const CXXMethodDecl *MD, const ASTContext &Context,
                            std::vector<RefactoringReplacement> &Replacements);

  static llvm::Expected<RefactoringResult> runInImplementationAST(
      ASTContext &Context, const FileID &File, const CXXRecordDecl *Class,
      ArrayRef<indexer::Indexed<const CXXMethodDecl *>> SelectedMethods);
};

class ImplementDeclaredObjCMethodsOperation
    : public ImplementDeclaredMethodsOperation<
          ObjCContainerDecl, ObjCMethodDecl,
          ImplementDeclaredObjCMethodsOperation> {
  const ObjCInterfaceDecl *Interface;

public:
  ImplementDeclaredObjCMethodsOperation(
      const ObjCContainerDecl *Container,
      ArrayRef<const ObjCMethodDecl *> SelectedMethods)
      : ImplementDeclaredMethodsOperation(Container, SelectedMethods) {
    if (const auto *CD = dyn_cast<ObjCCategoryDecl>(Container))
      Interface = CD->getClassInterface();
    else
      Interface = nullptr;
  }

  llvm::Expected<RefactoringResult>
  perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex) override;

  static llvm::Expected<RefactoringResult> runInImplementationAST(
      ASTContext &Context, const FileID &File,
      const ObjCContainerDecl *Container, const ObjCInterfaceDecl *Interface,
      ArrayRef<std::string> MethodDeclarations,
      ArrayRef<indexer::Indexed<const ObjCMethodDecl *>> SelectedMethods);
};

/// Returns true if the given Objective-C method has an implementation.
bool isImplemented(const ObjCMethodDecl *M) {
  if (M->hasBody() || M->isDefined())
    return true;
  return false;
}

} // end anonymous namespace

RefactoringOperationResult
clang::tooling::initiateImplementDeclaredMethodsOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  // Find the selected Class.
  auto SelectedDecl = Slice.innermostSelectedDecl([](const Decl *D) {
    return isa<CXXRecordDecl>(D) || isa<ObjCInterfaceDecl>(D) ||
           isa<ObjCCategoryDecl>(D);
  });
  if (!SelectedDecl)
    return None;
  // Look at the set of methods that intersect with the selection.
  if (const auto *CXXClass = dyn_cast<CXXRecordDecl>(SelectedDecl->getDecl())) {
    if (CXXClass->isDependentType())
      return RefactoringOperationResult("templates are unsupported");
    llvm::SmallVector<const CXXMethodDecl *, 8> SelectedMethods;
    for (const CXXMethodDecl *M : CXXClass->methods()) {
      if (M->isImplicit() || M->hasBody() || M->isPure() || M->isDefaulted() ||
          M->isDeletedAsWritten() || M->getDescribedFunctionTemplate())
        continue;
      if (Slice.isSourceRangeSelected(
              CharSourceRange::getTokenRange(M->getSourceRange())))
        SelectedMethods.push_back(M);
    }
    return ImplementDeclaredCXXMethodsOperation::initiate(
        CXXClass, SelectedMethods, CreateOperation);
  }
  const ObjCContainerDecl *Container =
      cast<ObjCContainerDecl>(SelectedDecl->getDecl());
  llvm::SmallVector<const ObjCMethodDecl *, 8> SelectedMethods;
  for (const ObjCMethodDecl *M : Container->methods()) {
    if (M->isImplicit() || isImplemented(M))
      continue;
    if (Slice.isSourceRangeSelected(
            CharSourceRange::getTokenRange(M->getSourceRange())))
      SelectedMethods.push_back(M);
  }
  // Method declarations from class extensions should be defined in class
  // @implementations.
  if (const auto *Category = dyn_cast<ObjCCategoryDecl>(Container)) {
    if (Category->IsClassExtension())
      Container = Category->getClassInterface();
  }
  return ImplementDeclaredObjCMethodsOperation::initiate(
      Container, SelectedMethods, CreateOperation);
}

llvm::Expected<RefactoringResult>
ImplementDeclaredCXXMethodsOperation::perform(
    ASTContext &Context, const Preprocessor &ThePreprocessor,
    const RefactoringOptionSet &Options, unsigned SelectedCandidateIndex) {
  if (Container->isLexicallyWithinFunctionOrMethod()) {
    // Local methods can be implemented inline.
    std::vector<RefactoringReplacement> Replacements;
    for (const CXXMethodDecl *MD : SelectedMethods)
      addInlineBody(MD, Context, Replacements);
    return std::move(Replacements);
  }
  using namespace indexer;
  return continueInExternalASTUnit(
      fileThatShouldContainImplementationOf(Container), runInImplementationAST,
      Container, filter(llvm::makeArrayRef(SelectedMethods),
                        [](const DeclEntity &D) { return !D.isDefined(); }));
}

void ImplementDeclaredCXXMethodsOperation::addInlineBody(
    const CXXMethodDecl *MD, const ASTContext &Context,
    std::vector<RefactoringReplacement> &Replacements) {
  SourceLocation EndLoc = MD->getEndLoc();
  SourceRange SemiRange = getRangeOfNextToken(
      EndLoc, tok::semi, Context.getSourceManager(), Context.getLangOpts());
  if (SemiRange.isValid()) {
    Replacements.push_back(RefactoringReplacement(SemiRange));
    EndLoc = SemiRange.getEnd();
  }
  SourceLocation InsertionLoc = getLastLineLocationUnlessItHasOtherTokens(
      EndLoc, Context.getSourceManager(), Context.getLangOpts());
  Replacements.push_back(
      RefactoringReplacement(SourceRange(InsertionLoc, InsertionLoc),
                             StringRef(" { \n  <#code#>;\n}")));
}

static const RecordDecl *findOutermostRecord(const RecordDecl *RD) {
  const RecordDecl *Result = RD;
  for (const DeclContext *DC = Result->getLexicalDeclContext();
       isa<RecordDecl>(DC); DC = Result->getLexicalDeclContext())
    Result = cast<RecordDecl>(DC);
  return Result;
}

static bool containsUsingOf(const NamespaceDecl *ND,
                            const ASTContext &Context) {
  for (const Decl *D : Context.getTranslationUnitDecl()->decls()) {
    if (const auto *UDD = dyn_cast<UsingDirectiveDecl>(D)) {
      if (UDD->getNominatedNamespace() == ND)
        return true;
    }
  }
  return false;
}

llvm::Expected<RefactoringResult>
ImplementDeclaredCXXMethodsOperation::runInImplementationAST(
    ASTContext &Context, const FileID &File, const CXXRecordDecl *Class,
    ArrayRef<indexer::Indexed<const CXXMethodDecl *>> SelectedMethods) {
  if (!Class)
    return llvm::make_error<RefactoringOperationError>(
        "the target class is not defined in the continuation AST unit");

  SourceManager &SM = Context.getSourceManager();

  // Find the defined methods of the class.
  llvm::SmallVector<const CXXMethodDecl *, 8> DefinedOutOfLineMethods;
  for (const CXXMethodDecl *M : Class->methods()) {
    if (M->isImplicit())
      continue;
    if (const FunctionDecl *MD = M->getDefinition()) {
      if (!MD->isOutOfLine())
        continue;
      SourceLocation Loc = SM.getExpansionLoc(MD->getBeginLoc());
      if (SM.getFileID(Loc) == File)
        DefinedOutOfLineMethods.push_back(cast<CXXMethodDecl>(MD));
    }
  }

  std::vector<RefactoringReplacement> Replacements;
  std::string MethodString;
  llvm::raw_string_ostream OS(MethodString);

  // Pick a good insertion location.
  SourceLocation InsertionLoc;
  const CXXMethodDecl *InsertAfterMethod = nullptr;
  NestedNameSpecifier *NamePrefix = nullptr;
  if (DefinedOutOfLineMethods.empty()) {
    const RecordDecl *OutermostRecord = findOutermostRecord(Class);
    InsertionLoc = SM.getExpansionRange(OutermostRecord->getEndLoc()).getEnd();
    if (SM.getFileID(InsertionLoc) == File) {
      // We can insert right after the class. Compute the appropriate
      // qualification.
      NamePrefix = NestedNameSpecifier::getRequiredQualification(
          Context, OutermostRecord->getLexicalDeclContext(),
          Class->getLexicalDeclContext());
    } else {
      // We can't insert after the end of the class, since the indexer told us
      // that some file should have the implementation of it, even when there
      // are no methods here. We should try to insert at the end of the file.
      InsertionLoc = SM.getLocForEndOfFile(File);
      NamePrefix = NestedNameSpecifier::getRequiredQualification(
          Context, Context.getTranslationUnitDecl(),
          Class->getLexicalDeclContext());
      llvm::SmallVector<const NamespaceDecl *, 4> Namespaces;
      for (const NestedNameSpecifier *Qualifier = NamePrefix; Qualifier;
           Qualifier = Qualifier->getPrefix()) {
        if (const NamespaceDecl *ND = Qualifier->getAsNamespace())
          Namespaces.push_back(ND);
      }
      // When the class is in a namespace, add a 'using' declaration if it's
      // needed and adjust the out-of-line qualification.
      if (!Namespaces.empty()) {
        const NamespaceDecl *InnermostNamespace = Namespaces[0];
        if (!containsUsingOf(InnermostNamespace, Context)) {
          std::string NamespaceString;
          llvm::raw_string_ostream NamespaceOS(NamespaceString);
          for (const NamespaceDecl *ND : llvm::reverse(Namespaces)) {
            if (!NamespaceOS.str().empty())
              NamespaceOS << "::";
            NamespaceOS << ND->getDeclName();
          }
          OS << "\nusing namespace " << NamespaceOS.str() << ";";
        }
        // Re-compute the name qualifier without the namespace.
        NamePrefix = NestedNameSpecifier::getRequiredQualification(
            Context, InnermostNamespace, Class->getLexicalDeclContext());
      }
    }
  } else {
    // Insert at the end of the defined methods.
    for (const CXXMethodDecl *M : DefinedOutOfLineMethods) {
      SourceLocation EndLoc = SM.getExpansionRange(M->getEndLoc()).getEnd();
      if (InsertionLoc.isInvalid() ||
          SM.isBeforeInTranslationUnit(InsertionLoc, EndLoc)) {
        InsertionLoc = EndLoc;
        InsertAfterMethod = M;
      }
    }
  }
  InsertionLoc = getLastLineLocationUnlessItHasOtherTokens(
      InsertionLoc, SM, Context.getLangOpts());

  PrintingPolicy PP = Context.getPrintingPolicy();
  PP.PolishForDeclaration = true;
  PP.SupressStorageClassSpecifiers = true;
  PP.SuppressStrongLifetime = true;
  PP.SuppressLifetimeQualifiers = true;
  PP.SuppressUnwrittenScope = true;
  OS << "\n";
  for (const auto &I : SelectedMethods) {
    const CXXMethodDecl *MD = I.Decl;
    // Check if the method is already defined.
    if (!MD)
      continue;

    // Drop the 'virtual' specifier.
    bool IsVirtual = MD->isVirtualAsWritten();
    const_cast<CXXMethodDecl *>(MD)->setVirtualAsWritten(false);

    // Drop the default arguments.
    llvm::SmallVector<std::pair<ParmVarDecl *, Expr *>, 4> DefaultArgs;
    for (const ParmVarDecl *P : MD->parameters()) {
      if (!P->hasDefaultArg())
        continue;
      Expr *E = const_cast<ParmVarDecl *>(P)->getDefaultArg();
      const_cast<ParmVarDecl *>(P)->setDefaultArg(nullptr);
      DefaultArgs.emplace_back(const_cast<ParmVarDecl *>(P), E);
    }

    // Add the nested name specifiers that are appropriate for an out-of-line
    // method.
    auto *Qualifier =
        InsertAfterMethod
            ? InsertAfterMethod->getQualifier()
            : NestedNameSpecifier::Create(
                  Context, /*Prefix=*/NamePrefix, /*Template=*/false,
                  Context.getRecordType(Class).getTypePtr());
    NestedNameSpecifierLoc PrevQualifierInfo = MD->getQualifierLoc();
    const_cast<CXXMethodDecl *>(MD)->setQualifierInfo(
        NestedNameSpecifierLoc(Qualifier, /*Loc=*/nullptr));

    OS << "\n";
    MD->print(OS, PP);
    OS << " { \n  <#code#>;\n}\n";

    // Restore the original method
    for (const auto &DefaultArg : DefaultArgs)
      DefaultArg.first->setDefaultArg(DefaultArg.second);
    const_cast<CXXMethodDecl *>(MD)->setVirtualAsWritten(IsVirtual);
    const_cast<CXXMethodDecl *>(MD)->setQualifierInfo(PrevQualifierInfo);
  }

  Replacements.push_back(RefactoringReplacement(
      SourceRange(InsertionLoc, InsertionLoc), std::move(OS.str())));

  return std::move(Replacements);
}

llvm::Expected<RefactoringResult>
ImplementDeclaredObjCMethodsOperation::perform(
    ASTContext &Context, const Preprocessor &ThePreprocessor,
    const RefactoringOptionSet &Options, unsigned SelectedCandidateIndex) {
  using namespace indexer;

  // Print the methods before running the continuation because the continuation
  // TU might not have these method declarations (e.g. category implemented in
  // the class implementation).
  PrintingPolicy PP = Context.getPrintingPolicy();
  PP.PolishForDeclaration = true;
  PP.SuppressStrongLifetime = true;
  PP.SuppressLifetimeQualifiers = true;
  PP.SuppressUnwrittenScope = true;
  std::vector<std::string> MethodDeclarations;
  for (const ObjCMethodDecl *MD : SelectedMethods) {
    std::string MethodDeclStr;
    llvm::raw_string_ostream MethodOS(MethodDeclStr);
    MD->print(MethodOS, PP);
    MethodDeclarations.push_back(std::move(MethodOS.str()));
  }

  return continueInExternalASTUnit(
      fileThatShouldContainImplementationOf(Container), runInImplementationAST,
      Container, Interface, MethodDeclarations,
      filter(llvm::makeArrayRef(SelectedMethods),
             [](const DeclEntity &D) { return !D.isDefined(); }));
}

static const ObjCImplDecl *
getImplementationContainer(const ObjCContainerDecl *Container,
                           const ObjCInterfaceDecl *Interface = nullptr) {
  if (!Container)
    return Interface ? getImplementationContainer(Interface) : nullptr;
  if (const auto *ID = dyn_cast<ObjCInterfaceDecl>(Container))
    return ID->getImplementation();
  if (const auto *CD = dyn_cast<ObjCCategoryDecl>(Container)) {
    if (const auto *Impl = CD->getImplementation())
      return Impl;
    return getImplementationContainer(Interface);
  }
  return nullptr;
}

llvm::Expected<RefactoringResult>
ImplementDeclaredObjCMethodsOperation::runInImplementationAST(
    ASTContext &Context, const FileID &File, const ObjCContainerDecl *Container,
    const ObjCInterfaceDecl *Interface,
    ArrayRef<std::string> MethodDeclarations,
    ArrayRef<indexer::Indexed<const ObjCMethodDecl *>> SelectedMethods) {
  const ObjCImplDecl *ImplementationContainer =
      getImplementationContainer(Container, Interface);
  if (!ImplementationContainer)
    return llvm::make_error<RefactoringOperationError>(
        "the target @interface is not implemented in the continuation AST "
        "unit");

  std::vector<RefactoringReplacement> Replacements;

  std::string MethodString;
  llvm::raw_string_ostream OS(MethodString);

  assert(MethodDeclarations.size() >= SelectedMethods.size() &&
         "fewer declarations than selected methods?");
  for (const auto &I : llvm::enumerate(SelectedMethods)) {
    indexer::Indexed<const ObjCMethodDecl *> Decl = I.value();
    // Skip methods that are already defined.
    if (!Decl.isNotDefined())
      continue;

    OS << StringRef(MethodDeclarations[I.index()]).drop_back(); // Drop the ';'
    OS << " { \n  <#code#>;\n}\n\n";
  }
  SourceLocation InsertionLoc = ImplementationContainer->getEndLoc();

  Replacements.push_back(RefactoringReplacement(
      SourceRange(InsertionLoc, InsertionLoc), std::move(OS.str())));

  return std::move(Replacements);
}
