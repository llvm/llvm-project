//===-- tools/extra/clang-reorder-fields/ReorderFieldsAction.cpp -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the
/// ReorderFieldsAction::newASTConsumer method
///
//===----------------------------------------------------------------------===//

#include "ReorderFieldsAction.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Refactoring.h"
#include <algorithm>
#include <string>

namespace clang {
namespace reorder_fields {
using namespace clang::ast_matchers;

/// \brief Finds the definition of a record by name.
///
/// \returns nullptr if the name is ambiguous or not found.
static const CXXRecordDecl *findDefinition(StringRef RecordName,
                                           ASTContext &Context) {
  auto Results = match(
      recordDecl(hasName(RecordName), isDefinition()).bind("cxxRecordDecl"),
      Context);
  if (Results.empty()) {
    llvm::errs() << "Definition of " << RecordName << "  not found\n";
    return nullptr;
  }
  if (Results.size() > 1) {
    llvm::errs() << "The name " << RecordName
                 << " is ambiguous, several definitions found\n";
    return nullptr;
  }
  return selectFirst<CXXRecordDecl>("cxxRecordDecl", Results);
}

/// \brief Calculates the new order of fields.
///
/// \returns empty vector if the list of fields doesn't match the definition.
static SmallVector<unsigned, 4>
getNewFieldsOrder(const CXXRecordDecl *Definition,
                  ArrayRef<std::string> DesiredFieldsOrder) {
  assert(Definition && "Definition is null");

  llvm::StringMap<unsigned> NameToIndex;
  for (const auto *Field : Definition->fields())
    NameToIndex[Field->getName()] = Field->getFieldIndex();

  if (DesiredFieldsOrder.size() != NameToIndex.size()) {
    llvm::errs() << "Number of provided fields doesn't match definition.\n";
    return {};
  }
  SmallVector<unsigned, 4> NewFieldsOrder;
  for (const auto &Name : DesiredFieldsOrder) {
    if (!NameToIndex.count(Name)) {
      llvm::errs() << "Field " << Name << " not found in definition.\n";
      return {};
    }
    NewFieldsOrder.push_back(NameToIndex[Name]);
  }
  assert(NewFieldsOrder.size() == NameToIndex.size());
  return NewFieldsOrder;
}

// FIXME: error-handling
/// \brief Replaces one range of source code by another.
static void
addReplacement(SourceRange Old, SourceRange New, const ASTContext &Context,
               std::map<std::string, tooling::Replacements> &Replacements) {
  StringRef NewText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(New),
                           Context.getSourceManager(), Context.getLangOpts());
  tooling::Replacement R(Context.getSourceManager(),
                         CharSourceRange::getTokenRange(Old), NewText,
                         Context.getLangOpts());
  consumeError(Replacements[R.getFilePath()].add(R));
}

/// \brief Reorders fields in the definition of a struct/class.
///
/// At the moment reodering of fields with
/// different accesses (public/protected/private) is not supported.
/// \returns true on success.
static bool reorderFieldsInDefinition(
    const CXXRecordDecl *Definition, ArrayRef<unsigned> NewFieldsOrder,
    const ASTContext &Context,
    std::map<std::string, tooling::Replacements> &Replacements) {
  assert(Definition && "Definition is null");

  SmallVector<const FieldDecl *, 10> Fields;
  for (const auto *Field : Definition->fields())
    Fields.push_back(Field);

  // Check that the permutation of the fields doesn't change the accesses
  for (const auto *Field : Definition->fields()) {
    const auto FieldIndex = Field->getFieldIndex();
    if (Field->getAccess() != Fields[NewFieldsOrder[FieldIndex]]->getAccess()) {
      llvm::errs() << "Currently reodering of fields with different accesses "
                      "is not supported\n";
      return false;
    }
  }

  for (const auto *Field : Definition->fields()) {
    const auto FieldIndex = Field->getFieldIndex();
    if (FieldIndex == NewFieldsOrder[FieldIndex])
      continue;
    addReplacement(Field->getSourceRange(),
                   Fields[NewFieldsOrder[FieldIndex]]->getSourceRange(),
                   Context, Replacements);
  }
  return true;
}

/// \brief Reorders initializers in a C++ struct/class constructor.
///
/// A constructor can have initializers for an arbitrary subset of the class's fields.
/// Thus, we need to ensure that we reorder just the initializers that are present.
static void reorderFieldsInConstructor(
    const CXXConstructorDecl *CtorDecl, ArrayRef<unsigned> NewFieldsOrder,
    const ASTContext &Context,
    std::map<std::string, tooling::Replacements> &Replacements) {
  assert(CtorDecl && "Constructor declaration is null");
  if (CtorDecl->isImplicit() || CtorDecl->getNumCtorInitializers() <= 1)
    return;

  // The method FunctionDecl::isThisDeclarationADefinition returns false
  // for a defaulted function unless that function has been implicitly defined.
  // Thus this assert needs to be after the previous checks.
  assert(CtorDecl->isThisDeclarationADefinition() && "Not a definition");

  SmallVector<unsigned, 10> NewFieldsPositions(NewFieldsOrder.size());
  for (unsigned i = 0, e = NewFieldsOrder.size(); i < e; ++i)
    NewFieldsPositions[NewFieldsOrder[i]] = i;

  SmallVector<const CXXCtorInitializer *, 10> OldWrittenInitializersOrder;
  SmallVector<const CXXCtorInitializer *, 10> NewWrittenInitializersOrder;
  for (const auto *Initializer : CtorDecl->inits()) {
    if (!Initializer->isWritten())
      continue;
    OldWrittenInitializersOrder.push_back(Initializer);
    NewWrittenInitializersOrder.push_back(Initializer);
  }
  auto ByFieldNewPosition = [&](const CXXCtorInitializer *LHS,
                                const CXXCtorInitializer *RHS) {
    assert(LHS && RHS);
    return NewFieldsPositions[LHS->getMember()->getFieldIndex()] <
           NewFieldsPositions[RHS->getMember()->getFieldIndex()];
  };
  std::sort(std::begin(NewWrittenInitializersOrder),
            std::end(NewWrittenInitializersOrder), ByFieldNewPosition);
  assert(OldWrittenInitializersOrder.size() ==
         NewWrittenInitializersOrder.size());
  for (unsigned i = 0, e = NewWrittenInitializersOrder.size(); i < e; ++i)
    if (OldWrittenInitializersOrder[i] != NewWrittenInitializersOrder[i])
      addReplacement(OldWrittenInitializersOrder[i]->getSourceRange(),
                     NewWrittenInitializersOrder[i]->getSourceRange(), Context,
                     Replacements);
}

/// \brief Reorders initializers in the brace initialization of an aggregate.
///
/// At the moment partial initialization is not supported.
/// \returns true on success
static bool reorderFieldsInInitListExpr(
    const InitListExpr *InitListEx, ArrayRef<unsigned> NewFieldsOrder,
    const ASTContext &Context,
    std::map<std::string, tooling::Replacements> &Replacements) {
  assert(InitListEx && "Init list expression is null");
  // We care only about InitListExprs which originate from source code. 
  // Implicit InitListExprs are created by the semantic analyzer.
  if (!InitListEx->isExplicit())
    return true;
  // The method InitListExpr::getSyntacticForm may return nullptr indicating that
  // the current initializer list also serves as its syntactic form.
  if (const auto *SyntacticForm = InitListEx->getSyntacticForm())
    InitListEx = SyntacticForm;
  // If there are no initializers we do not need to change anything.
  if (!InitListEx->getNumInits())
    return true;
  if (InitListEx->getNumInits() != NewFieldsOrder.size()) {
    llvm::errs() << "Currently only full initialization is supported\n";
    return false;
  }
  for (unsigned i = 0, e = InitListEx->getNumInits(); i < e; ++i)
    if (i != NewFieldsOrder[i])
      addReplacement(
          InitListEx->getInit(i)->getSourceRange(),
          InitListEx->getInit(NewFieldsOrder[i])->getSourceRange(), Context,
          Replacements);
  return true;
}

namespace {
class ReorderingConsumer : public ASTConsumer {
  StringRef RecordName;
  ArrayRef<std::string> DesiredFieldsOrder;
  std::map<std::string, tooling::Replacements> &Replacements;

public:
  ReorderingConsumer(StringRef RecordName,
                     ArrayRef<std::string> DesiredFieldsOrder,
                     std::map<std::string, tooling::Replacements> &Replacements)
      : RecordName(RecordName), DesiredFieldsOrder(DesiredFieldsOrder),
        Replacements(Replacements) {}

  ReorderingConsumer(const ReorderingConsumer &) = delete;
  ReorderingConsumer &operator=(const ReorderingConsumer &) = delete;

  void HandleTranslationUnit(ASTContext &Context) override {
    const CXXRecordDecl *RD = findDefinition(RecordName, Context);
    if (!RD)
      return;
    SmallVector<unsigned, 4> NewFieldsOrder =
        getNewFieldsOrder(RD, DesiredFieldsOrder);
    if (NewFieldsOrder.empty())
      return;
    if (!reorderFieldsInDefinition(RD, NewFieldsOrder, Context, Replacements))
      return;
    for (const auto *C : RD->ctors())
      if (const auto *D = dyn_cast<CXXConstructorDecl>(C->getDefinition()))
        reorderFieldsInConstructor(cast<const CXXConstructorDecl>(D),
                                   NewFieldsOrder, Context, Replacements);

    // We only need to reorder init list expressions for aggregate types.
    // For other types the order of constructor parameters is used,
    // which we don't change at the moment.
    // Now (v0) partial initialization is not supported.
    if (RD->isAggregate())
      for (auto Result :
           match(initListExpr(hasType(equalsNode(RD))).bind("initListExpr"),
                 Context))
        if (!reorderFieldsInInitListExpr(
                Result.getNodeAs<InitListExpr>("initListExpr"), NewFieldsOrder,
                Context, Replacements)) {
          Replacements.clear();
          return;
        }
  }
};
} // end anonymous namespace

std::unique_ptr<ASTConsumer> ReorderFieldsAction::newASTConsumer() {
  return llvm::make_unique<ReorderingConsumer>(RecordName, DesiredFieldsOrder,
                                               Replacements);
}

} // namespace reorder_fields
} // namespace clang
