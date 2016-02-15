//===--- ProTypeMemberInitCheck.cpp - clang-tidy---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProTypeMemberInitCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang::ast_matchers;
using llvm::SmallPtrSet;
using llvm::SmallPtrSetImpl;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

namespace {

AST_MATCHER(CXXConstructorDecl, isUserProvided) {
  return Node.isUserProvided();
}

static void
fieldsRequiringInit(const RecordDecl::field_range &Fields,
                    SmallPtrSetImpl<const FieldDecl *> &FieldsToInit) {
  for (const FieldDecl *F : Fields) {
    QualType Type = F->getType();
    if (Type->isPointerType() || Type->isBuiltinType())
      FieldsToInit.insert(F);
  }
}

void removeFieldsInitializedInBody(
    const Stmt &Stmt, ASTContext &Context,
    SmallPtrSetImpl<const FieldDecl *> &FieldDecls) {
  auto Matches =
      match(findAll(binaryOperator(
                hasOperatorName("="),
                hasLHS(memberExpr(member(fieldDecl().bind("fieldDecl")))))),
            Stmt, Context);
  for (const auto &Match : Matches)
    FieldDecls.erase(Match.getNodeAs<FieldDecl>("fieldDecl"));
}

// Creates comma separated list of fields requiring initialization in order of
// declaration.
std::string toCommaSeparatedString(
    const RecordDecl::field_range &FieldRange,
    const SmallPtrSetImpl<const FieldDecl *> &FieldsRequiringInit) {
  std::string List;
  llvm::raw_string_ostream Stream(List);
  size_t AddedFields = 0;
  for (const FieldDecl *Field : FieldRange) {
    if (FieldsRequiringInit.count(Field) > 0) {
      Stream << Field->getName();
      if (++AddedFields < FieldsRequiringInit.size())
        Stream << ", ";
    }
  }
  return Stream.str();
}

// Contains all fields in correct order that need to be inserted at the same
// location for pre C++11.
// There are 3 kinds of insertions:
// 1. The fields are inserted after an existing CXXCtorInitializer stored in
// InitializerBefore. This will be the case whenever there is a written
// initializer before the fields available.
// 2. The fields are inserted before the first existing initializer stored in
// InitializerAfter.
// 3. There are no written initializers and the fields will be inserted before
// the constructor's body creating a new initializer list including the ':'.
struct FieldsInsertion {
  const CXXCtorInitializer *InitializerBefore;
  const CXXCtorInitializer *InitializerAfter;
  SmallVector<const FieldDecl *, 4> Fields;

  SourceLocation getLocation(const ASTContext &Context,
                             const CXXConstructorDecl &Constructor) const {
    if (InitializerBefore != nullptr) {
      return Lexer::getLocForEndOfToken(InitializerBefore->getRParenLoc(), 0,
                                        Context.getSourceManager(),
                                        Context.getLangOpts());
    }
    auto StartLocation = InitializerAfter != nullptr
                             ? InitializerAfter->getSourceRange().getBegin()
                             : Constructor.getBody()->getLocStart();
    auto Token =
        lexer_utils::getPreviousNonCommentToken(Context, StartLocation);
    return Lexer::getLocForEndOfToken(Token.getLocation(), 0,
                                      Context.getSourceManager(),
                                      Context.getLangOpts());
  }

  std::string codeToInsert() const {
    assert(!Fields.empty() && "No fields to insert");
    std::string Code;
    llvm::raw_string_ostream Stream(Code);
    // Code will be inserted before the first written initializer after ':',
    // append commas.
    if (InitializerAfter != nullptr) {
      for (const auto *Field : Fields)
        Stream << " " << Field->getName() << "(),";
    } else {
      // The full initializer list is created, add extra space after
      // constructor's rparens.
      if (InitializerBefore == nullptr)
        Stream << " ";
      for (const auto *Field : Fields)
        Stream << ", " << Field->getName() << "()";
    }
    Stream.flush();
    // The initializer list is created, replace leading comma with colon.
    if (InitializerBefore == nullptr && InitializerAfter == nullptr)
      Code[1] = ':';
    return Code;
  }
};

SmallVector<FieldsInsertion, 16> computeInsertions(
    const CXXConstructorDecl::init_const_range &Inits,
    const RecordDecl::field_range &Fields,
    const SmallPtrSetImpl<const FieldDecl *> &FieldsRequiringInit) {
  // Find last written non-member initializer or null.
  const CXXCtorInitializer *LastWrittenNonMemberInit = nullptr;
  for (const CXXCtorInitializer *Init : Inits) {
    if (Init->isWritten() && !Init->isMemberInitializer())
      LastWrittenNonMemberInit = Init;
  }
  SmallVector<FieldsInsertion, 16> OrderedFields;
  OrderedFields.push_back({LastWrittenNonMemberInit, nullptr, {}});

  auto CurrentField = Fields.begin();
  for (const CXXCtorInitializer *Init : Inits) {
    if (Init->isWritten() && Init->isMemberInitializer()) {
      const FieldDecl *MemberField = Init->getMember();
      // Add all fields between current field and this member field the previous
      // FieldsInsertion if the field requires initialization.
      for (; CurrentField != Fields.end() && *CurrentField != MemberField;
           ++CurrentField) {
        if (FieldsRequiringInit.count(*CurrentField) > 0)
          OrderedFields.back().Fields.push_back(*CurrentField);
      }
      // If this is the first written member initializer and there was no
      // written non-member initializer set this initializer as
      // InitializerAfter.
      if (OrderedFields.size() == 1 &&
          OrderedFields.back().InitializerBefore == nullptr)
        OrderedFields.back().InitializerAfter = Init;
      OrderedFields.push_back({Init, nullptr, {}});
    }
  }
  // Add remaining fields that require initialization to last FieldsInsertion.
  for (; CurrentField != Fields.end(); ++CurrentField) {
    if (FieldsRequiringInit.count(*CurrentField) > 0)
      OrderedFields.back().Fields.push_back(*CurrentField);
  }
  return OrderedFields;
}

} // namespace

void ProTypeMemberInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstructorDecl(isDefinition(), isUserProvided(),
                                        unless(isInstantiated()))
                         .bind("ctor"),
                     this);
}

void ProTypeMemberInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  const auto &MemberFields = Ctor->getParent()->fields();

  SmallPtrSet<const FieldDecl *, 16> FieldsToInit;
  fieldsRequiringInit(MemberFields, FieldsToInit);
  if (FieldsToInit.empty())
    return;

  for (CXXCtorInitializer *Init : Ctor->inits()) {
    // Return early if this constructor simply delegates to another constructor
    // in the same class.
    if (Init->isDelegatingInitializer())
      return;
    if (!Init->isMemberInitializer())
      continue;
    FieldsToInit.erase(Init->getMember());
  }
  removeFieldsInitializedInBody(*Ctor->getBody(), *Result.Context,
                                FieldsToInit);
  if (FieldsToInit.empty())
    return;

  DiagnosticBuilder Diag =
      diag(Ctor->getLocStart(),
           "constructor does not initialize these built-in/pointer fields: %0")
      << toCommaSeparatedString(MemberFields, FieldsToInit);
  // Do not propose fixes in macros since we cannot place them correctly.
  if (Ctor->getLocStart().isMacroID())
    return;
  // For C+11 use in-class initialization which covers all future constructors
  // as well.
  if (Result.Context->getLangOpts().CPlusPlus11) {
    for (const auto *Field : FieldsToInit) {
      Diag << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(Field->getSourceRange().getEnd(), 0,
                                     Result.Context->getSourceManager(),
                                     Result.Context->getLangOpts()),
          "{}");
    }
    return;
  }
  for (const auto &FieldsInsertion :
       computeInsertions(Ctor->inits(), MemberFields, FieldsToInit)) {
    if (!FieldsInsertion.Fields.empty())
      Diag << FixItHint::CreateInsertion(
          FieldsInsertion.getLocation(*Result.Context, *Ctor),
          FieldsInsertion.codeToInsert());
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
