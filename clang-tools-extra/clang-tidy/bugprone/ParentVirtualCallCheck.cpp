//===--- ParentVirtualCallCheck.cpp - clang-tidy---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ParentVirtualCallCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cctype>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

using BasesVector = llvm::SmallVector<const CXXRecordDecl *, 5>;

static bool isParentOf(const CXXRecordDecl &Parent,
                       const CXXRecordDecl &ThisClass) {
  if (Parent.getCanonicalDecl() == ThisClass.getCanonicalDecl())
    return true;
  for (const CXXBaseSpecifier &Base : ThisClass.bases()) {
    auto *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    assert(BaseDecl);
    if (Parent.getCanonicalDecl() == BaseDecl->getCanonicalDecl())
      return true;
  }
  return false;
}

static BasesVector getParentsByGrandParent(const CXXRecordDecl &GrandParent,
                                           const CXXRecordDecl &ThisClass,
                                           const CXXMethodDecl &MemberDecl) {
  BasesVector Result;
  for (const auto &Base : ThisClass.bases()) {
    const auto *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    const CXXMethodDecl *ActualMemberDecl =
        MemberDecl.getCorrespondingMethodInClass(BaseDecl);
    if (!ActualMemberDecl)
      continue;
    // TypePtr is the nearest base class to ThisClass between ThisClass and
    // GrandParent, where MemberDecl is overridden. TypePtr is the class the
    // check proposes to fix to.
    const Type *TypePtr =
        ActualMemberDecl->getThisType(ActualMemberDecl->getASTContext())
            .getTypePtr();
    const CXXRecordDecl *RecordDeclType = TypePtr->getPointeeCXXRecordDecl();
    assert(RecordDeclType && "TypePtr is not a pointer to CXXRecordDecl!");
    if (RecordDeclType->getCanonicalDecl()->isDerivedFrom(&GrandParent))
      Result.emplace_back(RecordDeclType);
  }

  return Result;
}

static std::string getNameAsString(const NamedDecl *Decl) {
  std::string QualName;
  llvm::raw_string_ostream OS(QualName);
  PrintingPolicy PP(Decl->getASTContext().getPrintingPolicy());
  PP.SuppressUnwrittenScope = true;
  Decl->printQualifiedName(OS, PP);
  return OS.str();
}

// Returns E as written in the source code. Used to handle 'using' and
// 'typedef'ed names of grand-parent classes.
static std::string getExprAsString(const clang::Expr &E,
                                   clang::ASTContext &AC) {
  std::string Text = tooling::fixit::getText(E, AC).str();
  Text.erase(
      std::remove_if(
          Text.begin(), Text.end(),
          [](char c) { return std::isspace(static_cast<unsigned char>(c)); }),
      Text.end());
  return Text;
}

void ParentVirtualCallCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(memberExpr(hasDescendant(implicitCastExpr(
                                hasImplicitDestinationType(pointsTo(
                                    type(anything()).bind("castToType"))),
                                hasSourceExpression(cxxThisExpr(hasType(
                                    type(anything()).bind("thisType")))))))
                     .bind("member")),
          callee(cxxMethodDecl(isVirtual())))
          .bind("call"),
      this);
}

void ParentVirtualCallCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  assert(MatchedDecl);

  const auto *Member = Result.Nodes.getNodeAs<MemberExpr>("member");
  assert(Member);

  if (!Member->getQualifier())
    return;

  const auto *MemberDecl = cast<CXXMethodDecl>(Member->getMemberDecl());

  const auto *ThisTypePtr = Result.Nodes.getNodeAs<PointerType>("thisType");
  assert(ThisTypePtr);

  const auto *ThisType = ThisTypePtr->getPointeeCXXRecordDecl();
  assert(ThisType);

  const auto *CastToTypePtr = Result.Nodes.getNodeAs<Type>("castToType");
  assert(CastToTypePtr);

  const auto *CastToType = CastToTypePtr->getAsCXXRecordDecl();
  assert(CastToType);

  if (isParentOf(*CastToType, *ThisType))
    return;

  const BasesVector Parents =
      getParentsByGrandParent(*CastToType, *ThisType, *MemberDecl);

  if (Parents.empty())
    return;

  std::string ParentsStr;
  ParentsStr.reserve(30 * Parents.size());
  for (const CXXRecordDecl *Parent : Parents) {
    if (!ParentsStr.empty())
      ParentsStr.append(" or ");
    ParentsStr.append("'").append(getNameAsString(Parent)).append("'");
  }

  assert(Member->getQualifierLoc().getSourceRange().getBegin().isValid());
  auto Diag = diag(Member->getQualifierLoc().getSourceRange().getBegin(),
                   "qualified name '%0' refers to a member overridden "
                   "in subclass%1; did you mean %2?")
              << getExprAsString(*Member, *Result.Context)
              << (Parents.size() > 1 ? "es" : "") << ParentsStr;

  // Propose a fix if there's only one parent class...
  if (Parents.size() == 1 &&
      // ...unless parent class is templated
      !isa<ClassTemplateSpecializationDecl>(Parents.front()))
    Diag << FixItHint::CreateReplacement(
        Member->getQualifierLoc().getSourceRange(),
        getNameAsString(Parents.front()) + "::");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
