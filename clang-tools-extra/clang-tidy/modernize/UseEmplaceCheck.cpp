//===--- UseEmplaceCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseEmplaceCheck.h"
#include "../utils/OptionsUtils.h"
using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
// Identical to hasAnyName, except it does not take template specifiers into
// account. This is used to match the functions names as in
// DefaultEmplacyFunctions below without caring about the template types of the
// containers.
AST_MATCHER_P(NamedDecl, hasAnyNameIgnoringTemplates, std::vector<StringRef>,
              Names) {
  const std::string FullName = "::" + Node.getQualifiedNameAsString();

  // This loop removes template specifiers by only keeping characters not within
  // template brackets. We keep a depth count to handle nested templates. For
  // example, it'll transform a::b<c<d>>::e<f> to simply a::b::e.
  std::string FullNameTrimmed;
  int Depth = 0;
  for (const auto &Character : FullName) {
    if (Character == '<') {
      ++Depth;
    } else if (Character == '>') {
      --Depth;
    } else if (Depth == 0) {
      FullNameTrimmed.append(1, Character);
    }
  }

  // This loop is taken from HasNameMatcher::matchesNodeFullSlow in
  // clang/lib/ASTMatchers/ASTMatchersInternal.cpp and checks whether
  // FullNameTrimmed matches any of the given Names.
  const StringRef FullNameTrimmedRef = FullNameTrimmed;
  for (const StringRef Pattern : Names) {
    if (Pattern.startswith("::")) {
      if (FullNameTrimmed == Pattern)
        return true;
    } else if (FullNameTrimmedRef.endswith(Pattern) &&
               FullNameTrimmedRef.drop_back(Pattern.size()).endswith("::")) {
      return true;
    }
  }

  return false;
}

// Checks if the given matcher is the last argument of the given CallExpr.
AST_MATCHER_P(CallExpr, hasLastArgument,
              clang::ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  if (Node.getNumArgs() == 0)
    return false;

  return InnerMatcher.matches(*Node.getArg(Node.getNumArgs() - 1), Finder,
                              Builder);
}

// Checks if the given member call has the same number of arguments as the
// function had parameters defined (this is useful to check if there is only one
// variadic argument).
AST_MATCHER(CXXMemberCallExpr, hasSameNumArgsAsDeclNumParams) {
  if (Node.getMethodDecl()->isFunctionTemplateSpecialization())
    return Node.getNumArgs() == Node.getMethodDecl()
                                    ->getPrimaryTemplate()
                                    ->getTemplatedDecl()
                                    ->getNumParams();

  return Node.getNumArgs() == Node.getMethodDecl()->getNumParams();
}

AST_MATCHER(DeclRefExpr, hasExplicitTemplateArgs) {
  return Node.hasExplicitTemplateArgs();
}

const auto DefaultContainersWithPushBack =
    "::std::vector; ::std::list; ::std::deque";
const auto DefaultContainersWithPush =
    "::std::stack; ::std::queue; ::std::priority_queue";
const auto DefaultContainersWithPushFront =
    "::std::forward_list; ::std::list; ::std::deque";
const auto DefaultSmartPointers =
    "::std::shared_ptr; ::std::unique_ptr; ::std::auto_ptr; ::std::weak_ptr";
const auto DefaultTupleTypes = "::std::pair; ::std::tuple";
const auto DefaultTupleMakeFunctions = "::std::make_pair; ::std::make_tuple";
const auto DefaultEmplacyFunctions =
    "vector::emplace_back; vector::emplace;"
    "deque::emplace; deque::emplace_front; deque::emplace_back;"
    "forward_list::emplace_after; forward_list::emplace_front;"
    "list::emplace; list::emplace_back; list::emplace_front;"
    "set::emplace; set::emplace_hint;"
    "map::emplace; map::emplace_hint;"
    "multiset::emplace; multiset::emplace_hint;"
    "multimap::emplace; multimap::emplace_hint;"
    "unordered_set::emplace; unordered_set::emplace_hint;"
    "unordered_map::emplace; unordered_map::emplace_hint;"
    "unordered_multiset::emplace; unordered_multiset::emplace_hint;"
    "unordered_multimap::emplace; unordered_multimap::emplace_hint;"
    "stack::emplace; queue::emplace; priority_queue::emplace";
} // namespace

UseEmplaceCheck::UseEmplaceCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), IgnoreImplicitConstructors(Options.get(
                                         "IgnoreImplicitConstructors", false)),
      ContainersWithPushBack(utils::options::parseStringList(Options.get(
          "ContainersWithPushBack", DefaultContainersWithPushBack))),
      ContainersWithPush(utils::options::parseStringList(
          Options.get("ContainersWithPush", DefaultContainersWithPush))),
      ContainersWithPushFront(utils::options::parseStringList(Options.get(
          "ContainersWithPushFront", DefaultContainersWithPushFront))),
      SmartPointers(utils::options::parseStringList(
          Options.get("SmartPointers", DefaultSmartPointers))),
      TupleTypes(utils::options::parseStringList(
          Options.get("TupleTypes", DefaultTupleTypes))),
      TupleMakeFunctions(utils::options::parseStringList(
          Options.get("TupleMakeFunctions", DefaultTupleMakeFunctions))),
      EmplacyFunctions(utils::options::parseStringList(
          Options.get("EmplacyFunctions", DefaultEmplacyFunctions))) {}

void UseEmplaceCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME: Bunch of functionality that could be easily added:
  // + add handling of `insert` for stl associative container, but be careful
  // because this requires special treatment (it could cause performance
  // regression)
  // + match for emplace calls that should be replaced with insertion
  auto CallPushBack = cxxMemberCallExpr(
      hasDeclaration(functionDecl(hasName("push_back"))),
      on(hasType(hasCanonicalType(
          hasDeclaration(cxxRecordDecl(hasAnyName(ContainersWithPushBack)))))));

  auto CallPush =
      cxxMemberCallExpr(hasDeclaration(functionDecl(hasName("push"))),
                        on(hasType(hasCanonicalType(hasDeclaration(
                            cxxRecordDecl(hasAnyName(ContainersWithPush)))))));

  auto CallPushFront = cxxMemberCallExpr(
      hasDeclaration(functionDecl(hasName("push_front"))),
      on(hasType(hasCanonicalType(hasDeclaration(
          cxxRecordDecl(hasAnyName(ContainersWithPushFront)))))));

  auto CallEmplacy = cxxMemberCallExpr(
      hasDeclaration(
          functionDecl(hasAnyNameIgnoringTemplates(EmplacyFunctions))),
      on(hasType(hasCanonicalType(hasDeclaration(has(typedefNameDecl(
          hasName("value_type"), hasType(type(hasUnqualifiedDesugaredType(
                                     recordType().bind("value_type")))))))))));

  // We can't replace push_backs of smart pointer because
  // if emplacement fails (f.e. bad_alloc in vector) we will have leak of
  // passed pointer because smart pointer won't be constructed
  // (and destructed) as in push_back case.
  auto IsCtorOfSmartPtr =
      hasDeclaration(cxxConstructorDecl(ofClass(hasAnyName(SmartPointers))));

  // Bitfields binds only to consts and emplace_back take it by universal ref.
  auto BitFieldAsArgument = hasAnyArgument(
      ignoringImplicit(memberExpr(hasDeclaration(fieldDecl(isBitField())))));

  // Initializer list can't be passed to universal reference.
  auto InitializerListAsArgument = hasAnyArgument(
      ignoringImplicit(allOf(cxxConstructExpr(isListInitialization()),
                             unless(cxxTemporaryObjectExpr()))));

  // We could have leak of resource.
  auto NewExprAsArgument = hasAnyArgument(ignoringImplicit(cxxNewExpr()));
  // We would call another constructor.
  auto ConstructingDerived =
      hasParent(implicitCastExpr(hasCastKind(CastKind::CK_DerivedToBase)));

  // emplace_back can't access private or protected constructors.
  auto IsPrivateOrProtectedCtor =
      hasDeclaration(cxxConstructorDecl(anyOf(isPrivate(), isProtected())));

  auto HasInitList = anyOf(has(ignoringImplicit(initListExpr())),
                           has(cxxStdInitializerListExpr()));

  // FIXME: Discard 0/NULL (as nullptr), static inline const data members,
  // overloaded functions and template names.
  auto SoughtConstructExpr =
      cxxConstructExpr(
          unless(anyOf(IsCtorOfSmartPtr, HasInitList, BitFieldAsArgument,
                       InitializerListAsArgument, NewExprAsArgument,
                       ConstructingDerived, IsPrivateOrProtectedCtor)))
          .bind("ctor");
  auto HasConstructExpr = has(ignoringImplicit(SoughtConstructExpr));

  // allow for T{} to be replaced, even if no CTOR is declared
  auto HasConstructInitListExpr = has(initListExpr(anyOf(
      allOf(has(SoughtConstructExpr),
            has(cxxConstructExpr(argumentCountIs(0)))),
      has(cxxBindTemporaryExpr(has(SoughtConstructExpr),
                               has(cxxConstructExpr(argumentCountIs(0))))))));
  auto HasBracedInitListExpr =
      anyOf(has(cxxBindTemporaryExpr(HasConstructInitListExpr)),
            HasConstructInitListExpr);

  auto MakeTuple = ignoringImplicit(
      callExpr(callee(expr(ignoringImplicit(declRefExpr(
                   unless(hasExplicitTemplateArgs()),
                   to(functionDecl(hasAnyName(TupleMakeFunctions))))))))
          .bind("make"));

  // make_something can return type convertible to container's element type.
  // Allow the conversion only on containers of pairs.
  auto MakeTupleCtor = ignoringImplicit(cxxConstructExpr(
      has(materializeTemporaryExpr(MakeTuple)),
      hasDeclaration(cxxConstructorDecl(ofClass(hasAnyName(TupleTypes))))));

  auto SoughtParam =
      materializeTemporaryExpr(
          anyOf(has(MakeTuple), has(MakeTupleCtor), HasConstructExpr,
                HasBracedInitListExpr,
                has(cxxFunctionalCastExpr(HasConstructExpr)),
                has(cxxFunctionalCastExpr(HasBracedInitListExpr))))
          .bind("temporary_expr");

  auto HasConstructExprWithValueTypeType =
      has(ignoringImplicit(cxxConstructExpr(
          SoughtConstructExpr, hasType(type(hasUnqualifiedDesugaredType(
                                   type(equalsBoundNode("value_type"))))))));

  auto HasBracedInitListWithValueTypeType =
      anyOf(allOf(HasConstructInitListExpr,
                  has(initListExpr(hasType(type(hasUnqualifiedDesugaredType(
                      type(equalsBoundNode("value_type")))))))),
            has(cxxBindTemporaryExpr(
                HasConstructInitListExpr,
                has(initListExpr(hasType(type(hasUnqualifiedDesugaredType(
                    type(equalsBoundNode("value_type"))))))))));

  auto HasConstructExprWithValueTypeTypeAsLastArgument = hasLastArgument(
      materializeTemporaryExpr(
          anyOf(HasConstructExprWithValueTypeType,
                HasBracedInitListWithValueTypeType,
                has(cxxFunctionalCastExpr(HasConstructExprWithValueTypeType)),
                has(cxxFunctionalCastExpr(HasBracedInitListWithValueTypeType))))
          .bind("temporary_expr"));

  Finder->addMatcher(
      traverse(TK_AsIs, cxxMemberCallExpr(CallPushBack, has(SoughtParam),
                                          unless(isInTemplateInstantiation()))
                            .bind("push_back_call")),
      this);

  Finder->addMatcher(
      traverse(TK_AsIs, cxxMemberCallExpr(CallPush, has(SoughtParam),
                                          unless(isInTemplateInstantiation()))
                            .bind("push_call")),
      this);

  Finder->addMatcher(
      traverse(TK_AsIs, cxxMemberCallExpr(CallPushFront, has(SoughtParam),
                                          unless(isInTemplateInstantiation()))
                            .bind("push_front_call")),
      this);

  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxMemberCallExpr(
                   CallEmplacy, HasConstructExprWithValueTypeTypeAsLastArgument,
                   hasSameNumArgsAsDeclNumParams(),
                   unless(isInTemplateInstantiation()))
                   .bind("emplacy_call")),
      this);

  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxMemberCallExpr(
              CallEmplacy,
              on(hasType(cxxRecordDecl(has(typedefNameDecl(
                  hasName("value_type"),
                  hasType(type(
                      hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                          cxxRecordDecl(hasAnyName(SmallVector<StringRef, 2>(
                              TupleTypes.begin(), TupleTypes.end()))))))))))))),
              has(MakeTuple), hasSameNumArgsAsDeclNumParams(),
              unless(isInTemplateInstantiation()))
              .bind("emplacy_call")),
      this);
}

void UseEmplaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *PushBackCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("push_back_call");
  const auto *PushCall = Result.Nodes.getNodeAs<CXXMemberCallExpr>("push_call");
  const auto *PushFrontCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("push_front_call");
  const auto *EmplacyCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("emplacy_call");
  const auto *CtorCall = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  const auto *MakeCall = Result.Nodes.getNodeAs<CallExpr>("make");
  const auto *TemporaryExpr =
      Result.Nodes.getNodeAs<MaterializeTemporaryExpr>("temporary_expr");

  const CXXMemberCallExpr *Call = [&]() {
    if (PushBackCall) {
      return PushBackCall;
    }
    if (PushCall) {
      return PushCall;
    }
    if (PushFrontCall) {
      return PushFrontCall;
    }
    return EmplacyCall;
  }();

  assert(Call && "No call matched");
  assert((CtorCall || MakeCall) && "No push_back parameter matched");

  if (IgnoreImplicitConstructors && CtorCall && CtorCall->getNumArgs() >= 1 &&
      CtorCall->getArg(0)->getSourceRange() == CtorCall->getSourceRange())
    return;

  const auto FunctionNameSourceRange = CharSourceRange::getCharRange(
      Call->getExprLoc(), Call->getArg(0)->getExprLoc());

  auto Diag =
      EmplacyCall
          ? diag(TemporaryExpr ? TemporaryExpr->getBeginLoc()
                 : CtorCall    ? CtorCall->getBeginLoc()
                               : MakeCall->getBeginLoc(),
                 "unnecessary temporary object created while calling %0")
          : diag(Call->getExprLoc(), "use emplace%select{|_back|_front}0 "
                                     "instead of push%select{|_back|_front}0");
  if (EmplacyCall)
    Diag << Call->getMethodDecl()->getName();
  else if (PushCall)
    Diag << 0;
  else if (PushBackCall)
    Diag << 1;
  else
    Diag << 2;

  if (FunctionNameSourceRange.getBegin().isMacroID())
    return;

  if (PushBackCall) {
    const char *EmplacePrefix = MakeCall ? "emplace_back" : "emplace_back(";
    Diag << FixItHint::CreateReplacement(FunctionNameSourceRange,
                                         EmplacePrefix);
  } else if (PushCall) {
    const char *EmplacePrefix = MakeCall ? "emplace" : "emplace(";
    Diag << FixItHint::CreateReplacement(FunctionNameSourceRange,
                                         EmplacePrefix);
  } else if (PushFrontCall) {
    const char *EmplacePrefix = MakeCall ? "emplace_front" : "emplace_front(";
    Diag << FixItHint::CreateReplacement(FunctionNameSourceRange,
                                         EmplacePrefix);
  }

  const SourceRange CallParensRange =
      MakeCall ? SourceRange(MakeCall->getCallee()->getEndLoc(),
                             MakeCall->getRParenLoc())
               : CtorCall->getParenOrBraceRange();

  // Finish if there is no explicit constructor call.
  if (CallParensRange.getBegin().isInvalid())
    return;

  // FIXME: Will there ever be a CtorCall, if there is no TemporaryExpr?
  const SourceLocation ExprBegin = TemporaryExpr ? TemporaryExpr->getExprLoc()
                                   : CtorCall    ? CtorCall->getExprLoc()
                                                 : MakeCall->getExprLoc();

  // Range for constructor name and opening brace.
  const auto ParamCallSourceRange =
      CharSourceRange::getTokenRange(ExprBegin, CallParensRange.getBegin());

  // Range for constructor closing brace and end of temporary expr.
  const auto EndCallSourceRange = CharSourceRange::getTokenRange(
      CallParensRange.getEnd(),
      TemporaryExpr ? TemporaryExpr->getEndLoc() : CallParensRange.getEnd());

  Diag << FixItHint::CreateRemoval(ParamCallSourceRange)
       << FixItHint::CreateRemoval(EndCallSourceRange);

  if (MakeCall && EmplacyCall) {
    // Remove extra left parenthesis
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(MakeCall->getCallee()->getEndLoc(),
                                      MakeCall->getArg(0)->getBeginLoc()));
  }
}

void UseEmplaceCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreImplicitConstructors", IgnoreImplicitConstructors);
  Options.store(Opts, "ContainersWithPushBack",
                utils::options::serializeStringList(ContainersWithPushBack));
  Options.store(Opts, "ContainersWithPush",
                utils::options::serializeStringList(ContainersWithPush));
  Options.store(Opts, "ContainersWithPushFront",
                utils::options::serializeStringList(ContainersWithPushFront));
  Options.store(Opts, "SmartPointers",
                utils::options::serializeStringList(SmartPointers));
  Options.store(Opts, "TupleTypes",
                utils::options::serializeStringList(TupleTypes));
  Options.store(Opts, "TupleMakeFunctions",
                utils::options::serializeStringList(TupleMakeFunctions));
  Options.store(Opts, "EmplacyFunctions",
                utils::options::serializeStringList(EmplacyFunctions));
}

} // namespace clang::tidy::modernize
