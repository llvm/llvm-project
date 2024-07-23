//===--- IncorrectIteratorsCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectIteratorsCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
using SVU = llvm::SmallVector<unsigned, 3>;
/// Checks to see if a all the parameters of a template function with a given
/// index refer to the same type.
AST_MATCHER_P(FunctionDecl, areParametersSameTemplateType, SVU, Indexes) {
  const FunctionTemplateDecl *TemplateDecl = Node.getPrimaryTemplate();
  if (!TemplateDecl)
    return false;
  const FunctionDecl *FuncDecl = TemplateDecl->getTemplatedDecl();
  if (!FuncDecl)
    return false;
  assert(!Indexes.empty());
  if (llvm::any_of(Indexes, [Count(FuncDecl->getNumParams())](unsigned Index) {
        return Index >= Count;
      }))
    return false;
  const ParmVarDecl *FirstParam = FuncDecl->getParamDecl(Indexes.front());
  if (!FirstParam)
    return false;
  QualType Type = FirstParam->getOriginalType();
  for (auto Item : llvm::drop_begin(Indexes)) {
    const ParmVarDecl *Param = FuncDecl->getParamDecl(Item);
    if (!Param)
      return false;
    if (Param->getOriginalType() != Type)
      return false;
  }
  return true;
}

AST_MATCHER_P(FunctionDecl, isParameterTypeUnique, unsigned, Index) {
  const FunctionTemplateDecl *TemplateDecl = Node.getPrimaryTemplate();
  if (!TemplateDecl)
    return false;
  const FunctionDecl *FuncDecl = TemplateDecl->getTemplatedDecl();
  if (!FuncDecl)
    return false;
  if (Index >= FuncDecl->getNumParams())
    return false;
  const ParmVarDecl *MainParam = FuncDecl->getParamDecl(Index);
  if (!MainParam)
    return false;
  QualType Type = MainParam->getOriginalType();
  for (unsigned I = 0, E = FuncDecl->getNumParams(); I != E; ++I) {
    if (I == Index)
      continue;
    const ParmVarDecl *Param = FuncDecl->getParamDecl(I);
    if (!Param)
      continue;
    if (Param->getOriginalType() == Type)
      return false;
  }
  return true;
}

AST_MATCHER(Expr, isNegativeIntegerConstant) {
  if (auto Res = Node.getIntegerConstantExpr(Finder->getASTContext()))
    return Res->isNegative();
  return false;
}

struct NameMatchers {
  ArrayRef<StringRef> FreeNames;
  ArrayRef<StringRef> MethodNames;
};

struct NamePairs {
  NameMatchers BeginNames;
  NameMatchers EndNames;
};

struct FullState {
  NamePairs Forward;
  NamePairs Reversed;
  NamePairs Combined;
  NameMatchers All;
  ArrayRef<StringRef> MakeReverseIterator;
};

struct PairOverload {
  unsigned Begin;
  unsigned End;
  bool IsActive = true;
};

struct PairMethods {

  StringRef Name;
  SmallVector<PairOverload, 1> Indexes;
};

struct InternalMethods {
  struct Indexes {
    unsigned Index;
    bool IsActive = true;
  };
  StringRef Name;
  SmallVector<Indexes, 2> Indexes;
};

struct Descriptor {
  SmallVector<PairMethods> Ranges;
  SmallVector<InternalMethods> Internal;
  SmallVector<PairOverload, 1> Constructor;
};

struct AllowedContainer {
  StringRef Name;
  bool IsActive = true;
};
} // namespace

static constexpr char FirstRangeArg[] = "FirstRangeArg";
static constexpr char FirstRangeArgExpr[] = "FirstRangeArgExpr";
static constexpr char ReverseBeginBind[] = "ReverseBeginBind";
static constexpr char ReverseEndBind[] = "ReverseEndBind";
static constexpr char SecondRangeArg[] = "SecondRangeArg";
static constexpr char SecondRangeArgExpr[] = "SecondRangeArgExpr";
static constexpr char ArgMismatchBegin[] = "ArgMismatchBegin";
static constexpr char ArgMismatchEnd[] = "ArgMismatchEnd";
static constexpr char ArgMismatchExpr[] = "ArgMismatchExpr";
static constexpr char ArgMismatchRevBind[] = "ArgMismatchRevBind";
static constexpr char Callee[] = "Callee";
static constexpr char Internal[] = "Internal";
static constexpr char InternalOther[] = "InternalOther";
static constexpr char InternalArgument[] = "InternalArgument";
static constexpr char OutputRangeEnd[] = "OutputRangeEnd";
static constexpr char OutputRangeBegin[] = "OutputRangeBegin";
static constexpr char UnexpectedDec[] = "UndexpectedDec";
static constexpr char UnexpectedInc[] = "UnexpectedInc";
static constexpr char UnexpectedIncDecExpr[] = "UnexpectedIncDecExpr";
static constexpr char TriRangeArgBegin[] = "TriRangeArgBegin";
static constexpr char TriRangeArgEnd[] = "TriRangeArgEnd";
static constexpr char TriRangeArgExpr[] = "TriRangeArgExpr";

static auto
makeExprMatcher(ast_matchers::internal::Matcher<Expr> ArgumentMatcher,
                const NameMatchers &Names, ArrayRef<StringRef> MakeReverse,
                const NameMatchers &RevNames, StringRef RevBind) {
  auto MakeMatcher = [&ArgumentMatcher](const NameMatchers &Names) {
    return anyOf(
        cxxMemberCallExpr(argumentCountIs(0),
                          callee(cxxMethodDecl(hasAnyName(Names.MethodNames))),
                          on(ArgumentMatcher)),
        callExpr(argumentCountIs(1),
                 hasDeclaration(functionDecl(hasAnyName(Names.FreeNames))),
                 hasArgument(0, ArgumentMatcher)));
  };
  return expr(anyOf(MakeMatcher(Names),
                    callExpr(argumentCountIs(1),
                             callee(functionDecl(hasAnyName(MakeReverse))),
                             hasArgument(0, MakeMatcher(RevNames)))
                        .bind(RevBind)));
}

/// Detects a range function where the argument for the begin call differs from
/// the end call
/// @code
///   std::find(I.begin(), J.end());
/// @endcode
static auto makeRangeArgMismatch(unsigned BeginExpected, unsigned EndExpected,
                                 NamePairs Forward, NamePairs Reverse,
                                 ArrayRef<StringRef> MakeReverse) {

  auto MakeMatcher = [&MakeReverse, &BeginExpected,
                      &EndExpected](auto Primary, auto Backwards) {
    return allOf(
        hasArgument(BeginExpected,
                    makeExprMatcher(expr().bind(FirstRangeArg),
                                    Primary.BeginNames, MakeReverse,
                                    Backwards.EndNames, ReverseBeginBind)
                        .bind(FirstRangeArgExpr)),
        hasArgument(EndExpected,
                    makeExprMatcher(
                        expr(unless(matchers::isStatementIdenticalToBoundNode(
                                 FirstRangeArg)))
                            .bind(SecondRangeArg),
                        Primary.EndNames, MakeReverse, Backwards.BeginNames,
                        ReverseEndBind)
                        .bind(SecondRangeArgExpr)));
  };

  return allOf(
      argumentCountAtLeast(std::max(BeginExpected, EndExpected) + 1),
      anyOf(MakeMatcher(Forward, Reverse), MakeMatcher(Reverse, Forward)));
}

/// Detects a range function where we expect a call to begin but get a call to
/// end or vice versa
/// @code
///   std::find(X.end(), X.end());
/// @endcode
/// First argument likely should be begin
static auto makeUnexpectedBeginEndMatcher(unsigned Index, StringRef BindTo,
                                          NameMatchers Names,
                                          ArrayRef<StringRef> MakeReverse,
                                          const NameMatchers &RevNames) {
  return hasArgument(Index,
                     makeExprMatcher(expr().bind(BindTo), Names, MakeReverse,
                                     RevNames, ArgMismatchRevBind)
                         .bind(ArgMismatchExpr));
}

static auto makeUnexpectedBeginEndPair(unsigned BeginExpected,
                                       unsigned EndExpected,
                                       NamePairs BeginEndPairs,
                                       ArrayRef<StringRef> MakeReverse) {
  return eachOf(makeUnexpectedBeginEndMatcher(
                    BeginExpected, ArgMismatchBegin, BeginEndPairs.EndNames,
                    MakeReverse, BeginEndPairs.BeginNames),
                makeUnexpectedBeginEndMatcher(
                    EndExpected, ArgMismatchEnd, BeginEndPairs.BeginNames,
                    MakeReverse, BeginEndPairs.EndNames));
}

static auto makePairRangeMatcherInternal(
    ast_matchers::internal::Matcher<NamedDecl> NameMatcher,
    unsigned BeginExpected, unsigned EndExpected, const FullState &State) {
  return callExpr(
      callee(functionDecl(
          areParametersSameTemplateType({BeginExpected, EndExpected}),
          std::move(NameMatcher))),
      anyOf(makeRangeArgMismatch(BeginExpected, EndExpected, State.Forward,
                                 State.Reversed, State.MakeReverseIterator),
            makeUnexpectedBeginEndPair(BeginExpected, EndExpected,
                                       State.Combined,
                                       State.MakeReverseIterator)));
}

static auto make3RangeMatcherInternal(ArrayRef<StringRef> FuncNames,
                                      unsigned BeginExpected,
                                      unsigned MiddleExpected,
                                      unsigned EndExpected,
                                      const FullState &State) {
  return callExpr(
             callee(
                 functionDecl(areParametersSameTemplateType(
                                  {BeginExpected, EndExpected, MiddleExpected}),
                              hasAnyName(FuncNames))),
             eachOf(
                 anyOf(makeRangeArgMismatch(BeginExpected, EndExpected,
                                            State.Forward, State.Reversed,
                                            State.MakeReverseIterator),
                       makeUnexpectedBeginEndPair(BeginExpected, EndExpected,
                                                  State.Combined,
                                                  State.MakeReverseIterator)),
                 anyOf(allOf(hasArgument(
                                 MiddleExpected,
                                 makeExprMatcher(expr().bind(TriRangeArgExpr),
                                                 State.Forward.BeginNames,
                                                 State.MakeReverseIterator,
                                                 State.Reversed.EndNames,
                                                 ArgMismatchRevBind)
                                     .bind(TriRangeArgBegin)),
                             hasArgument(
                                 EndExpected,
                                 makeExprMatcher(
                                     matchers::isStatementIdenticalToBoundNode(
                                         TriRangeArgExpr),
                                     State.Forward.EndNames,
                                     State.MakeReverseIterator,
                                     State.Reversed.BeginNames, ""))),
                       allOf(hasArgument(
                                 MiddleExpected,
                                 makeExprMatcher(expr().bind(TriRangeArgExpr),
                                                 State.Forward.EndNames,
                                                 State.MakeReverseIterator,
                                                 State.Reversed.BeginNames,
                                                 ArgMismatchRevBind)
                                     .bind(TriRangeArgEnd)),
                             hasArgument(
                                 BeginExpected,
                                 makeExprMatcher(
                                     matchers::isStatementIdenticalToBoundNode(
                                         TriRangeArgExpr),
                                     State.Forward.BeginNames,
                                     State.MakeReverseIterator,
                                     State.Reversed.EndNames, ""))))))
      .bind(Callee);
}

/// The full matcher for functions that take a range with 2 arguments
static auto makePairRangeMatcher(ArrayRef<StringRef> FuncNames,
                                 unsigned BeginExpected, unsigned EndExpected,
                                 const FullState &State) {
  return makePairRangeMatcherInternal(hasAnyName(FuncNames), BeginExpected,
                                      EndExpected, State)
      .bind(Callee);
}

static auto makeHalfOpenMatcher(ArrayRef<StringRef> FuncNames,
                                unsigned BeginExpected, unsigned PotentialEnd,
                                const FullState &State) {
  auto NameMatcher = hasAnyName(FuncNames);
  return callExpr(
             anyOf(makePairRangeMatcherInternal(NameMatcher, BeginExpected,
                                                PotentialEnd, State),
                   allOf(callee(functionDecl(NameMatcher, isParameterTypeUnique(
                                                              BeginExpected))),
                         makeUnexpectedBeginEndMatcher(
                             BeginExpected, ArgMismatchBegin,
                             State.Combined.EndNames, State.MakeReverseIterator,
                             State.Combined.BeginNames))))
      .bind(Callee);
}

/// Detects if a function has a policy for the first argument.
/// If no policy detected, runs @param F matcher with the expected index,
/// otherwise rungs @param F matcher with expected index + 1 to account for the
/// the policy argument
template <typename Func, typename... Args>
auto runWithPolicy1(ArrayRef<StringRef> FuncNames, unsigned Expected, Func F,
                    Args &&...A) {
  return callExpr(anyOf(
      allOf(callee(functionDecl(areParametersSameTemplateType({0, 1}))),
            F(FuncNames, Expected, std::forward<Args>(A)...)),
      allOf(callee(functionDecl(unless(areParametersSameTemplateType({0, 1})))),
            F(FuncNames, Expected + 1, std::forward<Args>(A)...))));
}

/// Like @c runWithPolicy1 only it handles 2 arguments
template <typename Func, typename... Args>
auto runWithPolicy2(ArrayRef<StringRef> FuncNames, unsigned BeginExpected,
                    unsigned EndExpected, Func F, Args &&...A) {
  return callExpr(anyOf(
      allOf(callee(functionDecl(areParametersSameTemplateType({0, 1}))),
            F(FuncNames, BeginExpected, EndExpected, std::forward<Args>(A)...)),
      allOf(callee(functionDecl(unless(areParametersSameTemplateType({0, 1})))),
            F(FuncNames, BeginExpected + 1, EndExpected + 1,
              std::forward<Args>(A)...))));
}

/// Like @c runWithPolicy1 only it handles 2 arguments
template <typename Func, typename... Args>
auto runWithPolicy3(ArrayRef<StringRef> FuncNames, unsigned BeginExpected,
                    unsigned MiddleExpected, unsigned EndExpected, Func F,
                    Args &&...A) {
  return callExpr(anyOf(
      allOf(callee(functionDecl(areParametersSameTemplateType({0, 1}))),
            F(FuncNames, BeginExpected, MiddleExpected, EndExpected,
              std::forward<Args>(A)...)),
      allOf(callee(functionDecl(unless(areParametersSameTemplateType({0, 1})))),
            F(FuncNames, BeginExpected + 1, MiddleExpected + 1, EndExpected + 1,
              std::forward<Args>(A)...))));
}

static auto makeNamedExpectedBeginFullMatcher(ArrayRef<StringRef> FuncNames,
                                              unsigned ExpectedIndex,
                                              const FullState &State,
                                              StringRef BindTo) {
  return callExpr(argumentCountAtLeast(ExpectedIndex + 1),
                  callee(functionDecl(isParameterTypeUnique(ExpectedIndex),
                                      hasAnyName(FuncNames))),
                  makeUnexpectedBeginEndMatcher(
                      ExpectedIndex, BindTo, State.Combined.EndNames,
                      State.MakeReverseIterator, State.Combined.BeginNames))
      .bind(Callee);
}

/// Detects calls where a single output iterator is expected, yet an end of
/// container input is supplied Usually these arguments would be supplied with
/// things like `std::back_inserter`
static auto makeExpectedOutputFullMatcher(ArrayRef<StringRef> FuncNames,
                                          unsigned ExpectedIndex,
                                          const FullState &State) {
  return makeNamedExpectedBeginFullMatcher(FuncNames, ExpectedIndex, State,
                                           OutputRangeEnd);
}

/// Detects calls where a begin iterator is expected, yet an end of container is
/// supplied.
static auto makeExpectedBeginFullMatcher(ArrayRef<StringRef> FuncNames,
                                         unsigned ExpectedIndex,
                                         const FullState &State) {
  return makeNamedExpectedBeginFullMatcher(FuncNames, ExpectedIndex, State,
                                           ArgMismatchBegin);
}

/// Detects calls where an end iterator is expected, yet a begin of container is
/// supplied.
static auto makeExpectedEndFullMatcher(ArrayRef<StringRef> FuncNames,
                                       unsigned ExpectedIndex,
                                       const FullState &State) {
  return callExpr(argumentCountAtLeast(ExpectedIndex + 1),
                  callee(functionDecl(isParameterTypeUnique(ExpectedIndex),
                                      hasAnyName(FuncNames))),
                  makeUnexpectedBeginEndMatcher(ExpectedIndex, OutputRangeBegin,
                                                State.Combined.BeginNames,
                                                State.MakeReverseIterator,
                                                State.Combined.EndNames))
      .bind(Callee);
}

/// Handles the mess of overloads that is std::transform
static auto makeTransformArgsMatcher(ArrayRef<StringRef> FuncNames,
                                     const FullState &State) {
  auto FnMatch = callee(functionDecl(hasAnyName(FuncNames)));
  auto MakeSubMatch = [&](bool IsPolicy) {
    auto Offset = IsPolicy ? 1 : 0;
    return anyOf(
        allOf(argumentCountIs(4 + Offset), FnMatch,
              makeUnexpectedBeginEndMatcher(
                  2 + Offset, OutputRangeEnd, State.Combined.EndNames,
                  State.MakeReverseIterator, State.Combined.BeginNames)),
        allOf(
            argumentCountIs(5 + Offset), FnMatch,
            eachOf(makeUnexpectedBeginEndMatcher(
                       2 + Offset, ArgMismatchBegin, State.Combined.EndNames,
                       State.MakeReverseIterator, State.Combined.BeginNames),
                   makeUnexpectedBeginEndMatcher(
                       3 + Offset, OutputRangeEnd, State.Combined.EndNames,
                       State.MakeReverseIterator, State.Combined.BeginNames))));
  };
  return callExpr(anyOf(allOf(callee(functionDecl(
                                  areParametersSameTemplateType({0, 1}))),
                              MakeSubMatch(false)),
                        allOf(callee(functionDecl(unless(
                                  areParametersSameTemplateType({0, 1})))),
                              MakeSubMatch(true))))
      .bind(Callee);
}

template <typename T>
static std::optional<ast_matchers::internal::Matcher<T>>
combineAnyOf(std::vector<ast_matchers::internal::DynTypedMatcher> &&Items) {
  if (Items.empty())
    return std::nullopt;
  if (Items.size() == 1) {
    return Items.front().convertTo<T>();
  }
  return ast_matchers::internal::DynTypedMatcher::constructVariadic(
             ast_matchers::internal::DynTypedMatcher::VO_AnyOf,
             ASTNodeKind::getFromNodeKind<T>(), std::move(Items))
      .template convertTo<T>();
}

template <typename T>
static std::optional<ast_matchers::internal::Matcher<T>>
combineEachOf(std::vector<ast_matchers::internal::DynTypedMatcher> &&Items) {
  if (Items.empty())
    return std::nullopt;
  if (Items.size() == 1) {
    return Items.front().convertTo<T>();
  }
  return ast_matchers::internal::DynTypedMatcher::constructVariadic(
             ast_matchers::internal::DynTypedMatcher::VO_EachOf,
             ASTNodeKind::getFromNodeKind<T>(), std::move(Items))
      .template convertTo<T>();
}

static std::optional<ast_matchers::internal::Matcher<CXXMemberCallExpr>>
getContainerRangeMatchers(ArrayRef<PairMethods> Methods,
                          const FullState &State) {
  std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;
  for (auto [Name, Ranges] : Methods) {
    for (auto [BeginExpected, EndExpected, IsActive] : Ranges) {
      if (!IsActive)
        continue;
      Matchers.emplace_back(cxxMemberCallExpr(
          callee(cxxMethodDecl(
              areParametersSameTemplateType({BeginExpected, EndExpected}),
              hasName(Name))),
          anyOf(makeRangeArgMismatch(BeginExpected, EndExpected, State.Forward,
                                     State.Reversed, State.MakeReverseIterator),
                makeUnexpectedBeginEndPair(BeginExpected, EndExpected,
                                           State.Combined,
                                           State.MakeReverseIterator))));
    }
  }
  return combineAnyOf<CXXMemberCallExpr>(std::move(Matchers));
}

static std::optional<ast_matchers::internal::Matcher<CXXMemberCallExpr>>
getContainerInternalMatcher(InternalMethods Method, const FullState &State) {
  std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;
  for (auto [InternalExpected, IsActive] : Method.Indexes) {
    if (!IsActive)
      continue;
    Matchers.emplace_back(cxxMemberCallExpr(
        callee(cxxMethodDecl(hasParameter(
            InternalExpected,
            parmVarDecl(hasType(
                elaboratedType(namesType(typedefType(hasDeclaration(namedDecl(
                    hasAnyName("const_iterator", "iterator"))))))))))),
        hasArgument(InternalExpected,
                    makeExprMatcher(
                        expr(unless(matchers::isStatementIdenticalToBoundNode(
                                 Internal)))
                            .bind(InternalOther),
                        State.All, State.MakeReverseIterator, State.All,
                        ArgMismatchRevBind)
                        .bind(InternalArgument))));
  }
  if (auto Combined = combineEachOf<CXXMemberCallExpr>(std::move(Matchers)))
    return cxxMemberCallExpr(callee(cxxMethodDecl(hasName(Method.Name))),
                             std::move(*Combined));
  return std::nullopt;
}

static std::optional<ast_matchers::internal::Matcher<CXXMemberCallExpr>>
getContainerInternalMatchers(ArrayRef<InternalMethods> Methods,
                             const FullState &State) {
  std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;
  for (auto Method : Methods) {
    if (auto Matcher = getContainerInternalMatcher(Method, State)) {
      Matchers.push_back(std::move(*Matcher));
    }
  }
  return combineAnyOf<CXXMemberCallExpr>(std::move(Matchers));
}

static std::optional<ast_matchers::internal::Matcher<CXXConstructExpr>>
getContainerConstructorMatchers(ArrayRef<PairOverload> Constructors,
                                const FullState &State) {
  std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;
  for (auto [BeginExpected, EndExpected, IsActive] : Constructors) {
    if (!IsActive)
      continue;
    Matchers.emplace_back(cxxConstructExpr(
        hasDeclaration(cxxConstructorDecl(
            areParametersSameTemplateType({BeginExpected, EndExpected}))),
        anyOf(makeRangeArgMismatch(BeginExpected, EndExpected, State.Forward,
                                   State.Reversed, State.MakeReverseIterator),
              makeUnexpectedBeginEndPair(BeginExpected, EndExpected,
                                         State.Combined,
                                         State.MakeReverseIterator))));
  }
  return combineAnyOf<CXXConstructExpr>(std::move(Matchers));
}

static auto registerContainerDescriptor(MatchFinder *Finder,
                                        ClangTidyCheck *Check,
                                        ArrayRef<AllowedContainer> Container,
                                        const Descriptor &D,
                                        const FullState &State) {
  auto ContainerNames = llvm::to_vector(
      llvm::map_range(llvm::make_filter_range(
                          Container, [](auto &Item) { return Item.IsActive; }),
                      [](auto &Item) { return Item.Name; }));
  if (ContainerNames.empty())
    return;
  auto RangeMatcher = getContainerRangeMatchers(D.Ranges, State);
  auto InternalMatcher = getContainerInternalMatchers(D.Internal, State);
  if (RangeMatcher || InternalMatcher)
    Finder->addMatcher(
        cxxMemberCallExpr(
            thisPointerType(cxxRecordDecl(hasAnyName(ContainerNames))),
            on(expr().bind(Internal)),
            (RangeMatcher && InternalMatcher)
                ? eachOf(std::move(*RangeMatcher), std::move(*InternalMatcher))
                : (RangeMatcher ? std::move(*RangeMatcher)
                                : std::move(*InternalMatcher)))
            .bind(Callee),
        Check);
  if (auto Ctor = getContainerConstructorMatchers(D.Constructor, State)) {
    Finder->addMatcher(
        cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                             ofClass(hasAnyName(ContainerNames)))),
                         std::move(*Ctor))
            .bind(Callee),
        Check);
  }
}

/// Looks for calls that advance past the end of a range or before the start of
/// a range.
static auto makeUnexpectedIncDecMatcher(const FullState &State, bool IsInc) {
  auto Arg =
      makeExprMatcher(
          expr(), IsInc ? State.Combined.EndNames : State.Combined.BeginNames,
          State.MakeReverseIterator,
          IsInc ? State.Combined.BeginNames : State.Combined.EndNames,
          ArgMismatchRevBind)
          .bind(UnexpectedIncDecExpr);
  return expr(
             anyOf(
                 callExpr(
                     argumentCountAtLeast(1),
                     anyOf(
                         allOf(
                             anyOf(unless(hasArgument(1, anything())),
                                   hasArgument(
                                       1, unless(isNegativeIntegerConstant()))),
                             callee(functionDecl(hasName(
                                 IsInc ? "::std::next" : "::std::prev")))),
                         allOf(hasArgument(1, isNegativeIntegerConstant()),
                               callee(functionDecl(hasName(
                                   IsInc ? "::std::prev" : "::std::next"))))),
                     hasArgument(0, Arg)),
                 mapAnyOf(binaryOperator, cxxOperatorCallExpr)
                     .with(anyOf(
                         allOf(hasOperatorName(IsInc ? "+" : "-"), hasLHS(Arg),
                               hasRHS(
                                   allOf(hasType(isInteger()),
                                         unless(isNegativeIntegerConstant())))),
                         allOf(hasOperatorName(IsInc ? "-" : "+"), hasLHS(Arg),
                               hasRHS(allOf(hasType(isInteger()),
                                            isNegativeIntegerConstant()))))),
                 cxxMemberCallExpr(
                     anyOf(
                         allOf(
                             hasArgument(
                                 0, allOf(hasType(isInteger()),
                                          unless(isNegativeIntegerConstant()))),
                             callee(cxxMethodDecl(
                                 hasName(IsInc ? "operator+" : "operator-")))),
                         allOf(
                             hasArgument(0, allOf(hasType(isInteger()),
                                                  isNegativeIntegerConstant())),
                             callee(cxxMethodDecl(
                                 hasName(IsInc ? "operator-" : "operator+"))))),
                     on(Arg))))
      .bind(IsInc ? UnexpectedInc : UnexpectedDec);
}

void prependStdPrefix(llvm::MutableArrayRef<std::string> Items) {
  static constexpr llvm::StringLiteral Prefix = "::std::";
  llvm::for_each(Items, [](std::string &Item) {
    Item.insert(0, Prefix.data(), Prefix.size());
  });
}

/// Gets functions that a range and an output iterator to the end of another
/// range
static std::vector<std::string>
getSingleRangeWithRevOutputIterator(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"copy_backward"});
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result, std::array{"move_backward"});
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getSingleRangeWithHalfOpen(const LangOptions & /* LangOpts */) {
  std::vector<std::string> Result{{"inner_product"}};
  prependStdPrefix(Result);
  return Result;
}

/// Gets functions that take 2 whole ranges and optionally start with a policy
static std::vector<std::string>
getMultiRangePolicyFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"find_end", "find_first_of", "search",
                                        "includes", "lexicographical_compare"});
  if (LangOpts.CPlusPlus20)
    llvm::append_range(Result, std::array{"lexicographical_compare_three_way"});
  prependStdPrefix(Result);
  return Result;
}

/// Gets a function that takes 2 ranges where the second may be specified by
/// just a start iterator or a start/end pair, The range may optionally start
/// with a policy
static std::vector<std::string> getMultiRangePolicyPotentiallyHalfOpenFunctors(
    const LangOptions & /* LangOpts */) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"mismatch", "equal"});
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getMultiRangePotentiallyHalfOpenFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result, std::array{"is_permutation"});
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string> getMultiRangePolicyWithSingleOutputIterator(
    const LangOptions & /* LangOpts */) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"set_union", "set_intersection",
                                        "set_difference",
                                        "set_symmetric_difference", "merge"});
  prependStdPrefix(Result);
  return Result;
}

// Returns a vector of function that take a range in the first and second
// arguments
static std::vector<std::string>
getSingleRangeFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus17)
    llvm::append_range(Result, std::array{"sample"});
  else
    llvm::append_range(Result, std::array{"random_shuffle"});

  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result,
                       std::array{"shuffle", "partition_point", "iota"});

  llvm::append_range(Result, std::array{
                                 "lower_bound",
                                 "upper_bound",
                                 "equal_range",
                                 "binary_search",
                                 "push_heap",
                                 "pop_heap",
                                 "make_heap",
                                 "sort_heap",
                                 "next_permutation",
                                 "prev_permutation",
                                 "accumulate",
                             });
  prependStdPrefix(Result);
  return Result;
}

// Returns a vector of function that take a range in the first and second or
// second and third arguments
static std::vector<std::string>
getSingleRangePolicyFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result, std::array{
                                   "all_of",
                                   "any_of",
                                   "none_of",
                                   "is_partitioned",
                                   "is_sorted",
                                   "is_sorted_until",
                                   "is_heap",
                                   "is_heap_until",
                                   "minmax_element",
                               });

  if (LangOpts.CPlusPlus17)
    llvm::append_range(Result,
                       std::array{"reduce", "uninitialized_default_construct",
                                  "uninitialized_value_construct", "destroy"});
  if (LangOpts.CPlusPlus20)
    llvm::append_range(Result, std::array{"shift_left", "shift_right"});

  llvm::append_range(Result, std::array{
                                 "find",
                                 "find_if",
                                 "find_if_not",
                                 "adjacent_find",
                                 "count",
                                 "count_if",
                                 "search_n",
                                 "replace",
                                 "replace_if",
                                 "fill",
                                 "generate",
                                 "remove_if",
                                 "unique",
                                 "reverse",
                                 "partition",
                                 "stable_partition",
                                 "sort",
                                 "stable_sort",
                                 "max_element",
                                 "min_element",
                                 "uninitialized_fill",
                             });
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getSingleRangePolicyWithSingleOutputIteratorFunctions(
    const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result,
                       std::array{"copy", "copy_if", "move", "swap_ranges"});
  if (LangOpts.CPlusPlus17)
    llvm::append_range(
        Result, std::array{"exclusive_scan", "inclusive_scan",
                           "transform_reduce", "transform_exclusive_scan",
                           "transform_inclusive_scan", "uninitialized_move"});
  llvm::append_range(Result,
                     std::array{"replace_copy", "replace_copy_if",
                                "remove_copy_if", "unique_copy", "reverse_copy",
                                "adjacent_difference", "uninitialized_copy"

                     });
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getSingleRangePolicyWithTwoOutputIteratorFunctions(
    const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result, std::array{"partition_copy"});
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string> getSingleRangeWithSingleOutputIteratorFunctions(
    const LangOptions & /* LangOpts */) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{
                                 "partial_sum",
                             });
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getTriRangePolicyFunctions(const LangOptions & /* LangOpts */) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"rotate", "nth_element",
                                        "inplace_merge", "partial_sort"});
  prependStdPrefix(Result);
  return Result;
}

void IncorrectIteratorsCheck::registerMatchers(MatchFinder *Finder) {
  NamePairs Forward{NameMatchers{BeginFree, BeginMethod},
                    NameMatchers{EndFree, EndMethod}};
  NamePairs Reverse{NameMatchers{RBeginFree, RBeginMethod},
                    NameMatchers{REndFree, REndMethod}};
  llvm::SmallVector<StringRef, 8> CombinedFreeBegin{
      llvm::iterator_range{llvm::concat<StringRef>(BeginFree, RBeginFree)}};
  llvm::SmallVector<StringRef, 8> CombinedFreeEnd{
      llvm::iterator_range{llvm::concat<StringRef>(EndFree, REndFree)}};
  llvm::SmallVector<StringRef, 8> CombinedMethodBegin{
      llvm::iterator_range{llvm::concat<StringRef>(BeginMethod, RBeginMethod)}};
  llvm::SmallVector<StringRef, 8> CombinedMethodEnd{
      llvm::iterator_range{llvm::concat<StringRef>(EndMethod, REndMethod)}};
  llvm::SmallVector<StringRef, 16> AllFree{llvm::iterator_range{
      llvm::concat<StringRef>(CombinedFreeBegin, CombinedFreeEnd)}};
  llvm::SmallVector<StringRef, 16> AllMethod{llvm::iterator_range{
      llvm::concat<StringRef>(CombinedMethodBegin, CombinedMethodEnd)}};
  NamePairs Combined{NameMatchers{CombinedFreeBegin, CombinedMethodBegin},
                     NameMatchers{CombinedFreeEnd, CombinedMethodEnd}};
  FullState State{
      Forward, Reverse, Combined, {AllFree, AllMethod}, MakeReverseIterator};
  auto SingleRange = getSingleRangeFunctors(getLangOpts());
  auto SingleRangePolicy = getSingleRangePolicyFunctors(getLangOpts());
  auto SingleRangePolicyOutput =
      getSingleRangePolicyWithSingleOutputIteratorFunctions(getLangOpts());
  auto SingleRangePolicyTwoOutput =
      getSingleRangePolicyWithTwoOutputIteratorFunctions(getLangOpts());
  auto SingleRangeWithOutput =
      getSingleRangeWithSingleOutputIteratorFunctions(getLangOpts());
  auto SingleRangeHalfOpen = getSingleRangeWithHalfOpen(getLangOpts());
  auto SingleRangeBackwardsHalf =
      getSingleRangeWithRevOutputIterator(getLangOpts());
  auto MultiRangePolicy = getMultiRangePolicyFunctors(getLangOpts());
  auto MultiRangePolicyPotentiallyHalfOpen =
      getMultiRangePolicyPotentiallyHalfOpenFunctors(getLangOpts());
  auto MultiRangePotentiallyHalfOpen =
      getMultiRangePotentiallyHalfOpenFunctors(getLangOpts());
  auto MultiRangePolicySingleOutputIterator =
      getMultiRangePolicyWithSingleOutputIterator(getLangOpts());

  auto TriRangePolicy = getTriRangePolicyFunctions(getLangOpts());

  std::vector<std::string> TriRangePolicyOutput = {{"::std::rotate_copy"}};

  // Doesn't really fit into any of the other categories.
  // Can be ran under a polict and takes a full range with an output iterator,
  // or a full range, range begin and output iterator.
  std::vector<std::string> Transform = {{"::std::transform"}};

  static const auto ToRefs =
      [](std::initializer_list<std::reference_wrapper<std::vector<std::string>>>
             Items) {
        std::vector<StringRef> Result;
        for (const auto &Item : Items)
          llvm::append_range(Result, Item.get());
        return Result;
      };

  Finder->addMatcher(
      makePairRangeMatcher(
          ToRefs({SingleRange, SingleRangeWithOutput, SingleRangeHalfOpen,
                  SingleRangeBackwardsHalf, MultiRangePotentiallyHalfOpen

          }),
          0, 1, State),
      this);

  Finder->addMatcher(
      runWithPolicy2(ToRefs({SingleRangePolicy, SingleRangePolicyOutput,
                             SingleRangePolicyTwoOutput, MultiRangePolicy,
                             MultiRangePolicyPotentiallyHalfOpen, Transform,
                             MultiRangePolicySingleOutputIterator}),
                     0, 1, makePairRangeMatcher, State),
      this);

  Finder->addMatcher(
      makeExpectedBeginFullMatcher(ToRefs({SingleRangeHalfOpen}), 2, State),
      this);

  Finder->addMatcher(
      makeExpectedOutputFullMatcher(ToRefs({SingleRangeWithOutput}), 2, State),
      this);
  Finder->addMatcher(runWithPolicy1(ToRefs({SingleRangePolicyOutput,
                                            SingleRangePolicyTwoOutput}),
                                    2, makeExpectedOutputFullMatcher, State),
                     this);
  Finder->addMatcher(
      runWithPolicy1(ToRefs({SingleRangePolicyTwoOutput, TriRangePolicyOutput}),
                     3, makeExpectedOutputFullMatcher, State),
      this);
  Finder->addMatcher(
      makeExpectedEndFullMatcher(ToRefs({SingleRangeBackwardsHalf}), 2, State),
      this);
  Finder->addMatcher(runWithPolicy2(ToRefs({MultiRangePolicy}), 2, 3,
                                    makePairRangeMatcher, State),
                     this);
  Finder->addMatcher(
      makeHalfOpenMatcher(ToRefs({MultiRangePotentiallyHalfOpen}), 2, 3, State),
      this);
  Finder->addMatcher(
      runWithPolicy2(ToRefs({MultiRangePolicyPotentiallyHalfOpen,
                             MultiRangePolicySingleOutputIterator}),
                     2, 3, makeHalfOpenMatcher, State),
      this);

  Finder->addMatcher(
      runWithPolicy1(ToRefs({MultiRangePolicySingleOutputIterator}), 4,
                     makeExpectedOutputFullMatcher, State),
      this);

  Finder->addMatcher(makeTransformArgsMatcher(ToRefs({Transform}), State),
                     this);

  for (bool IsInc : {true, false})
    Finder->addMatcher(makeUnexpectedIncDecMatcher(State, IsInc), this);

  Finder->addMatcher(
      runWithPolicy3(ToRefs({TriRangePolicy, TriRangePolicyOutput}), 0, 1, 2,
                     &make3RangeMatcherInternal, State),
      this);

  registerContainerDescriptor(
      Finder, this, {{"::std::vector"}, {"::std::list"}, {"::std::deque"}},
      Descriptor{
          {
              {"insert", {{1, 2}}},
              {"assign", {{0, 1}}},
          },
          {{
              {"insert", {{0}}},
              {"emplace", {{0, static_cast<bool>(getLangOpts().CPlusPlus11)}}},
              {"erase", {{0}, {1}}},
              {"insert_range",
               {{0, static_cast<bool>(getLangOpts().CPlusPlus23)}}},
          }},
          {{0, 1}}},
      State);

  registerContainerDescriptor(
      Finder, this,
      {{"::std:::forward_list", static_cast<bool>(getLangOpts().CPlusPlus11)}},
      Descriptor{{
                     {"insert_after", {{1, 2}}},
                     {"assign", {{0, 1}}},

                     // FIXME: We should be enforcing that the container arg
                     // for these is the same as the second arg for the call
                     {"splice_after", {{2, 3}}},
                 },
                 {{
                     {"insert_after", {{0}}},
                     {"emplace_after", {{0}}},
                     {"erase_after", {{0}, {1}}},
                     {"splice_after", {{0}}},
                     {"insert_range_after",
                      {{0, static_cast<bool>(getLangOpts().CPlusPlus23)}}},
                 }},
                 {{0, 1}}},
      State);

  registerContainerDescriptor(
      Finder, this, {{"::std::basic_string"}},
      Descriptor{{
                     {"insert", {{1, 2}}},
                     {"append", {{0, 1}}},
                     {"assign", {{0, 1}}},
                 },
                 {{
                     {"insert", {{0}}},
                     {"erase", {{0}, {1}}},
                     {"insert_range",
                      {{0, static_cast<bool>(getLangOpts().CPlusPlus23)}}},
                 }},
                 {{0, 1}}},
      State);

  registerContainerDescriptor(
      Finder, this,
      {{"::std::set"},
       {"::std::multiset"},
       {"::std::unordered_set", static_cast<bool>(getLangOpts().CPlusPlus11)},
       {"::std::unordered_multiset",
        static_cast<bool>(getLangOpts().CPlusPlus11)},
       {"::std::multimap"},
       {"::std::unordered_multimap",
        static_cast<bool>(getLangOpts().CPlusPlus11)}},
      {{{"insert", {{0, 1}}}},
       {{"insert", {{0}}},
        {"emplace_hint", {{0, static_cast<bool>(getLangOpts().CPlusPlus11)}}},
        {"erase", {{0}, {1}}},
        {"extract", {{0, static_cast<bool>(getLangOpts().CPlusPlus17)}}}},
       {{0, 1}}},
      State);

  registerContainerDescriptor(
      Finder, this,
      {{"::std::map"},
       {"::std::unordered_map", static_cast<bool>(getLangOpts().CPlusPlus11)}},
      {{{"insert", {{0, 1}}}},
       {{"insert", {{0}}},
        {"insert_or_assign",
         {{0, static_cast<bool>(getLangOpts().CPlusPlus17)}}},
        {"emplace_hint", {{0, static_cast<bool>(getLangOpts().CPlusPlus11)}}},
        {"try_emplace", {{0, static_cast<bool>(getLangOpts().CPlusPlus17)}}},
        {"erase", {{0}, {1}}},
        {"extract", {{0, static_cast<bool>(getLangOpts().CPlusPlus17)}}}},
       {{0, 1}}},
      State);

  registerContainerDescriptor(
      Finder, this,
      {{"::std::stack"}, {"::std::queue"}, {"::std::priority_queue"}},
      {{}, {}, {{0, 1}}}, State);
}

static constexpr char MismatchedRangeNote[] =
    "%select{|different }0range passed as the %select{begin|end}0 iterator";

void IncorrectIteratorsCheck::check(const MatchFinder::MatchResult &Result) {
  auto AddRevNote = [&Result, this](bool IsBegin) {
    if (const auto *Rev =
            Result.Nodes.getNodeAs<CallExpr>(ArgMismatchRevBind)) {
      diag(Rev->getBeginLoc(),
           "%0 changes '%select{end|begin}1' into %select{a 'begin|an 'end}1' "
           "iterator",
           DiagnosticIDs::Note)
          << Rev->getSourceRange() << Rev->getDirectCallee() << IsBegin;
    }
  };

  for (auto [Name, Idx] :
       {std::pair{ArgMismatchBegin, 1}, std::pair{ArgMismatchEnd, 0},
        std::pair{OutputRangeBegin, 0}, std::pair{OutputRangeEnd, 2}}) {
    if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Name)) {
      diag(Node->getBeginLoc(),
           "'%select{begin|end|end}0' iterator supplied where %select{an "
           "'end'|a 'begin'|an output}0 iterator is expected")
          << Idx
          << Result.Nodes.getNodeAs<Expr>(ArgMismatchExpr)->getSourceRange();
      AddRevNote(Idx != 0);
      return;
    }
  }

  if (const auto *InternalArg =
          Result.Nodes.getNodeAs<Expr>(InternalArgument)) {
    diag(InternalArg->getBeginLoc(),
         "%0 called with an iterator for a different container")
        << Result.Nodes.getNodeAs<CallExpr>(Callee)->getDirectCallee();
    const auto *Object = Result.Nodes.getNodeAs<Expr>(Internal);
    diag(Object->getBeginLoc(), "container is specified here",
         DiagnosticIDs::Note)
        << Object->getSourceRange();
    const auto *Other = Result.Nodes.getNodeAs<Expr>(InternalOther);
    diag(Other->getBeginLoc(), "different container provided here",
         DiagnosticIDs::Note)
        << Other->getSourceRange();
    return;
  }

  if (const auto *Range1 = Result.Nodes.getNodeAs<Expr>(FirstRangeArg)) {
    const auto *Range2 = Result.Nodes.getNodeAs<Expr>(SecondRangeArg);
    const auto *FullRange1 = Result.Nodes.getNodeAs<Expr>(FirstRangeArgExpr);
    const auto *FullRange2 = Result.Nodes.getNodeAs<Expr>(SecondRangeArgExpr);
    assert(Range1 && Range2 && FullRange1 && FullRange2 && "Unexpected match");
    const auto *Call = Result.Nodes.getNodeAs<Expr>(Callee);
    const auto Func = isa<CallExpr>(Call) ? cast<CallExpr>(Call)
                                                ->getDirectCallee()
                                                ->getQualifiedNameAsString()
                                          : cast<CXXConstructExpr>(Call)
                                                ->getConstructor()
                                                ->getQualifiedNameAsString();
    diag(Call->getBeginLoc(), "mismatched ranges supplied to '%0'")
        << Func << FullRange1->getSourceRange() << FullRange2->getSourceRange();
    diag(Range1->getBeginLoc(), MismatchedRangeNote, DiagnosticIDs::Note)
        << false << FullRange1->getSourceRange();
    diag(Range2->getBeginLoc(), MismatchedRangeNote, DiagnosticIDs::Note)
        << true << FullRange2->getSourceRange();
    return;
  }

  for (auto [Name, IsInc] :
       {std::pair{UnexpectedInc, true}, std::pair{UnexpectedDec, false}}) {
    if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Name)) {
      diag(Node->getExprLoc(), "trying to %select{decrement before the "
                               "start|increment past the end}0 of a range")
          << IsInc
          << Result.Nodes.getNodeAs<Expr>(UnexpectedIncDecExpr)
                 ->getSourceRange();
      AddRevNote(IsInc);
      return;
    }
  }
  for (auto [Name, IsBegin] :
       {std::pair{TriRangeArgBegin, true}, std::pair{TriRangeArgEnd, false}}) {
    if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Name)) {
      diag(Node->getBeginLoc(),
           "'%select{end|begin}0' iterator passed as the middle iterator")
          << IsBegin << Node->getSourceRange();
      AddRevNote(IsBegin);
      return;
    }
  }
  llvm_unreachable("Unhandled matches");
}

IncorrectIteratorsCheck::IncorrectIteratorsCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      BeginFree(utils::options::parseStringList(
          Options.get("BeginFree", "::std::begin;::std::cbegin"))),
      EndFree(utils::options::parseStringList(
          Options.get("EndFree", "::std::end;::std::cend"))),
      BeginMethod(utils::options::parseStringList(
          Options.get("BeginMethod", "begin;cbegin"))),
      EndMethod(utils::options::parseStringList(
          Options.get("EndMethod", "end;cend"))),
      RBeginFree(utils::options::parseStringList(
          Options.get("RBeginFree", "::std::rbegin;::std::crbegin"))),
      REndFree(utils::options::parseStringList(
          Options.get("REndFree", "::std::rend;::std::crend"))),
      RBeginMethod(utils::options::parseStringList(
          Options.get("RBeginMethod", "rbegin;crbegin"))),
      REndMethod(utils::options::parseStringList(
          Options.get("REndMethod", "rend;crend"))),
      MakeReverseIterator(utils::options::parseStringList(
          Options.get("MakeReverseIterator", "::std::make_reverse_iterator"))) {
}

void IncorrectIteratorsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "BeginFree",
                utils::options::serializeStringList(BeginFree));
  Options.store(Opts, "EndFree", utils::options::serializeStringList(EndFree));
  Options.store(Opts, "BeginMethod",
                utils::options::serializeStringList(BeginMethod));
  Options.store(Opts, "EndMethod",
                utils::options::serializeStringList(EndMethod));
  Options.store(Opts, "RBeginFree",
                utils::options::serializeStringList(RBeginFree));
  Options.store(Opts, "REndFree",
                utils::options::serializeStringList(REndFree));
  Options.store(Opts, "RBeginMethod",
                utils::options::serializeStringList(RBeginMethod));
  Options.store(Opts, "REndMethod",
                utils::options::serializeStringList(REndMethod));
  Options.store(Opts, "MakeReverseIterator",
                utils::options::serializeStringList(MakeReverseIterator));
}

std::optional<TraversalKind>
IncorrectIteratorsCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}

bool IncorrectIteratorsCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

} // namespace clang::tidy::bugprone
