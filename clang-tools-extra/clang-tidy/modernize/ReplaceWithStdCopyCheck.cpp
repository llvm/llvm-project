//===--- ReplaceWithStdCopyCheck.cpp - clang-tidy----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceWithStdCopyCheck.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringRef.h"
#include <variant>

using namespace clang;
using namespace clang::ast_matchers;

namespace clang::tidy {

template <>
struct OptionEnumMapping<
    enum modernize::ReplaceWithStdCopyCheck::FlaggableCallees> {
  static llvm::ArrayRef<std::pair<
      modernize::ReplaceWithStdCopyCheck::FlaggableCallees, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        modernize::ReplaceWithStdCopyCheck::FlaggableCallees, StringRef>
        Mapping[] = {
            {modernize::ReplaceWithStdCopyCheck::FlaggableCallees::
                 MemmoveAndMemcpy,
             "MemmoveAndMemcpy"},
            {modernize::ReplaceWithStdCopyCheck::FlaggableCallees::OnlyMemmove,
             "OnlyMemmove"},
        };
    return {Mapping};
  }
};

template <>
struct OptionEnumMapping<
    enum modernize::ReplaceWithStdCopyCheck::StdCallsReplacementStyle> {
  static llvm::ArrayRef<std::pair<
      modernize::ReplaceWithStdCopyCheck::StdCallsReplacementStyle, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        modernize::ReplaceWithStdCopyCheck::StdCallsReplacementStyle, StringRef>
        Mapping[] = {
            {modernize::ReplaceWithStdCopyCheck::StdCallsReplacementStyle::
                 FreeFunc,
             "FreeFunc"},
            {modernize::ReplaceWithStdCopyCheck::StdCallsReplacementStyle::
                 MemberCall,
             "MemberCall"},
        };
    return {Mapping};
  }
};

namespace modernize {
namespace {
namespace arg {
namespace tag {
struct VariantPtrArgRef {
  llvm::StringLiteral AsContainer;
  llvm::StringLiteral AsCArray;
  llvm::StringLiteral Fallback;
};

struct PtrArgRefs {
  VariantPtrArgRef VariantRefs;
  llvm::StringLiteral CharType;
};

struct DestArg {
  static constexpr PtrArgRefs Refs = {
      {"DestArg::AsContainer", "DestArg::AsCArray", "DestArg::Fallback"},
      "DestArg::CharType"};
};
struct SourceArg {
  static constexpr PtrArgRefs Refs = {
      {"SourceArg::AsContainer", "SourceArg::AsCArray", "SourceArg::Fallback"},
      "SourceArg::CharType"

  };
};
struct SizeArg {};
} // namespace tag

template <bool IsConst> auto createPtrArgMatcher(tag::PtrArgRefs Refs) {
  auto [VariantRefs, ValueTypeRef] = Refs;
  auto [AsContainer, AsCArray] = VariantRefs;

  auto AllowedContainerNamesM = []() {
    if constexpr (IsConst) {
      return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
                        "::std::vector", "::std::basic_string");
    } else {
      return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
                        "::std::vector", "::std::basic_string",
                        "::std::basic_string_view", "::std::span");
    }
  }();

  auto ValueTypeM = type().bind(ValueTypeRef);

  auto AllowedContainerTypeM = hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(recordDecl(classTemplateSpecializationDecl(
          AllowedContainerNamesM,
          hasTemplateArgument(
              0, templateArgument(refersToType(
                     hasUnqualifiedDesugaredType(ValueTypeM)))))))));

  auto VariantContainerM =
      expr(hasType(AllowedContainerTypeM)).bind(AsContainer);
  auto VariantCArrayM =
      expr(hasType(arrayType(hasElementType(ValueTypeM)))).bind(AsCArray);

  auto StdDataReturnM = returns(pointerType(pointee(
      hasUnqualifiedDesugaredType(equalsBoundNode(ValueTypeRef.str())))));

  auto StdDataMemberDeclM =
      cxxMethodDecl(hasName("data"), parameterCountIs(0), StdDataReturnM);
  // ,unless(isConst()) // __jm__ ONLY difference between Dest
  // and Source calls, but maybe don't include it?

  auto StdDataFreeDeclM = functionDecl(
      hasName("::std::data"), parameterCountIs(1),
      StdDataReturnM); // __jm__ possibly elaborate on argument type here?

  auto StdDataMemberCallM = cxxMemberCallExpr(
      callee(StdDataMemberDeclM), argumentCountIs(0),
      on(expr(hasType(AllowedContainerTypeM)).bind(AsContainer)));

  auto ArrayOrContainerM = expr(anyOf(VariantCArrayM, VariantContainerM));

  auto StdDataFreeCallM = callExpr(callee(StdDataFreeDeclM), argumentCountIs(1),
                                   hasArgument(0, ArrayOrContainerM));

  return expr(anyOf(StdDataMemberCallM, StdDataFreeCallM, VariantCArrayM));
}

template <typename TAG> class MatcherLinker : public TAG {
public:
  static auto createMatcher();
  static void handleResult(const MatchFinder::MatchResult &Result);

private:
  using AllowedTAG = std::enable_if_t<std::is_same_v<TAG, tag::DestArg> ||
                                      std::is_same_v<TAG, tag::SourceArg> ||
                                      std::is_same_v<TAG, tag::SizeArg>>;
};

template <> auto MatcherLinker<tag::DestArg>::createMatcher() {
  return createPtrArgMatcher<false>(Refs);
}

template <> auto MatcherLinker<tag::SourceArg>::createMatcher() {
  return createPtrArgMatcher<true>(Refs);
}

template <> auto MatcherLinker<tag::SizeArg>::createMatcher() {
  // cases:
  // 1. call to std::size
  // 2. member call to .size()
  // 3. sizeof node or sizeof(node) (c-array)
  // 4. N * sizeof(type) (hints at sequence-related use, however none singled out)

  // need a robust way of exchanging information between matchers that will help in sequence identification
  // I see one possiblity
  // size "thinks" dest is rawptr with pointee sequence
  //   N * sizeof
  // size "thinks" src  is rawptr with pointee sequence
  // size agrees dest is container/c-array
  // size agrees src  is container/c-array

  return anything(); // __jm__ TODO
}

using Dest = MatcherLinker<tag::DestArg>;
using Source = MatcherLinker<tag::SourceArg>;
using Size = MatcherLinker<tag::SizeArg>;
} // namespace arg

constexpr llvm::StringLiteral ExpressionRef = "::std::memcpy";
constexpr llvm::StringLiteral StdCopyHeader = "<algorithm>";
constexpr llvm::StringLiteral ReturnValueDiscardedRef =
    "ReturnValueDiscardedRef";

// Helper Matcher which applies the given QualType Matcher either directly or by
// resolving a pointer type to its pointee. Used to match v.push_back() as well
// as p->push_back().
auto hasTypeOrPointeeType(
    const ast_matchers::internal::Matcher<QualType> &TypeMatcher) {
  return anyOf(hasType(TypeMatcher),
               hasType(pointerType(pointee(TypeMatcher))));
}

} // namespace

ReplaceWithStdCopyCheck::ReplaceWithStdCopyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      FlaggableCallees_(Options.getLocalOrGlobal("FlaggableCallees",
                                                 FlaggableCalleesDefault)),
      StdCallsReplacementStyle_(Options.getLocalOrGlobal(
          "StdCallsReplacementStyle", StdCallsReplacementStyleDefault)) {}

void ReplaceWithStdCopyCheck::registerMatchers(MatchFinder *Finder) {
  const auto ReturnValueUsedM =
      hasParent(compoundStmt().bind(ReturnValueDiscardedRef));

  const auto OffendingDeclM =
      functionDecl(parameterCountIs(3), hasAnyName(getFlaggableCallees()));

  // cases:
  // 1. One of the arguments is definitely a sequence and the other a pointer -
  // match
  // 2. Both source and dest are pointers, but size is of the form ((N :=
  // expr()) * sizeof(bytelike())) - match (false positive if N \in {0, 1})

  using namespace arg;
  const auto Expression =
      callExpr(callee(OffendingDeclM),
               anyOf(optionally(ReturnValueUsedM),
                     allOf(hasArgument(0, Dest::createMatcher()),
                           hasArgument(1, Source::createMatcher()),
                           hasArgument(2, Size::createMatcher()))));
  Finder->addMatcher(Expression.bind(ExpressionRef), this);
}

void ReplaceWithStdCopyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void ReplaceWithStdCopyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "FlaggableCallees", FlaggableCallees_);
  Options.store(Opts, "StdCallsReplacementStyle", StdCallsReplacementStyle_);
}

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "clang-tidy"

void ReplaceWithStdCopyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &CallNode = *Result.Nodes.getNodeAs<CallExpr>(ExpressionRef);

  auto Diag = diag(CallNode.getExprLoc(), "prefer std::copy_n to %0")
              << cast<NamedDecl>(CallNode.getCalleeDecl());

  tryIssueFixIt(Result, Diag, CallNode);

  {
    // using namespace std::literals;
    // Diag << FixItHint::CreateReplacement(
    //     DestArg->getSourceRange(),
    //     ("std::begin(" +
    //      tooling::fixit::getText(SrcArg->getSourceRange(), *Result.Context) +
    //      ")")
    //         .str());
    // Diag << FixItHint::CreateReplacement(
    //     SrcArg->getSourceRange(),
    //     tooling::fixit::getText(SizeArg->getSourceRange(), *Result.Context));
    // Diag << FixItHint::CreateReplacement(
    //     SizeArg->getSourceRange(),
    //     ("std::begin(" +
    //      tooling::fixit::getText(DestArg->getSourceRange(), *Result.Context)
    //      +
    //      ")")
    //         .str());
  }
  // dest source size -> source size dest
}

// void ReplaceWithStdCopyCheck::handleMemcpy(const CallExpr &Node) {

//   // FixIt
//   const CharSourceRange FunctionNameSourceRange =
//   CharSourceRange::getCharRange(
//       Node.getBeginLoc(), Node.getArg(0)->getBeginLoc());

//   Diag << FixItHint::CreateReplacement(FunctionNameSourceRange,
//   "std::copy_n(");
// }

// bool ReplaceWithStdCopyCheck::fixItPossible(
//     const MatchFinder::MatchResult &Result) {}

void ReplaceWithStdCopyCheck::tryIssueFixIt(
    const MatchFinder::MatchResult &Result, const DiagnosticBuilder &Diag,
    const CallExpr &CallNode) {
  // don't issue a fixit if the result of the call is used
  if (bool IsReturnValueUsed =
          Result.Nodes.getNodeAs<Stmt>(ReturnValueDiscardedRef) == nullptr;
      IsReturnValueUsed)
    return;

  // to make sure we can issue a FixIt, need to be pretty sure we're dealing
  // with a sequence and it is a byte sequence
  // 1. For now we are sure both source and dest are sequences, but not
  // necessarily of bytes
  // 2. We can relax conditions to where just one arg is a sequence, and the
  // other can then be a sequence or raw pointer
  // 3. when both dest and source are pointers (no expectations on the argument
  // as everything required is enforced by the type system) we can use a
  // heuristic using the form of the 3rd argument expression
  //

  // delete the memmove/memcpy
  // insert an std::copy_n
  struct CArrayTag {};
  struct ContainerTag {};
  struct RawPtrTag {};
  using PtrArgVariant = std::variant<CArrayTag, ContainerTag, RawPtrTag>;
  struct PtrArg {
    PtrArgVariant Tag;
    const Expr *Node;
  };

  const auto MakePtrArg = [&](arg::tag::VariantPtrArgRef Refs) -> PtrArg {
    if (const auto *Node =
            Result.Nodes.getNodeAs<Expr>(arg::Dest::Refs.VariantRefs.AsCArray);
        Node != nullptr) {
      // freestanding std::begin
      return {CArrayTag{}, Node};
    }
    if (const auto *Node = Result.Nodes.getNodeAs<Expr>(
            arg::Dest::Refs.VariantRefs.AsContainer);
        Node != nullptr) {
      return {ContainerTag{}, Node};
    }
    const auto *Node =
        Result.Nodes.getNodeAs<Expr>(arg::Dest::Refs.VariantRefs.Fallback);
    return {RawPtrTag{}, Node};
  };

  auto Dest = MakePtrArg(arg::Dest::Refs.VariantRefs);
  auto Source = MakePtrArg(arg::Source::Refs.VariantRefs);

  if (std::holds_alternative<RawPtrTag>(Dest.Tag) and
      std::holds_alternative<RawPtrTag>(Source.Tag) and CheckSizeArgPermitsFix()) {

  } else {
    
  }

    Diag << Inserter.createIncludeInsertion(
        Result.SourceManager->getFileID(CallNode.getBeginLoc()), StdCopyHeader);
}

void ReplaceWithStdCopyCheck::reorderArgs(DiagnosticBuilder &Diag,
                                          const CallExpr *MemcpyNode) {
  std::array<std::string, 3> Arg;

  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  PrintingPolicy Policy(LangOpts);

  // Retrieve all the arguments
  for (uint8_t I = 0; I < Arg.size(); I++) {
    llvm::raw_string_ostream S(Arg[I]);
    MemcpyNode->getArg(I)->printPretty(S, nullptr, Policy);
  }

  // Create lambda that return SourceRange of an argument
  auto GetSourceRange = [MemcpyNode](uint8_t ArgCount) -> SourceRange {
    return SourceRange(MemcpyNode->getArg(ArgCount)->getBeginLoc(),
                       MemcpyNode->getArg(ArgCount)->getEndLoc());
  };

  // Reorder the arguments
  Diag << FixItHint::CreateReplacement(GetSourceRange(0), Arg[1]);

  Arg[2] = Arg[1] + " + ((" + Arg[2] + ") / sizeof(*(" + Arg[1] + ")))";
  Diag << FixItHint::CreateReplacement(GetSourceRange(1), Arg[2]);

  Diag << FixItHint::CreateReplacement(GetSourceRange(2), Arg[0]);
}

llvm::ArrayRef<StringRef> ReplaceWithStdCopyCheck::getFlaggableCallees() const {
  switch (FlaggableCallees_) {
  case FlaggableCallees::OnlyMemmove:
    return {"::memmove", "::std::memmove"};
  case FlaggableCallees::MemmoveAndMemcpy:
    return {"::memmove", "::std::memmove", "::memcpy", "::std::memcpy"};
  }
}
} // namespace modernize
} // namespace clang::tidy