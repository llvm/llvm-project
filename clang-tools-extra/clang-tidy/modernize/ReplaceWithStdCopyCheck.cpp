//===--- ReplaceWithStdCopyCheck.cpp - clang-tidy----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceWithStdCopyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
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

namespace modernize {
namespace {

constexpr llvm::StringLiteral ExpressionRef = "::std::memcpy";
constexpr llvm::StringLiteral StdCopyHeader = "<algorithm>";
constexpr llvm::StringLiteral ReturnValueDiscardedRef =
    "ReturnValueDiscardedRef";

namespace ptrarg {

struct Refs {
  llvm::StringLiteral AsContainer;
  llvm::StringLiteral AsCArray;
  size_t FallbackParameterIdx;
  llvm::StringLiteral ValueType;
};

template <typename RefsT> auto createPtrArgMatcher() {
  constexpr Refs Refs = RefsT::Refs;

  auto AllowedContainerNamesM = []() {
    if constexpr (true)
      return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
                        "::std::vector", "::std::basic_string");
    else
      return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
                        "::std::vector", "::std::basic_string",
                        "::std::basic_string_view", "::std::span");
  }();

  auto ValueTypeM = type().bind(Refs.ValueType);

  auto AllowedContainerTypeM = hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(recordDecl(classTemplateSpecializationDecl(
          AllowedContainerNamesM,
          hasTemplateArgument(
              0, templateArgument(refersToType(
                     hasUnqualifiedDesugaredType(ValueTypeM)))))))));

  auto VariantContainerM =
      expr(hasType(AllowedContainerTypeM)).bind(Refs.AsContainer);
  auto VariantCArrayM =
      expr(hasType(arrayType(hasElementType(ValueTypeM)))).bind(Refs.AsCArray);

  auto StdDataReturnM = returns(pointerType(pointee(
      hasUnqualifiedDesugaredType(equalsBoundNode(Refs.ValueType.str())))));

  auto StdDataMemberDeclM =
      cxxMethodDecl(hasName("data"), parameterCountIs(0), StdDataReturnM);
  // ,unless(isConst()) // __jm__ ONLY difference between Dest
  // and Source calls, but maybe don't include it?

  auto StdDataFreeDeclM = functionDecl(
      hasName("::std::data"), parameterCountIs(1),
      StdDataReturnM); // __jm__ possibly elaborate on argument type here?

  auto StdDataMemberCallM = cxxMemberCallExpr(
      callee(StdDataMemberDeclM), argumentCountIs(0),
      on(expr(hasType(AllowedContainerTypeM)).bind(Refs.AsContainer)));

  auto ArrayOrContainerM = expr(anyOf(VariantCArrayM, VariantContainerM));

  auto StdDataFreeCallM = callExpr(callee(StdDataFreeDeclM), argumentCountIs(1),
                                   hasArgument(0, ArrayOrContainerM));

  // the last expr() in anyOf assumes previous matchers are ran eagerly from
  // left to right, still need to test this is the actual behaviour
  return expr(
      anyOf(StdDataMemberCallM, StdDataFreeCallM, VariantCArrayM, expr()));
}

namespace tag {
struct Container {};
struct CArray {};
struct RawPtr {};
} // namespace tag
struct PtrArg {
  std::variant<tag::Container, tag::CArray, tag::RawPtr> Tag;
  const Expr *Node;
};

template <typename RefT>
auto extractNode(const CallExpr &CallNode,
                 const MatchFinder::MatchResult &Result) -> PtrArg {
  constexpr Refs Refs = RefT::Refs;
  if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Refs.AsCArray))
    return {tag::CArray{}, Node};
  if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Refs.AsContainer))
    return {tag::Container{}, Node};
  return {tag::RawPtr{}, CallNode.getArg(Refs.FallbackParameterIdx)};
}

template <typename RefT>
QualType extractValueType(const MatchFinder::MatchResult &Result) {
  return *Result.Nodes.getNodeAs<QualType>(RefT::Refs.ValueType);
}
} // namespace ptrarg

namespace dst {
constexpr size_t argIndex = 0;
struct RefT {
  static constexpr ptrarg::Refs Refs = {
      "Dst::AsContainer",
      "Dst::AsCArray",
      0,
      "Dst::ValueType",
  };
};

auto createMatcher() { return ptrarg::createPtrArgMatcher<RefT>(); }
auto extractNode(const CallExpr &CallNode,
                 const MatchFinder::MatchResult &Result) {
  return ptrarg::extractNode<RefT>(CallNode, Result);
}
auto extractValueType(const MatchFinder::MatchResult &Result) {
  return ptrarg::extractValueType<RefT>(Result);
}
} // namespace dst

namespace src {
constexpr size_t argIndex = 1;

struct SrcRefsT {
  static constexpr ptrarg::Refs Refs = {
      "Src::AsContainer",
      "Src::AsCArray",
      1,
      "Src::ValueType",
  };
};

auto createMatcher() { return ptrarg::createPtrArgMatcher<SrcRefsT>(); }
auto extractNode(const CallExpr &CallNode,
                 const MatchFinder::MatchResult &Result) {
  return ptrarg::extractNode<SrcRefsT>(CallNode, Result);
}
auto extractValueType(const MatchFinder::MatchResult &Result) {
  return ptrarg::extractValueType<SrcRefsT>(Result);
}
} // namespace src

namespace size {
constexpr size_t argIndex = 2;
// cases:
// 1. sizeof(T) where T is a type
// 2. sizeof(rec) where rec is a CArray
// 3. N * sizeof(T)
// 4. strlen(rec) [+ 1]
// 5. other expr
namespace variant {
struct SizeOfExpr {
  const Expr *Arg;
};
struct NSizeOfExpr {
  const Expr *N;
  const Expr *Arg;
};
struct Strlen {
  const Expr *Arg;
};
} // namespace variant
using SizeArg = std::variant<variant::SizeOfExpr, variant::NSizeOfExpr,
                             variant::Strlen, const Expr *>;

struct SizeComponents {
  CharUnits Unit;
  std::string NExpr;
};

struct IArg {
  virtual ~IArg() = default;
  virtual SizeComponents extractSizeComponents() = 0;
};

static constexpr struct Refs {
  llvm::StringLiteral SizeOfArg;
  llvm::StringLiteral N;
  llvm::StringLiteral StrlenArg;
} Refs = {
    "Size::SizeOfExpr",
    "Size::N",
    "Size::StrlenArg",
};

auto createMatcher() {
  auto NSizeOfT =
      binaryOperator(hasOperatorName("*"),
                     hasOperands(expr().bind(Refs.N),
                                 sizeOfExpr(has(expr().bind(Refs.SizeOfArg)))));
  auto Strlen = callExpr(callee(functionDecl(hasAnyName(
                             "::strlen", "::std::strlen", "::wcslen",
                             "::std::wcslen", "::strnlen_s", "::std::strnlen_s",
                             "::wcsnlen_s", "::std::wcsnlen_s"))),
                         hasArgument(0, expr().bind(Refs.StrlenArg)));
  auto StrlenPlusOne = binaryOperator(
      hasOperatorName("+"), hasOperands(Strlen, integerLiteral(equals(1))));

  auto SizeOfExpr = sizeOfExpr(hasArgumentOfType(type().bind(Refs.SizeOfArg)));

  return expr(anyOf(NSizeOfT, Strlen, StrlenPlusOne, SizeOfExpr));
}

SizeArg extractNode(const CallExpr &CallNode,
                    const MatchFinder::MatchResult &Result) {
  if (const auto *SizeOfArgNode = Result.Nodes.getNodeAs<Expr>(Refs.SizeOfArg);
      SizeOfArgNode != nullptr) {
    if (const auto *NNode = Result.Nodes.getNodeAs<Expr>(Refs.N);
        NNode != nullptr)
      return variant::NSizeOfExpr{NNode, SizeOfArgNode};
    return variant::SizeOfExpr{SizeOfArgNode};
  }
  if (const auto *StrlenArgNode = Result.Nodes.getNodeAs<Expr>(Refs.StrlenArg);
      StrlenArgNode != nullptr) {
    return variant::Strlen{StrlenArgNode};
  }
  return CallNode.getArg(2);
}
// 1. N * sizeof(type) (commutative) - issue warning but no fixit, or fixit
// only if both src/dest are fixit friendly 1.1. This might allow fixits for
// wider-than-byte element collections
// 2. strlen(src|dest) ?(+ 1 (commutative)) -- issue warning but no fixit
// 4. sizeof(variable) only if that variable is of arrayType, if it's of
// ptrType that may indicate [de]serialization

} // namespace size

} // namespace

ReplaceWithStdCopyCheck::ReplaceWithStdCopyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      FlaggableCallees_(Options.getLocalOrGlobal("FlaggableCallees",
                                                 FlaggableCalleesDefault)) {}

void ReplaceWithStdCopyCheck::registerMatchers(MatchFinder *Finder) {
  const auto ReturnValueUsedM =
      hasParent(compoundStmt().bind(ReturnValueDiscardedRef));

  const auto OffendingDeclM =
      functionDecl(parameterCountIs(3), hasAnyName(getFlaggableCallees()));

  // cases:
  // 1. One of the arguments is definitely a collection and the other a pointer
  // - match
  // 2. Both source and dest are pointers, but size is of the form ((N :=
  // expr()) * sizeof(bytelike())) - match (false positive if N \in {0, 1})

  const auto Expression =
      callExpr(callee(OffendingDeclM),
               anyOf(optionally(ReturnValueUsedM),
                     allOf(hasArgument(0, dst::createMatcher()),
                           hasArgument(1, src::createMatcher()),
                           hasArgument(2, size::createMatcher()))));
  Finder->addMatcher(Expression.bind(ExpressionRef), this);
}

void ReplaceWithStdCopyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void ReplaceWithStdCopyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "FlaggableCallees", FlaggableCallees_);
}

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "clang-tidy"

void ReplaceWithStdCopyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &CallNode = *Result.Nodes.getNodeAs<CallExpr>(ExpressionRef);

  auto dstVT = dst::extractValueType(Result);
  auto srcVT = src::extractValueType(Result);

  // basis for converting size argument to std::copy_n's when issuing fixit
  auto valueTypeWidth = Result.Context->getTypeSizeInChars(dstVT);

  // Don't report cases where value type widths differ, as this might indicate
  // [de]serialization and hard to reason about replacements using std::copy[_n]
  if (valueTypeWidth != Result.Context->getTypeSizeInChars(srcVT))
    return;

  auto dst = dst::extractNode(CallNode, Result);
  auto src = src::extractNode(CallNode, Result);
  auto size = size::extractNode(CallNode, Result);

  // only have this function return a non-empty optional if the form of the size
  // argument strongly indicates collection-related usage
  auto exprAsString = [&](const Expr *Node) {
    return Lexer::getSourceText(
               CharSourceRange::getTokenRange(Node->getSourceRange()),
               *Result.SourceManager, getLangOpts())
        .str();
  };

  auto extractComponents = [&]() -> std::optional<size::SizeComponents> {
    if (const auto *NSizeOfExprNode =
            std::get_if<size::variant::NSizeOfExpr>(&size);
        NSizeOfExprNode != nullptr) {
      auto &[N, Arg] = *NSizeOfExprNode;
      auto sizeOfArgWidth = Result.Context->getTypeSizeInChars(Arg->getType());
      return {{sizeOfArgWidth, exprAsString(N)}};
    }
    if (const auto *StrlenNode = std::get_if<size::variant::Strlen>(&size);
        StrlenNode != nullptr) {
      auto strlenArgTypeWidth = Result.Context->getTypeSizeInChars(
          StrlenNode->Arg->getType()->getPointeeType());
      return {{strlenArgTypeWidth, exprAsString(CallNode.getArg(2))}};
    }
    return std::nullopt;
  };

  if (auto maybeComponents = extractComponents(); maybeComponents.has_value()) {
    auto [unit, nExpr] = *maybeComponents;
    if (unit != valueTypeWidth) {
      // __jm__ tricky to offer a helpful fixit here, as the size argument
      // doesn't seem to be related to the pointee types of src and dst
      return;
    }
    // __jm__ makes sense to assume
  }
  // might be a reinterpretation call, only other case where we don't flag
  if (std::holds_alternative<ptrarg::tag::RawPtr>(dst.Tag) and
      std::holds_alternative<ptrarg::tag::RawPtr>(src.Tag)) {
    // not supported yet, need to come up with a robust heuristic first
    return;
  }

  // both src and dst are collection expressions, enough to wrap them in
  // std::begin calls
  auto Diag = diag(CallNode.getExprLoc(), "prefer std::copy_n to %0")
              << cast<NamedDecl>(CallNode.getCalleeDecl());

  // don't issue a fixit if the result of the call is used
  if (bool IsReturnValueUsed =
          Result.Nodes.getNodeAs<Stmt>(ReturnValueDiscardedRef) == nullptr;
      IsReturnValueUsed)
    return;

  Diag << FixItHint::CreateRemoval(CallNode.getSourceRange());
  std::string srcFixit = "std::cbegin(" + exprAsString(src.Node) + ")";
  std::string dstFixit = "std::begin(" + exprAsString(dst.Node) + ")";

  auto calleeIsWideVariant =
      CallNode.getDirectCallee()->getParamDecl(0)->getType()->isWideCharType();

  auto calleeUnit = [&]() {
    if (calleeIsWideVariant) {
      return Result.Context->getTypeSizeInChars(
          (Result.Context->getWideCharType()));
    }
    return CharUnits::One();
  }();
  auto sizeFixit = [&]() -> std::string {
    if (valueTypeWidth == calleeUnit) {
      return exprAsString(CallNode.getArg(size::argIndex));
    }
    // try to factor out the unit from the size expression
    if (auto maybeSizeComponents = extractComponents();
        maybeSizeComponents.has_value() and
        maybeSizeComponents->Unit == valueTypeWidth) {
      return maybeSizeComponents->NExpr;
    }
    // last resort, divide by valueTypeWidth
    return exprAsString(CallNode.getArg(size::argIndex)) + " / " + "sizeof(" +
           dstVT.getAsString() + ")";
  }();

  Diag << FixItHint::CreateInsertion(CallNode.getEndLoc(),
                                     "std::copy_n(" + srcFixit + ", " +
                                         sizeFixit + ", " + dstFixit + ");");

  // if (const auto *NSizeofExprNode =
  //         std::get_if<size::variant::NSizeOfExpr>(&size);
  //     NSizeofExprNode != nullptr) {
  //   auto &[N, Arg] = *NSizeofExprNode;
  //   auto sizeOfArgWidth = Result.Context->getTypeSizeInChars(Arg->getType());
  //   issueFixitIfWidthsMatch(dst.Node, src.Node,
  //                           {sizeOfArgWidth, exprAsString(N)});
  //   return;
  // }
  // if (const auto *StrlenNode = std::get_if<size::variant::Strlen>(&size);
  //     StrlenNode != nullptr) {
  //   auto strlenArgTypeWidth = Result.Context->getTypeSizeInChars(
  //       StrlenNode->Arg->getType()->getPointeeType());
  //   issueFixitIfWidthsMatch(
  //       dst.Node, src.Node,
  //       {strlenArgTypeWidth, exprAsString(CallNode.getArg(2))});
  //   return;
  // }
  // if (const auto *SizeOfExprNode =
  //         std::get_if<size::variant::SizeOfExpr>(&size);
  //     SizeOfExprNode != nullptr) {
  //   auto &Arg = SizeOfExprNode->Arg;
  //   if (SizeOfExprNode->Arg->getType()->isArrayType()) {
  //     issueFixitIfWidthsMatch(
  //         dst.Node, src.Node,
  //         {CharUnits::One(), exprAsString(CallNode.getArg(2))});
  //     return;
  //   }
  //   // __jm__ weird bc we assume dst and src are collections
  //   // if Arg turns out to have the same type as dst or src then just suggest
  //   // copy via the assignment operator
  //   if (auto argType = Arg->getType(); argType == dstVT or argType == srcVT)
  //   {
  //     issueFixit{valueTypeWidth, exprAsString(CallNode.getArg(2))};
  //     return;
  //   }
  //   // __jm__ only flag this as suspicious with a further explanation in the
  //   // diagnostic TODO: a sizeof of an unrelated type when copying between
  //   // collections does not make a lot of sense

  //   auto sizeDRE = [&]() -> const DeclRefExpr * {
  //     if (const auto *Variant = std::get_if<size::variant::Strlen>(&size);
  //         Variant != nullptr) {
  //       return llvm::dyn_cast_if_present<DeclRefExpr>(Variant->Arg);
  //     }
  //     if (const auto *Variant =
  //     std::get_if<size::variant::SizeOfExpr>(&size);
  //         Variant != nullptr) {
  //       return llvm::dyn_cast_if_present<DeclRefExpr>(Variant->Arg);
  //     }
  //     if (const auto *Variant =
  //     std::get_if<size::variant::NSizeOfExpr>(&size);
  //         Variant != nullptr) {
  //       return llvm::dyn_cast_if_present<DeclRefExpr>(Variant->Arg);
  //     }
  //     return nullptr;
  //   }();

  // one thing the analysis of the size argument must return is an Expr* Node
  // that we can lift into std::copy_n's third argument

  // 1. strlen(DeclRefExpr) also hints that the referenced variable is a
  // collection

  // 2. sizeof(expr)
  //    2.1. expr is a c-array DeclRefExpr to one of src/dst, copy_n size
  //    becomes std::size(expr)
  //    2.2. both src and dst are not raw pointers and expr is a type of width
  //    equal to vtw, essentialy fallthrough to 3

  // 3. N * sizeof(expr) is okay when expr is a type with width ==
  // valueTypeWidth and N may be verbatim lifted into copy_n's third argument

  // to make sure we can issue a FixIt, need to be pretty sure we're dealing
  // with a collection and it is a byte collection
  // 1. For now we are sure both source and dest are collections, but not
  // necessarily of bytes
  // 2. We can relax conditions to where just one arg is a collection, and the
  // other can then be a collection or raw pointer. However, this is not
  // robust as there are cases where this may be used for unmarshalling
  // 3. when both dest and source are pointers (no expectations on the
  // argument as everything required is enforced by the type system) we can
  // use a heuristic using the form of the 3rd argument expression
  //

  // delete the memmove/memcpy
  // insert an std::copy_n

  // using PtrArgVariant = std::variant<CArrayTag, ContainerTag, RawPtrTag>;
  // struct PtrArg {
  //   PtrArgVariant Tag;
  //   const Expr *Node;
  // };

  // const auto MakePtrArg = [&](arg::tag::VariantPtrArgRef Refs) -> PtrArg {
  //   if (const auto *Node = Result.Nodes.getNodeAs<Expr>(
  //           arg::Dest::Refs.VariantRefs.AsCArray);
  //       Node != nullptr) {
  //     // freestanding std::begin
  //     return {CArrayTag{}, Node};
  //   }
  //   if (const auto *Node = Result.Nodes.getNodeAs<Expr>(
  //           arg::Dest::Refs.VariantRefs.AsContainer);
  //       Node != nullptr) {
  //     return {ContainerTag{}, Node};
  //   }
  //   const auto *Node = CallNode.getArg(Refs.FallbackParameterIdx);
  //   return {RawPtrTag{}, Node};
  // };

  // auto Dest = MakePtrArg(arg::Dest::Refs.VariantRefs);
  // auto Source = MakePtrArg(arg::Source::Refs.VariantRefs);

  // if (std::holds_alternative<RawPtrTag>(Dest.Tag) and
  //     std::holds_alternative<RawPtrTag>(Source.Tag) and
  //     CheckSizeArgPermitsFix()) {

  // } else {
  // }

  Diag << Inserter.createIncludeInsertion(
      Result.SourceManager->getFileID(CallNode.getBeginLoc()), StdCopyHeader);
  {
    // using namespace std::literals;
    // Diag << FixItHint::CreateReplacement(
    //     DestArg->getSourceRange(),
    //     ("std::begin(" +
    //      tooling::fixit::getText(SrcArg->getSourceRange(), *Result.Context)
    //      +
    //      ")")
    //         .str());
    // Diag << FixItHint::CreateReplacement(
    //     SrcArg->getSourceRange(),
    //     tooling::fixit::getText(SizeArg->getSourceRange(),
    //     *Result.Context));
    // Diag << FixItHint::CreateReplacement(
    //     SizeArg->getSourceRange(),
    //     ("std::begin(" +
    //      tooling::fixit::getText(DestArg->getSourceRange(),
    //      *Result.Context)
    //      +
    //      ")")
    //         .str());
  }
  // dest source size -> source size dest
}

llvm::ArrayRef<StringRef> ReplaceWithStdCopyCheck::getFlaggableCallees() const {
  switch (FlaggableCallees_) {
  case FlaggableCallees::OnlyMemmove:
    return {"::memmove", "::std::memmove", "::wmemmove", "::std::wmemmove"};
  case FlaggableCallees::MemmoveAndMemcpy:
    return {"::memmove",  "::std::memmove",  "::memcpy",  "::std::memcpy",
            "::wmemmove", "::std::wmemmove", "::wmemcpy", "::std::wmemcpy"};
  }
}
} // namespace modernize
} // namespace clang::tidy