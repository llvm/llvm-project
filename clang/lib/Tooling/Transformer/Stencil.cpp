//===--- Stencil.cpp - Stencil implementation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/Stencil.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Tooling/Transformer/SourceCodeBuilders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include <atomic>
#include <memory>
#include <string>

using namespace clang;
using namespace transformer;

using ast_matchers::MatchFinder;
using llvm::errc;
using llvm::Error;
using llvm::Expected;
using llvm::StringError;

static llvm::Expected<DynTypedNode>
getNode(const ast_matchers::BoundNodes &Nodes, StringRef Id) {
  auto &NodesMap = Nodes.getMap();
  auto It = NodesMap.find(Id);
  if (It == NodesMap.end())
    return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                               "Id not bound: " + Id);
  return It->second;
}

namespace {
// An arbitrary fragment of code within a stencil.
struct RawTextData {
  explicit RawTextData(std::string T) : Text(std::move(T)) {}
  std::string Text;
};

// A debugging operation to dump the AST for a particular (bound) AST node.
struct DebugPrintNodeData {
  explicit DebugPrintNodeData(std::string S) : Id(std::move(S)) {}
  std::string Id;
};

// Operators that take a single node Id as an argument.
enum class UnaryNodeOperator {
  Parens,
  Deref,
  MaybeDeref,
  AddressOf,
  MaybeAddressOf,
};

// Generic container for stencil operations with a (single) node-id argument.
struct UnaryOperationData {
  UnaryOperationData(UnaryNodeOperator Op, std::string Id)
      : Op(Op), Id(std::move(Id)) {}
  UnaryNodeOperator Op;
  std::string Id;
};

// The fragment of code corresponding to the selected range.
struct SelectorData {
  explicit SelectorData(RangeSelector S) : Selector(std::move(S)) {}
  RangeSelector Selector;
};

// A stencil operation to build a member access `e.m` or `e->m`, as appropriate.
struct AccessData {
  AccessData(StringRef BaseId, Stencil Member)
      : BaseId(std::string(BaseId)), Member(std::move(Member)) {}
  std::string BaseId;
  Stencil Member;
};

struct IfBoundData {
  IfBoundData(StringRef Id, Stencil TrueStencil, Stencil FalseStencil)
      : Id(std::string(Id)), TrueStencil(std::move(TrueStencil)),
        FalseStencil(std::move(FalseStencil)) {}
  std::string Id;
  Stencil TrueStencil;
  Stencil FalseStencil;
};

struct SequenceData {
  SequenceData(std::vector<Stencil> Stencils) : Stencils(std::move(Stencils)) {}
  std::vector<Stencil> Stencils;
};

std::string toStringData(const RawTextData &Data) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "\"";
  OS.write_escaped(Data.Text);
  OS << "\"";
  OS.flush();
  return Result;
}

std::string toStringData(const DebugPrintNodeData &Data) {
  return (llvm::Twine("dPrint(\"") + Data.Id + "\")").str();
}

std::string toStringData(const UnaryOperationData &Data) {
  StringRef OpName;
  switch (Data.Op) {
  case UnaryNodeOperator::Parens:
    OpName = "expression";
    break;
  case UnaryNodeOperator::Deref:
    OpName = "deref";
    break;
  case UnaryNodeOperator::MaybeDeref:
    OpName = "maybeDeref";
    break;
  case UnaryNodeOperator::AddressOf:
    OpName = "addressOf";
    break;
  case UnaryNodeOperator::MaybeAddressOf:
    OpName = "maybeAddressOf";
    break;
  }
  return (OpName + "(\"" + Data.Id + "\")").str();
}

std::string toStringData(const SelectorData &) { return "selection(...)"; }

std::string toStringData(const AccessData &Data) {
  return (llvm::Twine("access(\"") + Data.BaseId + "\", " +
          Data.Member->toString() + ")")
      .str();
}

std::string toStringData(const IfBoundData &Data) {
  return (llvm::Twine("ifBound(\"") + Data.Id + "\", " +
          Data.TrueStencil->toString() + ", " + Data.FalseStencil->toString() +
          ")")
      .str();
}

std::string toStringData(const MatchConsumer<std::string> &) {
  return "run(...)";
}

std::string toStringData(const SequenceData &Data) {
  llvm::SmallVector<std::string, 2> Parts;
  Parts.reserve(Data.Stencils.size());
  for (const auto &S : Data.Stencils)
    Parts.push_back(S->toString());
  return (llvm::Twine("seq(") + llvm::join(Parts, ", ") + ")").str();
}

// The `evalData()` overloads evaluate the given stencil data to a string, given
// the match result, and append it to `Result`. We define an overload for each
// type of stencil data.

Error evalData(const RawTextData &Data, const MatchFinder::MatchResult &,
               std::string *Result) {
  Result->append(Data.Text);
  return Error::success();
}

Error evalData(const DebugPrintNodeData &Data,
               const MatchFinder::MatchResult &Match, std::string *Result) {
  std::string Output;
  llvm::raw_string_ostream Os(Output);
  auto NodeOrErr = getNode(Match.Nodes, Data.Id);
  if (auto Err = NodeOrErr.takeError())
    return Err;
  NodeOrErr->print(Os, PrintingPolicy(Match.Context->getLangOpts()));
  *Result += Os.str();
  return Error::success();
}

Error evalData(const UnaryOperationData &Data,
               const MatchFinder::MatchResult &Match, std::string *Result) {
  const auto *E = Match.Nodes.getNodeAs<Expr>(Data.Id);
  if (E == nullptr)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Id not bound or not Expr: " + Data.Id);
  llvm::Optional<std::string> Source;
  switch (Data.Op) {
  case UnaryNodeOperator::Parens:
    Source = tooling::buildParens(*E, *Match.Context);
    break;
  case UnaryNodeOperator::Deref:
    Source = tooling::buildDereference(*E, *Match.Context);
    break;
  case UnaryNodeOperator::MaybeDeref:
    if (!E->getType()->isAnyPointerType()) {
      *Result += tooling::getText(*E, *Match.Context);
      return Error::success();
    }
    Source = tooling::buildDereference(*E, *Match.Context);
    break;
  case UnaryNodeOperator::AddressOf:
    Source = tooling::buildAddressOf(*E, *Match.Context);
    break;
  case UnaryNodeOperator::MaybeAddressOf:
    if (E->getType()->isAnyPointerType()) {
      *Result += tooling::getText(*E, *Match.Context);
      return Error::success();
    }
    Source = tooling::buildAddressOf(*E, *Match.Context);
    break;
  }
  if (!Source)
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Could not construct expression source from ID: " + Data.Id);
  *Result += *Source;
  return Error::success();
}

Error evalData(const SelectorData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  auto RawRange = Data.Selector(Match);
  if (!RawRange)
    return RawRange.takeError();
  CharSourceRange Range = Lexer::makeFileCharRange(
      *RawRange, *Match.SourceManager, Match.Context->getLangOpts());
  if (Range.isInvalid()) {
    // Validate the original range to attempt to get a meaningful error message.
    // If it's valid, then something else is the cause and we just return the
    // generic failure message.
    if (auto Err = tooling::validateEditRange(*RawRange, *Match.SourceManager))
      return handleErrors(std::move(Err), [](std::unique_ptr<StringError> E) {
        assert(E->convertToErrorCode() ==
                   llvm::make_error_code(errc::invalid_argument) &&
               "Validation errors must carry the invalid_argument code");
        return llvm::createStringError(
            errc::invalid_argument,
            "selected range could not be resolved to a valid source range; " +
                E->getMessage());
      });
    return llvm::createStringError(
        errc::invalid_argument,
        "selected range could not be resolved to a valid source range");
  }
  // Validate `Range`, because `makeFileCharRange` accepts some ranges that
  // `validateEditRange` rejects.
  if (auto Err = tooling::validateEditRange(Range, *Match.SourceManager))
    return joinErrors(
        llvm::createStringError(errc::invalid_argument,
                                "selected range is not valid for editing"),
        std::move(Err));
  *Result += tooling::getText(Range, *Match.Context);
  return Error::success();
}

Error evalData(const AccessData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  const auto *E = Match.Nodes.getNodeAs<Expr>(Data.BaseId);
  if (E == nullptr)
    return llvm::make_error<StringError>(errc::invalid_argument,
                                         "Id not bound: " + Data.BaseId);
  if (!E->isImplicitCXXThis()) {
    if (llvm::Optional<std::string> S =
            E->getType()->isAnyPointerType()
                ? tooling::buildArrow(*E, *Match.Context)
                : tooling::buildDot(*E, *Match.Context))
      *Result += *S;
    else
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Could not construct object text from ID: " + Data.BaseId);
  }
  return Data.Member->eval(Match, Result);
}

Error evalData(const IfBoundData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  auto &M = Match.Nodes.getMap();
  return (M.find(Data.Id) != M.end() ? Data.TrueStencil : Data.FalseStencil)
      ->eval(Match, Result);
}

Error evalData(const MatchConsumer<std::string> &Fn,
               const MatchFinder::MatchResult &Match, std::string *Result) {
  Expected<std::string> Value = Fn(Match);
  if (!Value)
    return Value.takeError();
  *Result += *Value;
  return Error::success();
}

Error evalData(const SequenceData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  for (const auto &S : Data.Stencils)
    if (auto Err = S->eval(Match, Result))
      return Err;
  return Error::success();
}

template <typename T> class StencilImpl : public StencilInterface {
  T Data;

public:
  template <typename... Ps>
  explicit StencilImpl(Ps &&... Args) : Data(std::forward<Ps>(Args)...) {}

  Error eval(const MatchFinder::MatchResult &Match,
             std::string *Result) const override {
    return evalData(Data, Match, Result);
  }

  std::string toString() const override { return toStringData(Data); }
};
} // namespace

Stencil transformer::detail::makeStencil(StringRef Text) {
  return std::make_shared<StencilImpl<RawTextData>>(std::string(Text));
}

Stencil transformer::detail::makeStencil(RangeSelector Selector) {
  return std::make_shared<StencilImpl<SelectorData>>(std::move(Selector));
}

Stencil transformer::dPrint(StringRef Id) {
  return std::make_shared<StencilImpl<DebugPrintNodeData>>(std::string(Id));
}

Stencil transformer::expression(llvm::StringRef Id) {
  return std::make_shared<StencilImpl<UnaryOperationData>>(
      UnaryNodeOperator::Parens, std::string(Id));
}

Stencil transformer::deref(llvm::StringRef ExprId) {
  return std::make_shared<StencilImpl<UnaryOperationData>>(
      UnaryNodeOperator::Deref, std::string(ExprId));
}

Stencil transformer::maybeDeref(llvm::StringRef ExprId) {
  return std::make_shared<StencilImpl<UnaryOperationData>>(
      UnaryNodeOperator::MaybeDeref, std::string(ExprId));
}

Stencil transformer::addressOf(llvm::StringRef ExprId) {
  return std::make_shared<StencilImpl<UnaryOperationData>>(
      UnaryNodeOperator::AddressOf, std::string(ExprId));
}

Stencil transformer::maybeAddressOf(llvm::StringRef ExprId) {
  return std::make_shared<StencilImpl<UnaryOperationData>>(
      UnaryNodeOperator::MaybeAddressOf, std::string(ExprId));
}

Stencil transformer::access(StringRef BaseId, Stencil Member) {
  return std::make_shared<StencilImpl<AccessData>>(BaseId, std::move(Member));
}

Stencil transformer::ifBound(StringRef Id, Stencil TrueStencil,
                             Stencil FalseStencil) {
  return std::make_shared<StencilImpl<IfBoundData>>(Id, std::move(TrueStencil),
                                                    std::move(FalseStencil));
}

Stencil transformer::run(MatchConsumer<std::string> Fn) {
  return std::make_shared<StencilImpl<MatchConsumer<std::string>>>(
      std::move(Fn));
}

Stencil transformer::catVector(std::vector<Stencil> Parts) {
  // Only one argument, so don't wrap in sequence.
  if (Parts.size() == 1)
    return std::move(Parts[0]);
  return std::make_shared<StencilImpl<SequenceData>>(std::move(Parts));
}
