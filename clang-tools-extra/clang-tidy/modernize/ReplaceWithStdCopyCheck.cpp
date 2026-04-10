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

namespace modernize {
namespace {

constexpr llvm::StringLiteral ExpressionRef = "ReplaceWithStdCopyCheckRef";
constexpr llvm::StringLiteral ReturnValueDiscardedRef =
    "ReturnValueDiscardedRef";

namespace ptrarg {

struct Refs {
  llvm::StringLiteral AsContainer;
  llvm::StringLiteral AsCArray;
  size_t FallbackParameterIdx;
  llvm::StringLiteral ValueType;
  llvm::StringLiteral PtrCastFnReturnType;
};

template <typename RefsT> auto createPtrArgMatcher() {
  constexpr Refs Refs = RefsT::Refs;

  auto AllowedContainerNamesM =
      hasAnyName("::std::vector", "::std::basic_string", "::std::array",
                 "::std::basic_string_view", "::std::span");

  // Note: a problem with this matcher is that it automatically desugars the
  // template argument type, so e.g. std::vector<std::int32_t> will bind
  // ValueType to int and not std::int32_t
  auto AllowedContainerTypeM = hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(recordDecl(classTemplateSpecializationDecl(
          AllowedContainerNamesM,
          hasTemplateArgument(0, templateArgument(refersToType(
                                     qualType().bind(Refs.ValueType)))))))));

  auto CArrayTypeM =
      constantArrayType(hasElementType(qualType().bind(Refs.ValueType)));

  auto VariantContainerM =
      expr(hasType(AllowedContainerTypeM)).bind(Refs.AsContainer);

  auto VariantCArrayM = expr(hasType(hasUnqualifiedDesugaredType(CArrayTypeM)))
                            .bind(Refs.AsCArray);

  auto StdDataReturnM =
      returns(pointerType(pointee(qualType().bind(Refs.PtrCastFnReturnType))));

  auto StdDataMemberDeclM = cxxMethodDecl(hasAnyName("data", "c_str"),
                                          parameterCountIs(0), StdDataReturnM);

  auto StdDataFreeDeclM = functionDecl(hasAnyName("::std::data", "::data"),
                                       parameterCountIs(1), StdDataReturnM);

  auto StdDataMemberCallM = cxxMemberCallExpr(
      callee(StdDataMemberDeclM), argumentCountIs(0),
      on(expr(hasType(AllowedContainerTypeM)).bind(Refs.AsContainer)));

  auto ArrayOrContainerM = expr(anyOf(VariantCArrayM, VariantContainerM));

  auto StdDataFreeCallM = callExpr(callee(StdDataFreeDeclM), argumentCountIs(1),
                                   hasArgument(0, ArrayOrContainerM));

  return expr(anyOf(StdDataMemberCallM, StdDataFreeCallM, VariantCArrayM));
}

namespace tag {
struct RawPtr {};
struct Container {};
struct CArray {};
} // namespace tag
struct PtrArg {
  std::variant<tag::RawPtr, tag::Container, tag::CArray> Tag;
  const Expr &Node;
};

template <typename RefT>
PtrArg extractNode(const CallExpr &CallNode,
                   const MatchFinder::MatchResult &Result) {
  constexpr Refs Refs = RefT::Refs;
  if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Refs.AsCArray);
      Node != nullptr)
    return {tag::CArray{}, *Node};
  if (const auto *Node = Result.Nodes.getNodeAs<Expr>(Refs.AsContainer);
      Node != nullptr)
    return {tag::Container{}, *Node};
  return {tag::RawPtr{}, *CallNode.getArg(Refs.FallbackParameterIdx)};
}

template <typename RefT>
const QualType *extractValueType(const MatchFinder::MatchResult &Result) {
  constexpr Refs Refs = RefT::Refs;

  // checking equality is done here as opposed to when matching because the
  // equalsBoundNode matcher depends on the match order.
  // Already considered swapping the role of the node
  // matchers, having one bind and the other match using equalsBoundNode, but
  // PtrCastFnReturnType is only present in some scenarios,
  // so it's not applicable.
  const auto *MaybeRetType =
      Result.Nodes.getNodeAs<QualType>(Refs.PtrCastFnReturnType);
  const auto *ValueType = Result.Nodes.getNodeAs<QualType>(Refs.ValueType);

  // stripping qualifiers is necessary in cases like when matching a call
  // to const T* vector<char>::data() const;
  // the container value_type (char) is not const qualified,
  // but the return type of data() is.
  if (MaybeRetType != nullptr and
      MaybeRetType->getCanonicalType().getUnqualifiedType() !=
          ValueType->getCanonicalType().getUnqualifiedType())
    return nullptr;
  return ValueType;
}
} // namespace ptrarg

namespace dst {
constexpr size_t ArgIndex = 0;
struct RefT {
  static constexpr ptrarg::Refs Refs = {"Dst::AsContainer", "Dst::AsCArray",
                                        ArgIndex, "Dst::ValueType",
                                        "Dst::PtrCastFnReturnType"};
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
constexpr size_t ArgIndex = 1;

struct SrcRefsT {
  static constexpr ptrarg::Refs Refs = {"Src::AsContainer", "Src::AsCArray",
                                        ArgIndex, "Src::ValueType",
                                        "Src::PtrCastFnReturnType"};
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
constexpr size_t ArgIndex = 2;

namespace variant {
struct SizeOfCArray {
  const Expr &Array;
};
struct NSizeOfType {
  const Expr &N;
  const QualType &Arg;
};
struct SizeOfDivSizeOf {
  const Expr &Array;
  const QualType &DivSizeOfType;
};
} // namespace variant
using SizeArg = std::variant<const Expr *, variant::SizeOfCArray,
                             variant::NSizeOfType, variant::SizeOfDivSizeOf>;

static constexpr struct Refs {
  llvm::StringLiteral SizeOfCArray;
  llvm::StringLiteral NSizeOfTypeN;
  llvm::StringLiteral NSizeOfTypeArg;
  llvm::StringLiteral SizeOfDivSizeOfArray;
  llvm::StringLiteral SizeOfDivSizeOfArg;
} Refs = {"Size::SizeOfCArray", "Size::NSizeOfTypeN", "Size::NSizeOfTypeArg",
          "Size::SizeOfDivSizeOfArray", "Size::SizeOfDivSizeOfArg"};

auto createMatcher() {
  // NOTE: this check does not detect common, invalid patterns
  // like sizeof(_) * sizeof(_), etc. since other checks exist for those.

  // patterns of the size argument that may be modified :
  // 1. sizeof(arr)
  //    - invalid if callee is a wide variant,
  //      should be sizeof(arr) / sizeof(wchar_like)
  //    - otherwise -> std::size(arr)
  // 2. N * sizeof(value_like)
  //    - invalid if callee is a wide variant, should just be N
  //    - otherwise when sizeof(value_like) == sizeof(value_type) -> N
  // 3. sizeof(arr) / sizeof(value_like)
  //    - valid if callee is a wide variant -> std::size(arr)
  //    - valid if sizeof(value_like) == 1
  //    - invalid otherwise, will fall back to (expr) / sizeof(value_type)

  constexpr auto SizeOfCArray = [](llvm::StringLiteral Ref) {
    return sizeOfExpr(
        has(expr(hasType(hasUnqualifiedDesugaredType(constantArrayType())))
                .bind(Ref)));
  };

  constexpr auto SizeOfType = [](llvm::StringLiteral Ref) {
    return sizeOfExpr(hasArgumentOfType(qualType().bind(Ref)));
  };

  auto NSizeOfTypeM = binaryOperator(
      hasOperatorName("*"), hasOperands(expr().bind(Refs.NSizeOfTypeN),
                                        SizeOfType(Refs.NSizeOfTypeArg)));

  auto SizeOfDivSizeOfM = binaryOperator(
      hasOperatorName("/"), hasOperands(SizeOfCArray(Refs.SizeOfDivSizeOfArray),
                                        SizeOfType(Refs.SizeOfDivSizeOfArg)));

  return expr(
      anyOf(NSizeOfTypeM, SizeOfCArray(Refs.SizeOfCArray), SizeOfDivSizeOfM));
}

SizeArg extractNode(const CallExpr &CallNode,
                    const MatchFinder::MatchResult &Result) {
  auto TryExtractFromBoundTags =
      [&Nodes = Result.Nodes]() -> std::optional<SizeArg> {
    if (const auto *Array = Nodes.getNodeAs<Expr>(Refs.SizeOfCArray);
        Array != nullptr)
      return variant::SizeOfCArray{*Array};
    llvm::errs() << __LINE__ << '\n';
    if (const auto *N = Nodes.getNodeAs<Expr>(Refs.NSizeOfTypeN);
        N != nullptr) {
      if (const auto *Arg = Nodes.getNodeAs<QualType>(Refs.NSizeOfTypeArg);
          Arg != nullptr)
        return variant::NSizeOfType{*N, *Arg};
      return std::nullopt;
    }
    if (const auto *Array = Nodes.getNodeAs<Expr>(Refs.SizeOfDivSizeOfArray);
        Array != nullptr) {
      if (const auto *SizeOfArg =
              Nodes.getNodeAs<QualType>(Refs.SizeOfDivSizeOfArg);
          SizeOfArg != nullptr)
        return variant::SizeOfDivSizeOf{*Array, *SizeOfArg};
      return std::nullopt;
    }
    return std::nullopt;
  };

  if (auto MaybeSize = TryExtractFromBoundTags(); MaybeSize.has_value())
    return *MaybeSize;
  return CallNode.getArg(ArgIndex);
}
} // namespace size

auto createCalleeMatcher(bool FlagMemcpy) {
  if (FlagMemcpy)
    return hasAnyName("::memmove", "::std::memmove", "::memcpy",
                      "::std::memcpy", "::wmemmove", "::std::wmemmove",
                      "::wmemcpy", "::std::wmemcpy");
  return hasAnyName("::memmove", "::std::memmove", "::wmemmove",
                    "::std::wmemmove");
}

} // namespace

ReplaceWithStdCopyCheck::ReplaceWithStdCopyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      FlagMemcpy(Options.getLocalOrGlobal("FlagMemcpy", false)) {}

void ReplaceWithStdCopyCheck::registerMatchers(MatchFinder *Finder) {
  const auto ReturnValueUsedM =
      hasParent(compoundStmt().bind(ReturnValueDiscardedRef));

  const auto OffendingDeclM =
      functionDecl(parameterCountIs(3), createCalleeMatcher(FlagMemcpy));

  const auto ExpressionM = callExpr(
      callee(OffendingDeclM), optionally(ReturnValueUsedM),
      allOf(optionally(hasArgument(dst::ArgIndex, dst::createMatcher())),
            optionally(hasArgument(src::ArgIndex, src::createMatcher())),
            optionally(hasArgument(size::ArgIndex, size::createMatcher()))));
  Finder->addMatcher(ExpressionM.bind(ExpressionRef), this);
}

void ReplaceWithStdCopyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void ReplaceWithStdCopyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "FlagMemcpy", FlagMemcpy);
}

void ReplaceWithStdCopyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &CallNode = *Result.Nodes.getNodeAs<CallExpr>(ExpressionRef);

  auto Dst = dst::extractNode(CallNode, Result);
  auto Src = src::extractNode(CallNode, Result);
  auto Size = size::extractNode(CallNode, Result);

  const auto *DstVT = dst::extractValueType(Result);
  if (DstVT == nullptr)
    return;
  const auto *SrcVT = src::extractValueType(Result);
  if (SrcVT == nullptr)
    return;

  auto ExprAsString = [&](const Expr &Node) {
    return Lexer::getSourceText(
               CharSourceRange::getTokenRange(Node.getSourceRange()),
               *Result.SourceManager, getLangOpts())
        .str();
  };

  bool DstIsRawPtr = std::holds_alternative<ptrarg::tag::RawPtr>(Dst.Tag);
  bool SrcIsRawPtr = std::holds_alternative<ptrarg::tag::RawPtr>(Src.Tag);

  auto CheckIsFalsePositive = [&]() {
    // This case could be supported but it requires a robust heuristic
    // over the form of the size argument
    // to deduce we are dealing with pointers to collections
    // When only one of the arguments is a raw pointer, then there are still
    // valid scenarios in which it is a singleton to be memmoved over, like
    // [de]serialization.
    return DstIsRawPtr or SrcIsRawPtr;
  };
  if (CheckIsFalsePositive())
    return;

  auto Diag = diag(CallNode.getExprLoc(), "prefer std::copy_n to %0")
              << cast<NamedDecl>(CallNode.getCalleeDecl());

  // the value type widths are helpful when translating the size argument
  // from byte units (memcpy etc.) to element units (std::copy_n),
  auto DstTypeWidth = Result.Context->getTypeSizeInChars(*DstVT);
  auto SrcTypeWidth = Result.Context->getTypeSizeInChars(*SrcVT);

  auto CalleeIsWideVariant = [&]() {
    const auto *Callee = CallNode.getDirectCallee();
    if (Callee == nullptr)
      return false;
    auto *ParamDecl = Callee->getParamDecl(0);
    if (ParamDecl == nullptr)
      return false;
    return ParamDecl->getType()->getPointeeType()->isWideCharType();
  }();

  auto CalleeUnit = [&]() {
    if (CalleeIsWideVariant)
      return Result.Context->getTypeSizeInChars(
          (Result.Context->getWideCharType()));
    return CharUnits::One();
  }();

  auto CheckIsFixable = [&]() {
    // If the width types differ, it's hard to reason about what would be a
    // helpful replacement, so just don't issue a fixit in this case
    if (DstTypeWidth != SrcTypeWidth)
      return false;

    // don't issue a fixit if the result of the call is used
    if (bool IsReturnValueUsed =
            Result.Nodes.getNodeAs<Stmt>(ReturnValueDiscardedRef) == nullptr;
        IsReturnValueUsed)
      return false;

    // for widechar variants assume that the value types are also
    // of wchar_t width, to make analysis easier.
    if (CalleeIsWideVariant and DstTypeWidth != CalleeUnit)
      return false;

    return true;
  };
  if (not CheckIsFixable())
    return;

  assert(DstTypeWidth == SrcTypeWidth);
  const auto &ValueTypeWidth = DstTypeWidth;

  auto SrcFixit = [&]() {
    auto AsString = ExprAsString(Src.Node);
    if (SrcIsRawPtr)
      return AsString;
    return (llvm::Twine() + "std::cbegin(" + AsString + ")").str();
  }();

  auto DstFixit = [&]() {
    auto AsString = ExprAsString(Dst.Node);
    if (DstIsRawPtr)
      return AsString;
    return (llvm::Twine() + "std::begin(" + AsString + ")").str();
  }();

  // This function is used to specify when a type from a sizeof(_) call is
  // considered equivalent to the value type of the collection.
  // For now it is relaxed because the matcher desugars
  // the container value types automatically.
  auto CheckIsEquivValueType = [&DstVT](QualType SizeArgType) {
    return SizeArgType->getCanonicalTypeUnqualified() ==
           (*DstVT)->getCanonicalTypeUnqualified();
  };

  auto ByteModifySizeArg = [&]() -> std::string {
    if (const auto *SizeOfCArray =
            std::get_if<size::variant::SizeOfCArray>(&Size);
        SizeOfCArray != nullptr) {
      // simply replaces sizeof(arr) with std::size(arr)
      return (llvm::Twine() + "std::size(" + ExprAsString(SizeOfCArray->Array) +
              ")")
          .str();
    }
    if (const auto *NSizeofExprNode =
            std::get_if<size::variant::NSizeOfType>(&Size);
        NSizeofExprNode != nullptr) {
      auto &[N, Arg] = *NSizeofExprNode;
      // In this case it is easy to factor out the byte multiplier
      // by just dropping the sizeof expression from the size computation
      if (CheckIsEquivValueType(Arg))
        return ExprAsString(N);
    }
    if (const auto *SizeOfDivSizeOf =
            std::get_if<size::variant::SizeOfDivSizeOf>(&Size);
        SizeOfDivSizeOf != nullptr) {
      auto &[Array, DivSizeOfType] = *SizeOfDivSizeOf;
      if (CheckIsEquivValueType(DivSizeOfType) and ValueTypeWidth == CalleeUnit)
        return (llvm::Twine() + "std::size(" + ExprAsString(Array) + ")").str();
    }

    // In the specific case where the collections' value types are byte-wide,
    // no unit conversion of the size argument is necessary
    if (ValueTypeWidth == CalleeUnit)
      return ExprAsString(*CallNode.getArg(size::ArgIndex));

    // For all other cases where the value type is wider than one byte,
    // and the size argument is of a form that is not easy to factor the unit
    // out of, perform explicit division to ensure it is in element units
    return (llvm::Twine() + "(" +
            ExprAsString(*CallNode.getArg(size::ArgIndex)) + ") / sizeof(" +
            DstVT->getAsString() + ")")
        .str();
  };

  auto WideModifySizeArg = [&]() {
    if (const auto *SizeOfDivSizeOf =
            std::get_if<size::variant::SizeOfDivSizeOf>(&Size);
        SizeOfDivSizeOf != nullptr) {
      auto &[Array, DivSizeOfType] = *SizeOfDivSizeOf;
      if (CheckIsEquivValueType(DivSizeOfType))
        return (llvm::Twine() + "std::size(" + ExprAsString(Array) + ")").str();
    }
    // Since we assume wide variants' value type width equals wchar_t,
    // the units should already be unified and no modifications to the size
    // argument are necessary
    return ExprAsString(*CallNode.getArg(size::ArgIndex));
  };

  auto SizeFixit =
      CalleeIsWideVariant ? WideModifySizeArg() : ByteModifySizeArg();

  Diag << FixItHint::CreateRemoval(CallNode.getSourceRange())
       << FixItHint::CreateInsertion(CallNode.getBeginLoc(),
                                     (llvm::Twine() + "std::copy_n(" +
                                      SrcFixit + ", " + SizeFixit + ", " +
                                      DstFixit + ")")
                                         .str())
       << Inserter.createIncludeInsertion(
              Result.SourceManager->getFileID(CallNode.getBeginLoc()),
              "<algorithm>");

  // All containers will contain an std::[c]begin declaration with their
  // definition, with the exception of constant c-arrays
  if (std::holds_alternative<ptrarg::tag::CArray>(Dst.Tag) and
      std::holds_alternative<ptrarg::tag::CArray>(Src.Tag))
    Diag << Inserter.createIncludeInsertion(
        Result.SourceManager->getFileID(CallNode.getBeginLoc()), "<iterator>");
}

} // namespace modernize
} // namespace clang::tidy