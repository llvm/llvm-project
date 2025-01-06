//===--- ReplaceWithStdCopyCheck.cpp - clang-tidy----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceWithStdCopyCheck.h"
#include "ReplaceAutoPtrCheck.h"
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

constexpr llvm::StringLiteral ExpressionRef = "::std::memcpy";
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
// rn matchers have a problem with binding types correctly, clang-query build up
// the matchers from this file tmrw for debugging
template <typename RefsT> auto createPtrArgMatcher() {
  constexpr Refs Refs = RefsT::Refs;

  auto AllowedContainerNamesM = []() {
    // return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
    //                   "::std::vector", "::std::basic_string",
    //                   "::std::array");
    return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
                      "::std::vector", "::std::basic_string", "::std::array",
                      "::std::basic_string_view", "::std::span");
  }();

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

  auto VariantRawPtrM =
      expr(hasType(pointerType(pointee(qualType().bind(Refs.ValueType)))));

  auto StdDataReturnM =
      returns(pointerType(pointee(qualType().bind(Refs.PtrCastFnReturnType))));

  auto StdDataMemberDeclM =
      cxxMethodDecl(hasName("data"), parameterCountIs(0), StdDataReturnM);
  // ,unless(isConst()) // __jm__ ONLY difference between Dest
  // and Source calls, but maybe don't include it?

  auto StdDataFreeDeclM = functionDecl(hasAnyName("::std::data", "::data"),
                                       parameterCountIs(1), StdDataReturnM);

  auto StdDataMemberCallM = cxxMemberCallExpr(
      callee(StdDataMemberDeclM), argumentCountIs(0),
      on(expr(hasType(AllowedContainerTypeM)).bind(Refs.AsContainer)));

  auto ArrayOrContainerM = expr(anyOf(VariantCArrayM, VariantContainerM));

  auto StdDataFreeCallM = callExpr(callee(StdDataFreeDeclM), argumentCountIs(1),
                                   hasArgument(0, ArrayOrContainerM));

  // the last expr() in anyOf assumes previous matchers are ran eagerly from
  // left to right, still need to test this is the actual behaviour
  return expr(anyOf(StdDataMemberCallM, StdDataFreeCallM, VariantCArrayM,
                    VariantRawPtrM));
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
  const auto *MaybeRetType =
      Result.Nodes.getNodeAs<QualType>(Refs.PtrCastFnReturnType);
  const auto *ValueType = Result.Nodes.getNodeAs<QualType>(Refs.ValueType);
  // checking equality is done here as opposed to when matching because the
  // equalsBoundNode matcher depends on the match order and the
  // PtrCastFnReturnType is only present in some scenarios
  // if (MaybeRetType != nullptr)
  //   llvm::errs()
  //       << "MaybeRetType: "
  //       <<
  //       MaybeRetType->getCanonicalType().getUnqualifiedType().getAsString()
  //       << '\n';
  // if (ValueType != nullptr)
  //   llvm::errs()
  //       << "ValueType: "
  //       << ValueType->getCanonicalType().getUnqualifiedType().getAsString()
  //       << '\n';

  // stripping qualifiers is necessary in cases like when matching a call
  // to const T* data() const; of a std::vector<char> -
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
  static constexpr ptrarg::Refs Refs = {"Dst::AsContainer", "Dst::AsCArray", 0,
                                        "Dst::ValueType",
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
  static constexpr ptrarg::Refs Refs = {"Src::AsContainer", "Src::AsCArray", 1,
                                        "Src::ValueType",
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
// cases:
// 1. sizeof(T) where T is a type
// 2. sizeof(rec) where rec is a CArray
// 3. N * sizeof(T)
// 4. strlen(rec) [+ 1]
// 5. other expr
namespace variant {
struct SizeOfExpr {
  const Expr *N = nullptr;
  const Expr &Arg;
};
struct SizeOfType {
  const Expr *N = nullptr;
  const QualType T;
};
struct Strlen {
  const Expr &Arg;
};
} // namespace variant
using SizeArg = std::variant<const Expr *, variant::SizeOfExpr,
                             variant::SizeOfType, variant::Strlen>;

struct SizeComponents {
  CharUnits Unit;
  std::string NExpr;
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
  auto SizeOfExprM = sizeOfExpr(expr().bind(Refs.SizeOfArg));

  auto NSizeOfExprM = binaryOperator(
      hasOperatorName("*"), hasOperands(expr().bind(Refs.N), SizeOfExprM));

  auto StrlenM =
      callExpr(callee(functionDecl(hasAnyName(
                   "::strlen", "::std::strlen", "::wcslen", "::std::wcslen",
                   "::strnlen_s", "::std::strnlen_s", "::wcsnlen_s",
                   "::std::wcsnlen_s"))),
               hasArgument(0, expr().bind(Refs.StrlenArg)));

  auto StrlenPlusOneM = binaryOperator(
      hasOperatorName("+"), hasOperands(StrlenM, integerLiteral(equals(1))));

  return expr(anyOf(NSizeOfExprM, StrlenM, StrlenPlusOneM, SizeOfExprM));
}

SizeArg extractNode(const CallExpr &CallNode,
                    const MatchFinder::MatchResult &Result) {
  llvm::errs() << "Dumps Size:\n";
  CallNode.dump();
  llvm::errs() << __LINE__ << '\n';
  if (const auto *SizeOfExprNode =
          Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>(Refs.SizeOfArg);
      SizeOfExprNode != nullptr) {
    llvm::errs() << __LINE__ << '\n';
    const auto *NNode = Result.Nodes.getNodeAs<Expr>(Refs.N);
    if (const auto *ArgAsExpr = SizeOfExprNode->getArgumentExpr();
        ArgAsExpr != nullptr)
      return variant::SizeOfExpr{NNode, *ArgAsExpr};
    llvm::errs() << __LINE__ << '\n';
    return variant::SizeOfType{NNode, SizeOfExprNode->getTypeOfArgument()};
  }
  llvm::errs() << __LINE__ << '\n';
  if (const auto *StrlenArgNode = Result.Nodes.getNodeAs<Expr>(Refs.StrlenArg);
      StrlenArgNode != nullptr)
    return variant::Strlen{*StrlenArgNode};
  llvm::errs() << __LINE__ << '\n';
  return CallNode.getArg(2);
}
// 1. N * sizeof(type) (commutative) - issue warning but no fixit, or fixit
// only if both src/dest are fixit friendly 1.1. This might allow fixits for
// wider-than-byte element collections
// 2. strlen(src|dest) ?(+ 1 (commutative)) -- issue warning but no fixit
// 4. sizeof(variable) only if that variable is of arrayType, if it's of
// ptrType that may indicate [de]serialization

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

  // cases:
  // 1. One of the arguments is definitely a collection and the other a pointer
  // - match
  // 2. Both source and dest are pointers, but size is of the form ((N :=
  // expr()) * sizeof(bytelike())) - match (false positive if N \in {0, 1})

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
  llvm::errs() << __LINE__ << '\n';
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
  llvm::errs() << "Call: " << ExprAsString(CallNode) << '\n';

  // only have this function return a non-empty optional if the form of the size
  // argument strongly indicates collection-related usage
  auto ExtractComponents = [&]() -> std::optional<size::SizeComponents> {
    llvm::errs() << __LINE__ << '\n';
    if (const auto *NSizeOfTypeNode =
            std::get_if<size::variant::SizeOfType>(&Size);
        NSizeOfTypeNode != nullptr) {
      llvm::errs() << __LINE__ << '\n';
      auto &[N, T] = *NSizeOfTypeNode;

      auto WidthT = Result.Context->getTypeSizeInChars(T);
      auto StrN = N == nullptr ? "1" : ExprAsString(*N);

      return {{WidthT, StrN}};
    }
    if (const auto *NSizeOfExprNode =
            std::get_if<size::variant::SizeOfExpr>(&Size);
        NSizeOfExprNode != nullptr) {
      llvm::errs() << __LINE__ << '\n';
      auto &[N, Arg] = *NSizeOfExprNode;
      if (Arg.getType()->isConstantArrayType()) {
      }
      auto SizeOfArgWidth = Result.Context->getTypeSizeInChars(Arg.getType());
      llvm::errs() << SizeOfArgWidth.getQuantity() << '\n';
      return {{SizeOfArgWidth, ExprAsString(N)}};
    }
    // if (const auto *NSizeOfExprNode =
    //         std::get_if<size::variant::SizeOfExpr>(&Size);
    //     NSizeOfExprNode != nullptr) {
    //   llvm::errs() << __LINE__ << '\n' << "chuj\n";
    //   auto &[N, Arg] = *NSizeOfExprNode;
    //   auto SizeOfArgWidth =
    //   Result.Context->getTypeSizeInChars(Arg.getType()); llvm::errs() <<
    //   SizeOfArgWidth.getQuantity() << '\n'; return {{SizeOfArgWidth,
    //   ExprAsString(N)}};
    // }
    if (const auto *StrlenNode = std::get_if<size::variant::Strlen>(&Size);
        StrlenNode != nullptr) {
      llvm::errs() << __LINE__ << '\n';
      auto StrlenArgTypeWidth = Result.Context->getTypeSizeInChars(
          StrlenNode->Arg.getType()->getPointeeType());
      return {{StrlenArgTypeWidth, ExprAsString(*CallNode.getArg(2))}};
    }
    llvm::errs() << __LINE__ << '\n';
    return std::nullopt;
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

  // basis for converting size argument to std::copy_n's when issuing fixit
  auto DstTypeWidth = Result.Context->getTypeSizeInChars(*DstVT);
  auto SrcTypeWidth = Result.Context->getTypeSizeInChars(*SrcVT);

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

    return true;
  };
  if (not CheckIsFixable())
    return;

  // From here we can assume dst and src have equal value type widths
  const auto &ValueTypeWidth = DstTypeWidth;

  auto SrcFixit = [&]() {
    auto AsString = ExprAsString(Src.Node);
    if (SrcIsRawPtr)
      return AsString;
    return "std::cbegin(" + AsString + ")";
  }();

  auto DstFixit = [&]() {
    auto AsString = ExprAsString(Dst.Node);
    if (DstIsRawPtr)
      return AsString;
    return "std::begin(" + AsString + ")";
  }();

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
    if (CalleeIsWideVariant) {
      return Result.Context->getTypeSizeInChars(
          (Result.Context->getWideCharType()));
    }
    return CharUnits::One();
  }();

  auto SizeFixit = [&]() -> std::string {
    // try to factor out the unit from the size expression
    llvm::errs() << __LINE__ << '\n';
    if (auto MaybeSizeComponents = ExtractComponents();
        MaybeSizeComponents.has_value()) {

      // __jm__ TODO
      llvm::errs() << __LINE__ << '\n';
      return MaybeSizeComponents->NExpr;
    }
    // last resort, divide by ValueTypeWidth
    llvm::errs() << __LINE__ << '\n';
    return ExprAsString(*CallNode.getArg(size::ArgIndex)) + " / " + "sizeof(" +
           DstVT->getAsString() + ")";
  }();

  Diag << FixItHint::CreateRemoval(CallNode.getSourceRange())
       << FixItHint::CreateInsertion(CallNode.getBeginLoc(),
                                     "std::copy_n(" + SrcFixit + ", " +
                                         SizeFixit + ", " + DstFixit + ");")
       << Inserter.createIncludeInsertion(
              Result.SourceManager->getFileID(CallNode.getBeginLoc()),
              "<algorithm>");

  // All containers will contain an std::[c]begin declaration with their
  // definition, with the exception of constant c-arrays
  if (std::holds_alternative<ptrarg::tag::CArray>(Dst.Tag) and
      std::holds_alternative<ptrarg::tag::CArray>(Src.Tag))
    Diag << Inserter.createIncludeInsertion(
        Result.SourceManager->getFileID(CallNode.getBeginLoc()), "<iterator>");

  // if (const auto *NSizeofExprNode =
  //         std::get_if<size::variant::NSizeOfExpr>(&size);
  //     NSizeofExprNode != nullptr) {
  //   auto &[N, Arg] = *NSizeofExprNode;
  //   auto SizeOfArgWidth =
  //   Result.Context->getTypeSizeInChars(Arg->getType());
  //   issueFixitIfWidthsMatch(dst.Node, src.Node,
  //                           {SizeOfArgWidth, ExprAsString(N)});
  //   return;
  // }
  // if (const auto *StrlenNode = std::get_if<size::variant::Strlen>(&size);
  //     StrlenNode != nullptr) {
  //   auto StrlenArgTypeWidth = Result.Context->getTypeSizeInChars(
  //       StrlenNode->Arg->getType()->getPointeeType());
  //   issueFixitIfWidthsMatch(
  //       dst.Node, src.Node,
  //       {StrlenArgTypeWidth, ExprAsString(CallNode.getArg(2))});
  //   return;
  // }
  // if (const auto *SizeOfExprNode =
  //         std::get_if<size::variant::SizeOfExpr>(&size);
  //     SizeOfExprNode != nullptr) {
  //   auto &Arg = SizeOfExprNode->Arg;
  //   if (SizeOfExprNode->Arg->getType()->isArrayType()) {
  //     issueFixitIfWidthsMatch(
  //         dst.Node, src.Node,
  //         {CharUnits::One(), ExprAsString(CallNode.getArg(2))});
  //     return;
  //   }
  //   // __jm__ weird bc we assume dst and src are collections
  //   // if Arg turns out to have the same type as dst or src then just
  //   suggest
  //   // copy via the assignment operator
  //   if (auto argType = Arg->getType(); argType == DstVT or argType ==
  //   SrcVT)
  //   {
  //     issueFixit{ValueTypeWidth, ExprAsString(CallNode.getArg(2))};
  //     return;
  //   }
  //   // __jm__ only flag this as suspicious with a further explanation in
  //   the
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
  // ValueTypeWidth and N may be verbatim lifted into copy_n's third argument

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
}

} // namespace modernize
} // namespace clang::tidy