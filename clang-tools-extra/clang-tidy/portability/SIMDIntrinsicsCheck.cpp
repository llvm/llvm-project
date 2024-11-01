//===--- SIMDIntrinsicsCheck.cpp - clang-tidy------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIMDIntrinsicsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"

using namespace clang::ast_matchers;

namespace clang::tidy::portability {

namespace {

// If the callee has parameter of VectorType or pointer to VectorType,
// or the return type is VectorType, we consider it a vector function
// and a candidate for checking.
AST_MATCHER(FunctionDecl, isVectorFunction) {
  bool IsVector = Node.getReturnType()->isVectorType();
  for (const ParmVarDecl *Parm : Node.parameters()) {
    QualType Type = Parm->getType();
    if (Type->isPointerType())
      Type = Type->getPointeeType();
    if (Type->isVectorType())
      IsVector = true;
  }
  return IsVector;
}

} // namespace

static StringRef trySuggestPpc(StringRef Name) {
  if (!Name.consume_front("vec_"))
    return {};

  return llvm::StringSwitch<StringRef>(Name)
      // [simd.alg]
      .Case("max", "$std::max")
      .Case("min", "$std::min")
      // [simd.binary]
      .Case("add", "operator+ on $simd objects")
      .Case("sub", "operator- on $simd objects")
      .Case("mul", "operator* on $simd objects")
      .Default({});
}

static StringRef trySuggestX86(StringRef Name) {
  if (!(Name.consume_front("_mm_") || Name.consume_front("_mm256_") ||
        Name.consume_front("_mm512_")))
    return {};

  // [simd.alg]
  if (Name.startswith("max_"))
    return "$simd::max";
  if (Name.startswith("min_"))
    return "$simd::min";

  // [simd.binary]
  if (Name.startswith("add_"))
    return "operator+ on $simd objects";
  if (Name.startswith("sub_"))
    return "operator- on $simd objects";
  if (Name.startswith("mul_"))
    return "operator* on $simd objects";

  return {};
}

SIMDIntrinsicsCheck::SIMDIntrinsicsCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), Std(Options.get("Std", "")),
      Suggest(Options.get("Suggest", false)) {}

void SIMDIntrinsicsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Std", Std);
  Options.store(Opts, "Suggest", Suggest);
}

void SIMDIntrinsicsCheck::registerMatchers(MatchFinder *Finder) {
  // If Std is not specified, infer it from the language options.
  // libcxx implementation backports it to C++11 std::experimental::simd.
  if (Std.empty())
    Std = getLangOpts().CPlusPlus20 ? "std" : "std::experimental";

  Finder->addMatcher(callExpr(callee(functionDecl(
                                  matchesName("^::(_mm_|_mm256_|_mm512_|vec_)"),
                                  isVectorFunction())),
                              unless(isExpansionInSystemHeader()))
                         .bind("call"),
                     this);
}

void SIMDIntrinsicsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  assert(Call != nullptr);
  const FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee)
    return;

  StringRef Old = Callee->getName();
  StringRef New;
  llvm::Triple::ArchType Arch =
      Result.Context->getTargetInfo().getTriple().getArch();

  // We warn or suggest if this SIMD intrinsic function has a std::simd
  // replacement.
  switch (Arch) {
  default:
    break;
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    New = trySuggestPpc(Old);
    break;
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    New = trySuggestX86(Old);
    break;
  }

  // We have found a std::simd replacement.
  if (!New.empty()) {
    // If Suggest is true, give a P0214 alternative, otherwise point it out it
    // is non-portable.
    if (Suggest) {
      static const llvm::Regex StdRegex("\\$std"), SimdRegex("\\$simd");
      diag(Call->getExprLoc(), "'%0' can be replaced by %1")
          << Old
          << SimdRegex.sub(SmallString<32>({Std, "::simd"}),
                           StdRegex.sub(Std, New));
    } else {
      diag("'%0' is a non-portable %1 intrinsic function")
          << Old << llvm::Triple::getArchTypeName(Arch);
    }
  }
}

} // namespace clang::tidy::portability
