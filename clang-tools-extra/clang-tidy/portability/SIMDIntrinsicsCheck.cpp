//===--- SIMDIntrinsicsCheck.cpp - clang-tidy------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SIMDIntrinsicsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Regex.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace portability {

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

static StringRef TrySuggestPPC(StringRef Name) {
  if (!Name.consume_front("vec_"))
    return {};

  static const llvm::StringMap<StringRef> Mapping{
    // [simd.alg]
    {"max", "$std::max"},
    {"min", "$std::min"},

    // [simd.binary]
    {"add", "operator+ on $simd objects"},
    {"sub", "operator- on $simd objects"},
    {"mul", "operator* on $simd objects"},
  };

  auto It = Mapping.find(Name);
  if (It != Mapping.end())
    return It->second;
  return {};
}

static StringRef TrySuggestX86(StringRef Name) {
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
      Suggest(Options.get("Suggest", 0) != 0) {}

void SIMDIntrinsicsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Std", "");
  Options.store(Opts, "Suggest", 0);
}

void SIMDIntrinsicsCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;
  // If Std is not specified, infer it from the language options.
  // libcxx implementation backports it to C++11 std::experimental::simd.
  if (Std.empty())
    Std = getLangOpts().CPlusPlus2a ? "std" : "std::experimental";

  Finder->addMatcher(callExpr(callee(functionDecl(allOf(
                                  matchesName("^::(_mm_|_mm256_|_mm512_|vec_)"),
                                  isVectorFunction()))),
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
    New = TrySuggestPPC(Old);
    break;
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    New = TrySuggestX86(Old);
    break;
  }

  // We have found a std::simd replacement.
  if (!New.empty()) {
    std::string Message;
    // If Suggest is true, give a P0214 alternative, otherwise point it out it
    // is non-portable.
    if (Suggest) {
      Message = (Twine("'") + Old + "' can be replaced by " + New).str();
      Message = llvm::Regex("\\$std").sub(Std, Message);
      Message =
          llvm::Regex("\\$simd").sub((Std.str() + "::simd").str(), Message);
    } else {
      Message = (Twine("'") + Old + "' is a non-portable " +
                 llvm::Triple::getArchTypeName(Arch) + " intrinsic function")
                    .str();
    }
    diag(Call->getExprLoc(), Message);
  }
}

} // namespace portability
} // namespace tidy
} // namespace clang
