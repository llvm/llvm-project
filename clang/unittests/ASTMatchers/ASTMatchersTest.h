//===- unittest/Tooling/ASTMatchersTest.h - Matcher tests helpers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ASTMATCHERS_ASTMATCHERSTEST_H
#define LLVM_CLANG_UNITTESTS_ASTMATCHERS_ASTMATCHERSTEST_H

#include "clang/AST/APValue.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Testing/TestClangConfig.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/FixIt.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace ast_matchers {

using clang::tooling::buildASTFromCodeWithArgs;
using clang::tooling::FileContentMappings;
using clang::tooling::FrontendActionFactory;
using clang::tooling::newFrontendActionFactory;
using clang::tooling::runToolOnCodeWithArgs;

class BoundNodesCallback {
public:
  virtual ~BoundNodesCallback() {}
  virtual bool run(const BoundNodes *BoundNodes, ASTContext *Context) = 0;
  virtual void onEndOfTranslationUnit() {}
};

// If 'FindResultVerifier' is not NULL, sets *Verified to the result of
// running 'FindResultVerifier' with the bound nodes as argument.
// If 'FindResultVerifier' is NULL, sets *Verified to true when Run is called.
class VerifyMatch : public MatchFinder::MatchCallback {
public:
  VerifyMatch(std::unique_ptr<BoundNodesCallback> FindResultVerifier,
              bool *Verified)
      : Verified(Verified), FindResultReviewer(std::move(FindResultVerifier)) {}

  void run(const MatchFinder::MatchResult &Result) override {
    if (FindResultReviewer != nullptr) {
      *Verified |= FindResultReviewer->run(&Result.Nodes, Result.Context);
    } else {
      *Verified = true;
    }
  }

  void onEndOfTranslationUnit() override {
    if (FindResultReviewer)
      FindResultReviewer->onEndOfTranslationUnit();
  }

private:
  bool *const Verified;
  const std::unique_ptr<BoundNodesCallback> FindResultReviewer;
};

inline ArrayRef<TestLanguage> langCxx11OrLater() {
  static const TestLanguage Result[] = {Lang_CXX11, Lang_CXX14, Lang_CXX17,
                                        Lang_CXX20, Lang_CXX23};
  return ArrayRef<TestLanguage>(Result);
}

inline ArrayRef<TestLanguage> langCxx14OrLater() {
  static const TestLanguage Result[] = {Lang_CXX14, Lang_CXX17, Lang_CXX20,
                                        Lang_CXX23};
  return ArrayRef<TestLanguage>(Result);
}

inline ArrayRef<TestLanguage> langCxx17OrLater() {
  static const TestLanguage Result[] = {Lang_CXX17, Lang_CXX20, Lang_CXX23};
  return ArrayRef<TestLanguage>(Result);
}

inline ArrayRef<TestLanguage> langCxx20OrLater() {
  static const TestLanguage Result[] = {Lang_CXX20, Lang_CXX23};
  return ArrayRef<TestLanguage>(Result);
}

inline ArrayRef<TestLanguage> langCxx23OrLater() {
  static const TestLanguage Result[] = {Lang_CXX23};
  return ArrayRef<TestLanguage>(Result);
}

template <typename T>
testing::AssertionResult matchesConditionally(
    const Twine &Code, const T &AMatcher, bool ExpectMatch,
    ArrayRef<std::string> CompileArgs,
    const FileContentMappings &VirtualMappedFiles = FileContentMappings(),
    StringRef Filename = "input.cc") {
  bool Found = false, DynamicFound = false;
  MatchFinder Finder;
  VerifyMatch VerifyFound(nullptr, &Found);
  Finder.addMatcher(AMatcher, &VerifyFound);
  VerifyMatch VerifyDynamicFound(nullptr, &DynamicFound);
  if (!Finder.addDynamicMatcher(AMatcher, &VerifyDynamicFound))
    return testing::AssertionFailure() << "Could not add dynamic matcher";
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  std::vector<std::string> Args = {
      // Some tests need rtti/exceptions on.
      "-frtti", "-fexceptions",
      // Ensure that tests specify the C++ standard version that they need.
      "-Werror=c++14-extensions", "-Werror=c++17-extensions",
      "-Werror=c++20-extensions"};
  // Append additional arguments at the end to allow overriding the default
  // choices that we made above.
  llvm::copy(CompileArgs, std::back_inserter(Args));
  if (llvm::find(Args, "-target") == Args.end()) {
    // Use an unknown-unknown triple so we don't instantiate the full system
    // toolchain.  On Linux, instantiating the toolchain involves stat'ing
    // large portions of /usr/lib, and this slows down not only this test, but
    // all other tests, via contention in the kernel.
    //
    // FIXME: This is a hack to work around the fact that there's no way to do
    // the equivalent of runToolOnCodeWithArgs without instantiating a full
    // Driver.  We should consider having a function, at least for tests, that
    // invokes cc1.
    Args.push_back("-target");
    Args.push_back("i386-unknown-unknown");
  }

  if (!runToolOnCodeWithArgs(
          Factory->create(), Code, Args, Filename, "clang-tool",
          std::make_shared<PCHContainerOperations>(), VirtualMappedFiles)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (Found != DynamicFound) {
    return testing::AssertionFailure()
           << "Dynamic match result (" << DynamicFound
           << ") does not match static result (" << Found << ")";
  }
  if (!Found && ExpectMatch) {
    return testing::AssertionFailure()
           << "Could not find match in \"" << Code << "\"";
  } else if (Found && !ExpectMatch) {
    return testing::AssertionFailure()
           << "Found unexpected match in \"" << Code << "\"";
  }
  return testing::AssertionSuccess();
}

template <typename T>
testing::AssertionResult
matchesConditionally(const Twine &Code, const T &AMatcher, bool ExpectMatch,
                     ArrayRef<TestLanguage> TestLanguages) {
  for (auto Lang : TestLanguages) {
    auto Result = matchesConditionally(
        Code, AMatcher, ExpectMatch, getCommandLineArgsForTesting(Lang),
        FileContentMappings(), getFilenameForTesting(Lang));
    if (!Result)
      return Result;
  }

  return testing::AssertionSuccess();
}

template <typename T>
testing::AssertionResult
matches(const Twine &Code, const T &AMatcher,
        ArrayRef<TestLanguage> TestLanguages = {Lang_CXX11}) {
  return matchesConditionally(Code, AMatcher, true, TestLanguages);
}

template <typename T>
testing::AssertionResult
notMatches(const Twine &Code, const T &AMatcher,
           ArrayRef<TestLanguage> TestLanguages = {Lang_CXX11}) {
  return matchesConditionally(Code, AMatcher, false, TestLanguages);
}

template <typename T>
testing::AssertionResult matchesObjC(const Twine &Code, const T &AMatcher,
                                     bool ExpectMatch = true) {
  return matchesConditionally(Code, AMatcher, ExpectMatch,
                              {"-fobjc-nonfragile-abi", "-Wno-objc-root-class",
                               "-fblocks", "-Wno-incomplete-implementation"},
                              FileContentMappings(), "input.m");
}

template <typename T>
testing::AssertionResult matchesC(const Twine &Code, const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true, {}, FileContentMappings(),
                              "input.c");
}

template <typename T>
testing::AssertionResult notMatchesObjC(const Twine &Code, const T &AMatcher) {
  return matchesObjC(Code, AMatcher, false);
}

// Function based on matchesConditionally with "-x cuda" argument added and
// small CUDA header prepended to the code string.
template <typename T>
testing::AssertionResult
matchesConditionallyWithCuda(const Twine &Code, const T &AMatcher,
                             bool ExpectMatch, llvm::StringRef CompileArg) {
  const std::string CudaHeader =
      "typedef unsigned int size_t;\n"
      "#define __constant__ __attribute__((constant))\n"
      "#define __device__ __attribute__((device))\n"
      "#define __global__ __attribute__((global))\n"
      "#define __host__ __attribute__((host))\n"
      "#define __shared__ __attribute__((shared))\n"
      "struct dim3 {"
      "  unsigned x, y, z;"
      "  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1)"
      "      : x(x), y(y), z(z) {}"
      "};"
      "typedef struct cudaStream *cudaStream_t;"
      "int cudaConfigureCall(dim3 gridSize, dim3 blockSize,"
      "                      size_t sharedSize = 0,"
      "                      cudaStream_t stream = 0);"
      "extern \"C\" unsigned __cudaPushCallConfiguration("
      "    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = "
      "0);";

  bool Found = false, DynamicFound = false;
  MatchFinder Finder;
  VerifyMatch VerifyFound(nullptr, &Found);
  Finder.addMatcher(AMatcher, &VerifyFound);
  VerifyMatch VerifyDynamicFound(nullptr, &DynamicFound);
  if (!Finder.addDynamicMatcher(AMatcher, &VerifyDynamicFound))
    return testing::AssertionFailure() << "Could not add dynamic matcher";
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.  Using an explicit
  // unknown-unknown triple is good for a large speedup, because it lets us
  // avoid constructing a full system triple.
  std::vector<std::string> Args = {
      "-xcuda",  "-fno-ms-extensions",     "--cuda-host-only",     "-nocudainc",
      "-target", "x86_64-unknown-unknown", std::string(CompileArg)};
  if (!runToolOnCodeWithArgs(Factory->create(), CudaHeader + Code, Args)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (Found != DynamicFound) {
    return testing::AssertionFailure()
           << "Dynamic match result (" << DynamicFound
           << ") does not match static result (" << Found << ")";
  }
  if (!Found && ExpectMatch) {
    return testing::AssertionFailure()
           << "Could not find match in \"" << Code << "\"";
  } else if (Found && !ExpectMatch) {
    return testing::AssertionFailure()
           << "Found unexpected match in \"" << Code << "\"";
  }
  return testing::AssertionSuccess();
}

template <typename T>
testing::AssertionResult matchesWithCuda(const Twine &Code, const T &AMatcher) {
  return matchesConditionallyWithCuda(Code, AMatcher, true, "-std=c++11");
}

template <typename T>
testing::AssertionResult notMatchesWithCuda(const Twine &Code,
                                            const T &AMatcher) {
  return matchesConditionallyWithCuda(Code, AMatcher, false, "-std=c++11");
}

template <typename T>
testing::AssertionResult matchesWithOpenMP(const Twine &Code,
                                           const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true, {"-fopenmp=libomp"});
}

template <typename T>
testing::AssertionResult notMatchesWithOpenMP(const Twine &Code,
                                              const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false, {"-fopenmp=libomp"});
}

template <typename T>
testing::AssertionResult matchesWithOpenMP51(const Twine &Code,
                                             const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true,
                              {"-fopenmp=libomp", "-fopenmp-version=51"});
}

template <typename T>
testing::AssertionResult notMatchesWithOpenMP51(const Twine &Code,
                                                const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false,
                              {"-fopenmp=libomp", "-fopenmp-version=51"});
}

template <typename T>
testing::AssertionResult matchAndVerifyResultConditionally(
    const Twine &Code, const T &AMatcher,
    std::unique_ptr<BoundNodesCallback> FindResultVerifier, bool ExpectResult,
    ArrayRef<std::string> Args = {}, StringRef Filename = "input.cc",
    const FileContentMappings &VirtualMappedFiles = {}) {
  bool VerifiedResult = false;
  MatchFinder Finder;
  VerifyMatch VerifyVerifiedResult(std::move(FindResultVerifier),
                                   &VerifiedResult);
  Finder.addMatcher(AMatcher, &VerifyVerifiedResult);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.  Using an explicit
  // unknown-unknown triple is good for a large speedup, because it lets us
  // avoid constructing a full system triple.
  std::vector<std::string> CompileArgs = {"-std=gnu++11", "-target",
                                          "i386-unknown-unknown"};
  // Append additional arguments at the end to allow overriding the default
  // choices that we made above.
  llvm::copy(Args, std::back_inserter(CompileArgs));

  if (!runToolOnCodeWithArgs(
          Factory->create(), Code, CompileArgs, Filename, "clang-tool",
          std::make_shared<PCHContainerOperations>(), VirtualMappedFiles)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (!VerifiedResult && ExpectResult) {
    return testing::AssertionFailure()
           << "Could not verify result in \"" << Code << "\"";
  } else if (VerifiedResult && !ExpectResult) {
    return testing::AssertionFailure()
           << "Verified unexpected result in \"" << Code << "\"";
  }

  VerifiedResult = false;
  SmallString<256> Buffer;
  std::unique_ptr<ASTUnit> AST(buildASTFromCodeWithArgs(
      Code.toStringRef(Buffer), CompileArgs, Filename, "clang-tool",
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(), VirtualMappedFiles));
  if (!AST.get())
    return testing::AssertionFailure()
           << "Parsing error in \"" << Code << "\" while building AST";
  Finder.matchAST(AST->getASTContext());
  if (!VerifiedResult && ExpectResult) {
    return testing::AssertionFailure()
           << "Could not verify result in \"" << Code << "\" with AST";
  } else if (VerifiedResult && !ExpectResult) {
    return testing::AssertionFailure()
           << "Verified unexpected result in \"" << Code << "\" with AST";
  }

  return testing::AssertionSuccess();
}

// FIXME: Find better names for these functions (or document what they
// do more precisely).
template <typename T>
testing::AssertionResult
matchAndVerifyResultTrue(const Twine &Code, const T &AMatcher,
                         std::unique_ptr<BoundNodesCallback> FindResultVerifier,
                         ArrayRef<std::string> Args = {},
                         StringRef Filename = "input.cc") {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, std::move(FindResultVerifier),
      /*ExpectResult=*/true, Args, Filename);
}

template <typename T>
testing::AssertionResult matchAndVerifyResultFalse(
    const Twine &Code, const T &AMatcher,
    std::unique_ptr<BoundNodesCallback> FindResultVerifier,
    ArrayRef<std::string> Args = {}, StringRef Filename = "input.cc") {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, std::move(FindResultVerifier),
      /*ExpectResult=*/false, Args, Filename);
}

// Implements a run method that returns whether BoundNodes contains a
// Decl bound to Id that can be dynamically cast to T.
// Optionally checks that the check succeeded a specific number of times.
template <typename T> class VerifyIdIsBoundTo : public BoundNodesCallback {
public:
  // Create an object that checks that a node of type \c T was bound to \c Id.
  // Does not check for a certain number of matches.
  explicit VerifyIdIsBoundTo(llvm::StringRef Id)
      : Id(std::string(Id)), ExpectedCount(-1), Count(0) {}

  // Create an object that checks that a node of type \c T was bound to \c Id.
  // Checks that there were exactly \c ExpectedCount matches.
  VerifyIdIsBoundTo(llvm::StringRef Id, int ExpectedCount)
      : Id(std::string(Id)), ExpectedCount(ExpectedCount), Count(0) {}

  // Create an object that checks that a node of type \c T was bound to \c Id.
  // Checks that there was exactly one match with the name \c ExpectedName.
  // Note that \c T must be a NamedDecl for this to work.
  VerifyIdIsBoundTo(llvm::StringRef Id, llvm::StringRef ExpectedName,
                    int ExpectedCount = 1)
      : Id(std::string(Id)), ExpectedCount(ExpectedCount), Count(0),
        ExpectedName(std::string(ExpectedName)) {}

  void onEndOfTranslationUnit() override {
    if (ExpectedCount != -1) {
      EXPECT_EQ(ExpectedCount, Count);
    }
    if (!ExpectedName.empty()) {
      EXPECT_EQ(ExpectedName, Name);
    }
    Count = 0;
    Name.clear();
  }

  ~VerifyIdIsBoundTo() override {
    EXPECT_EQ(0, Count);
    EXPECT_EQ("", Name);
  }

  bool run(const BoundNodes *Nodes, ASTContext * /*Context*/) override {
    const BoundNodes::IDToNodeMap &M = Nodes->getMap();
    if (Nodes->getNodeAs<T>(Id)) {
      ++Count;
      if (const NamedDecl *Named = Nodes->getNodeAs<NamedDecl>(Id)) {
        Name = Named->getNameAsString();
      } else if (const NestedNameSpecifier *NNS =
                     Nodes->getNodeAs<NestedNameSpecifier>(Id)) {
        llvm::raw_string_ostream OS(Name);
        NNS->print(OS, PrintingPolicy(LangOptions()));
      }
      BoundNodes::IDToNodeMap::const_iterator I = M.find(Id);
      EXPECT_NE(M.end(), I);
      if (I != M.end()) {
        EXPECT_EQ(Nodes->getNodeAs<T>(Id), I->second.get<T>());
      }
      return true;
    }
    EXPECT_TRUE(M.count(Id) == 0 ||
                M.find(Id)->second.template get<T>() == nullptr);
    return false;
  }

private:
  const std::string Id;
  const int ExpectedCount;
  int Count;
  const std::string ExpectedName;
  std::string Name;
};

namespace detail {
template <typename T>
using hasDump_t = decltype(std::declval<const T &>().dump());
template <typename T>
constexpr bool hasDump = llvm::is_detected<hasDump_t, T>::value;

template <typename T>
using hasGetSourceRange_t =
    decltype(std::declval<const T &>().getSourceRange());
template <typename T>
constexpr bool hasGetSourceRange =
    llvm::is_detected<hasGetSourceRange_t, T>::value;

template <typename T, std::enable_if_t<hasGetSourceRange<T>, bool> = true>
std::optional<std::string> getText(const T *const Node,
                                   const ASTContext &Context) {
  return tooling::fixit::getText(*Node, Context).str();
}
inline std::optional<std::string> getText(const Attr *const Attribute,
                                          const ASTContext &) {
  return Attribute->getSpelling();
}
inline std::optional<std::string> getText(const void *const,
                                          const ASTContext &) {
  return std::nullopt;
}

template <typename T>
auto getSourceRange(const T *const Node)
    -> std::optional<decltype(Node->getSourceRange())> {
  return Node->getSourceRange();
}
inline std::optional<SourceRange> getSourceRange(const void *const) {
  return std::nullopt;
}

template <typename T>
auto getLocation(const T *const Node)
    -> std::optional<decltype(Node->getLocation())> {
  return Node->getLocation();
}
inline std::optional<SourceLocation> getLocation(const void *const) {
  return std::nullopt;
}

template <typename T>
auto getBeginLoc(const T *const Node)
    -> std::optional<decltype(Node->getBeginLoc())> {
  return Node->getBeginLoc();
}
inline std::optional<SourceLocation> getBeginLoc(const void *const) {
  return std::nullopt;
}

inline std::optional<SourceLocation>
getLocOfTagDeclFromType(const Type *const Node) {
  if (Node->isArrayType())
    if (const auto *const AType = Node->getPointeeOrArrayElementType()) {
      return getLocOfTagDeclFromType(AType);
    }
  if (const auto *const TDecl = Node->getAsTagDecl()) {
    return TDecl->getLocation();
  }
  return std::nullopt;
}
inline std::optional<SourceLocation>
getLocOfTagDeclFromType(const void *const Node) {
  return std::nullopt;
}

template <typename T>
auto getExprLoc(const T *const Node)
    -> std::optional<decltype(Node->getBeginLoc())> {
  return Node->getBeginLoc();
}
inline std::optional<SourceLocation> getExprLoc(const void *const) {
  return std::nullopt;
}

// Provides a string for test failures to show what was matched.
template <typename T>
static std::optional<std::string>
getNodeDescription(const T *const Node, const ASTContext *const Context) {
  if constexpr (std::is_same_v<T, QualType>) {
    return Node->getAsString();
  }
  if constexpr (std::is_base_of_v<NamedDecl, T>) {
    return Node->getNameAsString();
  }
  if constexpr (std::is_base_of_v<Type, T>) {
    return QualType{Node, 0}.getAsString();
  }

  return detail::getText(Node, *Context);
}

template <typename T>
bool shouldIgnoreNode(const T *const Node, const ASTContext &Context) {
  if (const auto Range = detail::getSourceRange(Node); Range.has_value()) {
    if (Range->isInvalid() ||
        !Context.getSourceManager().isInMainFile(Range->getBegin()))
      return true;
  } else if (const auto Loc = detail::getExprLoc(Node); Loc.has_value()) {
    if (Loc->isInvalid() || !Context.getSourceManager().isInMainFile(*Loc))
      return true;
  } else if (const auto Loc = detail::getLocation(Node); Loc.has_value()) {
    if (Loc->isInvalid() || !Context.getSourceManager().isInMainFile(*Loc))
      return true;
  } else if (const auto Loc = detail::getBeginLoc(Node); Loc.has_value()) {
    if (Loc->isInvalid() || !Context.getSourceManager().isInMainFile(*Loc))
      return true;
  } else if (const auto Loc = detail::getLocOfTagDeclFromType(Node);
             Loc.has_value()) {
    if (Loc->isInvalid() || !Context.getSourceManager().isInMainFile(*Loc))
      return true;
  }
  return false;
}
} // namespace detail

enum class MatchKind {
  Code,
  Name,
  TypeStr,
};

inline llvm::StringRef toString(const MatchKind Kind) {
  switch (Kind) {
  case MatchKind::Code:
    return "Code";
  case MatchKind::Name:
    return "Name";
  case MatchKind::TypeStr:
    return "TypeStr";
  }
  llvm_unreachable("Unhandled MatchKind");
}

template <typename T> class VerifyBoundNodeMatch : public BoundNodesCallback {
public:
  class Match {
  public:
    Match(const MatchKind Kind, std::string MatchString,
          const size_t MatchCount = 1)
        : Kind(Kind), MatchString(std::move(MatchString)),
          RemainingMatches(MatchCount) {}

    bool shouldRemoveMatched(const T *const Node) {
      --RemainingMatches;
      return RemainingMatches == 0U;
    }

    template <typename U>
    static std::optional<std::string>
    getMatchText(const U *const Node, const ASTContext &Context,
                 const MatchKind Kind, const bool EmitFailures = true) {
      if constexpr (std::is_same_v<U, NestedNameSpecifier>) {
        if (const IdentifierInfo *const Info = Node->getAsIdentifier())
          return Info->getName().str();
        if (const NamespaceDecl *const NS = Node->getAsNamespace())
          return getMatchText(NS, Context, Kind, EmitFailures);
        if (const NamespaceAliasDecl *const Alias = Node->getAsNamespaceAlias())
          return getMatchText(Alias, Context, Kind, EmitFailures);
        if (const CXXRecordDecl *const RDecl = Node->getAsRecordDecl())
          return getMatchText(RDecl, Context, Kind, EmitFailures);
        if (const Type *const RDecl = Node->getAsType())
          return getMatchText(RDecl, Context, Kind, EmitFailures);
      }

      switch (Kind) {
      case MatchKind::Code:
        return detail::getText(Node, Context);
      case MatchKind::Name:
        return getNameText(Node, EmitFailures);
      case MatchKind::TypeStr:
        return getTypeStrText(Node, EmitFailures);
      }
    }

    bool isMatch(const T *const Node, const ASTContext &Context) const {
      if (const auto OptMatchText = getMatchText(Node, Context, Kind))
        return *OptMatchText == MatchString;

      return false;
    }

    std::string getAsString() const {
      return llvm::formatv("MatchKind: {0}, MatchString: '{1}', "
                           "RemainingMatches: {2}",
                           toString(Kind), MatchString, RemainingMatches)
          .str();
    }

    MatchKind getMatchKind() const { return Kind; }
    llvm::StringRef getMatchString() const { return MatchString; }

  private:
    template <typename U>
    static std::optional<std::string>
    getNameText(const U *const Node, const bool EmitFailures = true) {
      if constexpr (std::is_base_of_v<Decl, U>) {
        if (const auto *const NDecl = llvm::dyn_cast<NamedDecl>(Node))
          return NDecl->getNameAsString();
        if (EmitFailures)
          ADD_FAILURE() << "'MatchKind::Name' requires 'U' to be a'NamedDecl'.";
      }

      if (EmitFailures)
        ADD_FAILURE() << "'MatchKind::Name' requires 'U' to be a "
                         "'NamedDecl', but 'U' is not derived from 'Decl'.";
      return std::nullopt;
    }

    template <typename U>
    static std::optional<std::string>
    getTypeStrText(const U *const Node, const bool EmitFailures = true) {
      if constexpr (std::is_base_of_v<Type, U>)
        return QualType(Node, 0).getAsString();
      if constexpr (std::is_base_of_v<Decl, U>) {
        if (const auto *const TDecl = llvm::dyn_cast<TypeDecl>(Node))
          return getTypeStrText(TDecl->getTypeForDecl());
        if (const auto *const VDecl = llvm::dyn_cast<ValueDecl>(Node))
          return VDecl->getType().getAsString();
      }
      if (EmitFailures)
        ADD_FAILURE() << "Match kind is 'TypeStr', but node of type 'U' is "
                         "not handled.";
      return std::nullopt;
    }

    MatchKind Kind;
    std::string MatchString;
    size_t RemainingMatches;
  };

  VerifyBoundNodeMatch(std::string Id, std::vector<Match> Matches)
      : Id(std::move(Id)), ExpectedMatches(std::move(Matches)),
        Matches(ExpectedMatches) {}

  bool run(const BoundNodes *const Nodes, ASTContext *const Context) override {
    const auto *const Node = Nodes->getNodeAs<T>(Id);
    if (Node == nullptr) {
      ADD_FAILURE() << "Expected Id '" << Id << "' to be bound to 'T'.";
      return true;
    }

    if constexpr (std::is_base_of_v<Decl, T>)
      if (const auto *const NDecl = llvm::dyn_cast<NamedDecl>(Node))
        if (const auto *Identifier = NDecl->getIdentifier();
            Identifier != nullptr && Identifier->getBuiltinID() > 0)
          return true;

    if (detail::shouldIgnoreNode(Node, *Context)) {
      return false;
    }

    const auto Iter = llvm::find_if(Matches, [Node, Context](const Match &M) {
      return M.isMatch(Node, *Context);
    });
    if (Iter == Matches.end()) {
      const auto NodeText =
          detail::getNodeDescription(Node, Context).value_or("<unknown>");
      const auto IsMultilineNodeText = NodeText.find('\n') != std::string::npos;
      ADD_FAILURE() << "No match of node '" << (IsMultilineNodeText ? "\n" : "")
                    << NodeText << (IsMultilineNodeText ? "\n" : "")
                    << "' was expected.\n"
                    << "No match with remaining matches:"
                    << getMatchComparisonText(Matches, Node, *Context)
                    << "Match strings of Node for possible intended matches:"
                    << getPossibleMatchStrings(Node, *Context)
                    << "Already found matches:"
                    << getMatchesAsString(FoundMatches) << "Expected matches:"
                    << getMatchesAsString(ExpectedMatches);
      if constexpr (detail::hasDump<T>)
        Node->dump();
      return true;
    }

    if (Iter->shouldRemoveMatched(Node)) {
      FoundMatches.push_back(*Iter);
      Matches.erase(Iter);
    }

    return true;
  }

  void onEndOfTranslationUnit() override {
    if (!ExpectedMatches.empty() && Matches.size() == ExpectedMatches.size())
      ADD_FAILURE() << "No matches were found.\n"
                    << "Expected matches:"
                    << getMatchesAsString(ExpectedMatches);
    else
      EXPECT_TRUE(Matches.empty())
          << "Not all expected matches were found.\n"
          << "Remaining matches:" << getMatchesAsString(Matches)
          << "Already found matches:" << getMatchesAsString(FoundMatches)
          << "Expected matches:" << getMatchesAsString(ExpectedMatches);

    Matches = ExpectedMatches;
    FoundMatches.clear();

    EXPECT_TRUE(FoundMatches.empty());
  }

private:
  static std::string getMatchesAsString(const std::vector<Match> &Matches) {
    if (Matches.empty())
      return " none\n";
    std::string FormattedMatches{"\n"};
    for (const Match &M : Matches)
      FormattedMatches += "\t" + M.getAsString() + ",\n";

    return FormattedMatches;
  }
  static std::string getMatchComparisonText(const std::vector<Match> &Matches,
                                            const T *const Node,
                                            const ASTContext &Context) {
    if (Matches.empty())
      return " none\n";
    std::string MatchStrings{"\n"};
    for (const Match &M : Matches)
      MatchStrings += llvm::formatv(
          "\tMatchKind: {0}: '{1}' vs '{2}',\n", toString(M.getMatchKind()),
          Match::getMatchText(Node, Context, M.getMatchKind(), false)
              .value_or("<unknown>"),
          M.getMatchString());

    return MatchStrings;
  }

  static std::string getPossibleMatchStrings(const T *Node,
                                             const ASTContext &Context) {
    std::string MatchStrings{"\n"};
    for (const auto Kind :
         {MatchKind::Code, MatchKind::Name, MatchKind::TypeStr})
      MatchStrings +=
          llvm::formatv("\tMatchKind:  {0}: '{1}',\n", toString(Kind),
                        Match::getMatchText(Node, Context, Kind, false)
                            .value_or("<unknown>"))
              .str();
    return MatchStrings;
  }

  const std::string Id;
  const std::vector<Match> ExpectedMatches;
  std::vector<Match> Matches;
  std::vector<Match> FoundMatches{};
};

class ASTMatchersTest : public ::testing::Test,
                        public ::testing::WithParamInterface<TestClangConfig> {
protected:
  template <typename T>
  testing::AssertionResult matches(const Twine &Code, const T &AMatcher) {
    const TestClangConfig &TestConfig = GetParam();
    return clang::ast_matchers::matchesConditionally(
        Code, AMatcher, /*ExpectMatch=*/true, TestConfig.getCommandLineArgs(),
        FileContentMappings(), getFilenameForTesting(TestConfig.Language));
  }

  template <typename T>
  testing::AssertionResult notMatches(const Twine &Code, const T &AMatcher) {
    const TestClangConfig &TestConfig = GetParam();
    return clang::ast_matchers::matchesConditionally(
        Code, AMatcher, /*ExpectMatch=*/false, TestConfig.getCommandLineArgs(),
        FileContentMappings(), getFilenameForTesting(TestConfig.Language));
  }
};

class ASTMatchersDocTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<TestClangConfig> {
protected:
  template <typename T>
  testing::AssertionResult
  matches(const Twine &Code, const T &AMatcher,
          std::unique_ptr<BoundNodesCallback> FindResultVerifier,
          const ArrayRef<std::string> CompileArgs = {},
          const FileContentMappings &VirtualMappedFiles = {}) {
    const TestClangConfig &TestConfig = GetParam();

    auto Args = TestConfig.getCommandLineArgs();
    Args.insert(Args.end(), CompileArgs.begin(), CompileArgs.end());
    return clang::ast_matchers::matchAndVerifyResultConditionally(
        Code, AMatcher, std::move(FindResultVerifier), /*ExpectMatch=*/true,
        Args, getFilenameForTesting(TestConfig.Language), VirtualMappedFiles);
  }

  template <typename T>
  testing::AssertionResult
  notMatches(const Twine &Code, const T &AMatcher,
             std::unique_ptr<BoundNodesCallback> FindResultVerifier,
             const ArrayRef<std::string> CompileArgs = {},
             const FileContentMappings &VirtualMappedFiles = {}) {
    const TestClangConfig &TestConfig = GetParam();

    auto Args = TestConfig.getCommandLineArgs();
    Args.insert(Args.begin(), CompileArgs.begin(), CompileArgs.end());
    return clang::ast_matchers::matchAndVerifyResultConditionally(
        Code, AMatcher, std::move(FindResultVerifier), /*ExpectMatch=*/false,
        Args, getFilenameForTesting(TestConfig.Language), VirtualMappedFiles);
  }
};
} // namespace ast_matchers
} // namespace clang

#endif // LLVM_CLANG_UNITTESTS_AST_MATCHERS_AST_MATCHERS_TEST_H
