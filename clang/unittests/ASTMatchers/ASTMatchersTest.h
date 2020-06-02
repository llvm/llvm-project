//===- unittest/Tooling/ASTMatchersTest.h - Matcher tests helpers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ASTMATCHERS_ASTMATCHERSTEST_H
#define LLVM_CLANG_UNITTESTS_ASTMATCHERS_ASTMATCHERSTEST_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

using clang::tooling::buildASTFromCodeWithArgs;
using clang::tooling::newFrontendActionFactory;
using clang::tooling::runToolOnCodeWithArgs;
using clang::tooling::FrontendActionFactory;
using clang::tooling::FileContentMappings;

class BoundNodesCallback {
public:
  virtual ~BoundNodesCallback() {}
  virtual bool run(const BoundNodes *BoundNodes) = 0;
  virtual bool run(const BoundNodes *BoundNodes, ASTContext *Context) = 0;
  virtual void onEndOfTranslationUnit() {}
};

// If 'FindResultVerifier' is not NULL, sets *Verified to the result of
// running 'FindResultVerifier' with the bound nodes as argument.
// If 'FindResultVerifier' is NULL, sets *Verified to true when Run is called.
class VerifyMatch : public MatchFinder::MatchCallback {
public:
  VerifyMatch(std::unique_ptr<BoundNodesCallback> FindResultVerifier, bool *Verified)
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

enum class LanguageMode {
  Cxx11,
  Cxx14,
  Cxx17,
  Cxx2a,
  Cxx11OrLater,
  Cxx14OrLater,
  Cxx17OrLater,
  Cxx2aOrLater
};

template <typename T>
testing::AssertionResult matchesConditionally(
    const Twine &Code, const T &AMatcher, bool ExpectMatch,
    llvm::ArrayRef<llvm::StringRef> CompileArgs,
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
  // Some tests need rtti/exceptions on.  Use an unknown-unknown triple so we
  // don't instantiate the full system toolchain.  On Linux, instantiating the
  // toolchain involves stat'ing large portions of /usr/lib, and this slows down
  // not only this test, but all other tests, via contention in the kernel.
  //
  // FIXME: This is a hack to work around the fact that there's no way to do the
  // equivalent of runToolOnCodeWithArgs without instantiating a full Driver.
  // We should consider having a function, at least for tests, that invokes cc1.
  std::vector<std::string> Args(CompileArgs.begin(), CompileArgs.end());
  Args.insert(Args.end(), {"-frtti", "-fexceptions",
                           "-target", "i386-unknown-unknown"});
  if (!runToolOnCodeWithArgs(
          Factory->create(), Code, Args, Filename, "clang-tool",
          std::make_shared<PCHContainerOperations>(), VirtualMappedFiles)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (Found != DynamicFound) {
    return testing::AssertionFailure() << "Dynamic match result ("
                                       << DynamicFound
                                       << ") does not match static result ("
                                       << Found << ")";
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
testing::AssertionResult matchesConditionally(
    const Twine &Code, const T &AMatcher, bool ExpectMatch,
    llvm::StringRef CompileArg,
    const FileContentMappings &VirtualMappedFiles = FileContentMappings(),
    StringRef Filename = "input.cc") {
  return matchesConditionally(Code, AMatcher, ExpectMatch,
                              llvm::makeArrayRef(CompileArg),
                              VirtualMappedFiles, Filename);
}

template <typename T>
testing::AssertionResult
matchesConditionally(const Twine &Code, const T &AMatcher, bool ExpectMatch,
                     const LanguageMode &Mode) {
  std::vector<LanguageMode> LangModes;
  switch (Mode) {
  case LanguageMode::Cxx11:
  case LanguageMode::Cxx14:
  case LanguageMode::Cxx17:
  case LanguageMode::Cxx2a:
    LangModes = {Mode};
    break;
  case LanguageMode::Cxx11OrLater:
    LangModes = {LanguageMode::Cxx11, LanguageMode::Cxx14, LanguageMode::Cxx17,
                 LanguageMode::Cxx2a};
    break;
  case LanguageMode::Cxx14OrLater:
    LangModes = {LanguageMode::Cxx14, LanguageMode::Cxx17, LanguageMode::Cxx2a};
    break;
  case LanguageMode::Cxx17OrLater:
    LangModes = {LanguageMode::Cxx17, LanguageMode::Cxx2a};
    break;
  case LanguageMode::Cxx2aOrLater:
    LangModes = {LanguageMode::Cxx2a};
  }

  for (auto Mode : LangModes) {
    StringRef LangModeArg;
    switch (Mode) {
    case LanguageMode::Cxx11:
      LangModeArg = "-std=c++11";
      break;
    case LanguageMode::Cxx14:
      LangModeArg = "-std=c++14";
      break;
    case LanguageMode::Cxx17:
      LangModeArg = "-std=c++17";
      break;
    case LanguageMode::Cxx2a:
      LangModeArg = "-std=c++2a";
      break;
    default:
      llvm_unreachable("Invalid language mode");
    }

    auto Result = matchesConditionally(Code, AMatcher, ExpectMatch,
                                       {LangModeArg, "-Werror=c++14-extensions",
                                        "-Werror=c++17-extensions",
                                        "-Werror=c++20-extensions"});
    if (!Result)
      return Result;
  }

  return testing::AssertionSuccess();
}

template <typename T>
testing::AssertionResult
matches(const Twine &Code, const T &AMatcher,
        const LanguageMode &Mode = LanguageMode::Cxx11) {
  return matchesConditionally(Code, AMatcher, true, Mode);
}

template <typename T>
testing::AssertionResult
notMatches(const Twine &Code, const T &AMatcher,
           const LanguageMode &Mode = LanguageMode::Cxx11) {
  return matchesConditionally(Code, AMatcher, false, Mode);
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
  return matchesConditionally(Code, AMatcher, true, "", FileContentMappings(),
                              "input.c");
}

template <typename T>
testing::AssertionResult matchesC99(const Twine &Code, const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true, "-std=c99",
                              FileContentMappings(), "input.c");
}

template <typename T>
testing::AssertionResult notMatchesC(const Twine &Code, const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false, "", FileContentMappings(),
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
      "    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = 0);";

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
  if (!runToolOnCodeWithArgs(Factory->create(),
                             CudaHeader + Code, Args)) {
    return testing::AssertionFailure() << "Parsing error in \"" << Code << "\"";
  }
  if (Found != DynamicFound) {
    return testing::AssertionFailure() << "Dynamic match result ("
                                       << DynamicFound
                                       << ") does not match static result ("
                                       << Found << ")";
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
  return matchesConditionally(Code, AMatcher, true, "-fopenmp=libomp");
}

template <typename T>
testing::AssertionResult notMatchesWithOpenMP(const Twine &Code,
                                              const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false, "-fopenmp=libomp");
}

template <typename T>
testing::AssertionResult matchAndVerifyResultConditionally(
    const Twine &Code, const T &AMatcher,
    std::unique_ptr<BoundNodesCallback> FindResultVerifier, bool ExpectResult) {
  bool VerifiedResult = false;
  MatchFinder Finder;
  VerifyMatch VerifyVerifiedResult(std::move(FindResultVerifier), &VerifiedResult);
  Finder.addMatcher(AMatcher, &VerifyVerifiedResult);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.  Using an explicit
  // unknown-unknown triple is good for a large speedup, because it lets us
  // avoid constructing a full system triple.
  std::vector<std::string> Args = {"-std=gnu++98", "-target",
                                   "i386-unknown-unknown"};
  if (!runToolOnCodeWithArgs(Factory->create(), Code, Args)) {
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
  std::unique_ptr<ASTUnit> AST(
      buildASTFromCodeWithArgs(Code.toStringRef(Buffer), Args));
  if (!AST.get())
    return testing::AssertionFailure() << "Parsing error in \"" << Code
                                       << "\" while building AST";
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
testing::AssertionResult matchAndVerifyResultTrue(
    const Twine &Code, const T &AMatcher,
    std::unique_ptr<BoundNodesCallback> FindResultVerifier) {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, std::move(FindResultVerifier), true);
}

template <typename T>
testing::AssertionResult matchAndVerifyResultFalse(
    const Twine &Code, const T &AMatcher,
    std::unique_ptr<BoundNodesCallback> FindResultVerifier) {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, std::move(FindResultVerifier), false);
}

// Implements a run method that returns whether BoundNodes contains a
// Decl bound to Id that can be dynamically cast to T.
// Optionally checks that the check succeeded a specific number of times.
template <typename T>
class VerifyIdIsBoundTo : public BoundNodesCallback {
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

  bool run(const BoundNodes *Nodes) override {
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

  bool run(const BoundNodes *Nodes, ASTContext *Context) override {
    return run(Nodes);
  }

private:
  const std::string Id;
  const int ExpectedCount;
  int Count;
  const std::string ExpectedName;
  std::string Name;
};

} // namespace ast_matchers
} // namespace clang

#endif  // LLVM_CLANG_UNITTESTS_AST_MATCHERS_AST_MATCHERS_TEST_H
