//===- unittest/Tooling/ASTMatchersTest.h - Matcher tests helpers ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  VerifyMatch(BoundNodesCallback *FindResultVerifier, bool *Verified)
      : Verified(Verified), FindResultReviewer(FindResultVerifier) {}

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
  BoundNodesCallback *const FindResultReviewer;
};

template <typename T>
testing::AssertionResult matchesConditionally(
    const std::string &Code, const T &AMatcher, bool ExpectMatch,
    llvm::StringRef CompileArg,
    const FileContentMappings &VirtualMappedFiles = FileContentMappings(),
    const std::string &Filename = "input.cc") {
  bool Found = false, DynamicFound = false;
  MatchFinder Finder;
  VerifyMatch VerifyFound(nullptr, &Found);
  Finder.addMatcher(AMatcher, &VerifyFound);
  VerifyMatch VerifyDynamicFound(nullptr, &DynamicFound);
  if (!Finder.addDynamicMatcher(AMatcher, &VerifyDynamicFound))
    return testing::AssertionFailure() << "Could not add dynamic matcher";
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.
  std::vector<std::string> Args;
  Args.push_back(CompileArg);
  // Some tests need rtti/exceptions on
  Args.push_back("-frtti");
  Args.push_back("-fexceptions");
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
testing::AssertionResult matches(const std::string &Code, const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true, "-std=c++11");
}

template <typename T>
testing::AssertionResult notMatches(const std::string &Code,
                                    const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false, "-std=c++11");
}

template <typename T>
testing::AssertionResult matchesObjC(const std::string &Code,
                                     const T &AMatcher) {
  return matchesConditionally(
    Code, AMatcher, true,
    "", FileContentMappings(), "input.m");
}

template <typename T>
testing::AssertionResult matchesC(const std::string &Code, const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, true, "", FileContentMappings(),
                              "input.c");
}

template <typename T>
testing::AssertionResult notMatchesC(const std::string &Code,
                                     const T &AMatcher) {
  return matchesConditionally(Code, AMatcher, false, "", FileContentMappings(),
                              "input.c");
}

template <typename T>
testing::AssertionResult notMatchesObjC(const std::string &Code,
                                     const T &AMatcher) {
  return matchesConditionally(
    Code, AMatcher, false,
    "", FileContentMappings(), "input.m");
}


// Function based on matchesConditionally with "-x cuda" argument added and
// small CUDA header prepended to the code string.
template <typename T>
testing::AssertionResult matchesConditionallyWithCuda(
    const std::string &Code, const T &AMatcher, bool ExpectMatch,
    llvm::StringRef CompileArg) {
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
      "                      cudaStream_t stream = 0);";

  bool Found = false, DynamicFound = false;
  MatchFinder Finder;
  VerifyMatch VerifyFound(nullptr, &Found);
  Finder.addMatcher(AMatcher, &VerifyFound);
  VerifyMatch VerifyDynamicFound(nullptr, &DynamicFound);
  if (!Finder.addDynamicMatcher(AMatcher, &VerifyDynamicFound))
    return testing::AssertionFailure() << "Could not add dynamic matcher";
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.
  std::vector<std::string> Args;
  Args.push_back("-xcuda");
  Args.push_back("-fno-ms-extensions");
  Args.push_back("--cuda-host-only");
  Args.push_back("-nocudainc");
  Args.push_back(CompileArg);
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
testing::AssertionResult matchesWithCuda(const std::string &Code,
                                         const T &AMatcher) {
  return matchesConditionallyWithCuda(Code, AMatcher, true, "-std=c++11");
}

template <typename T>
testing::AssertionResult notMatchesWithCuda(const std::string &Code,
                                    const T &AMatcher) {
  return matchesConditionallyWithCuda(Code, AMatcher, false, "-std=c++11");
}

template <typename T>
testing::AssertionResult
matchAndVerifyResultConditionally(const std::string &Code, const T &AMatcher,
                                  BoundNodesCallback *FindResultVerifier,
                                  bool ExpectResult) {
  std::unique_ptr<BoundNodesCallback> ScopedVerifier(FindResultVerifier);
  bool VerifiedResult = false;
  MatchFinder Finder;
  VerifyMatch VerifyVerifiedResult(FindResultVerifier, &VerifiedResult);
  Finder.addMatcher(AMatcher, &VerifyVerifiedResult);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  // Some tests use typeof, which is a gnu extension.
  std::vector<std::string> Args(1, "-std=gnu++98");
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
  std::unique_ptr<ASTUnit> AST(buildASTFromCodeWithArgs(Code, Args));
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
testing::AssertionResult
matchAndVerifyResultTrue(const std::string &Code, const T &AMatcher,
                         BoundNodesCallback *FindResultVerifier) {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, FindResultVerifier, true);
}

template <typename T>
testing::AssertionResult
matchAndVerifyResultFalse(const std::string &Code, const T &AMatcher,
                          BoundNodesCallback *FindResultVerifier) {
  return matchAndVerifyResultConditionally(
      Code, AMatcher, FindResultVerifier, false);
}

} // end namespace ast_matchers
} // end namespace clang

#endif  // LLVM_CLANG_UNITTESTS_AST_MATCHERS_AST_MATCHERS_TEST_H
