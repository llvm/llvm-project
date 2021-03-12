//===- ASTSrcLocProcessor.h ---------------------------------*- C++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DUMPTOOL_ASTSRCLOCPROCESSOR_H
#define LLVM_CLANG_TOOLING_DUMPTOOL_ASTSRCLOCPROCESSOR_H

#include "APIData.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {

class CompilerInstance;

namespace tooling {

class ASTSrcLocProcessor : public ast_matchers::MatchFinder::MatchCallback {
public:
  explicit ASTSrcLocProcessor(StringRef JsonPath);

  std::unique_ptr<ASTConsumer> createASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File);

  void generate();

private:
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  llvm::StringMap<StringRef> ClassInheritance;
  llvm::StringMap<std::vector<StringRef>> ClassesInClade;
  llvm::StringMap<ClassData> ClassEntries;

  std::string JsonPath;
  std::unique_ptr<clang::ast_matchers::MatchFinder> Finder;
};

} // namespace tooling
} // namespace clang

#endif
