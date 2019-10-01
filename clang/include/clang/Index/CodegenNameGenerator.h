//===- CodegenNameGenerator.h - Codegen name generation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Determines the name that the symbol will get for code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_CODEGENNAMEGENERATOR_H
#define LLVM_CLANG_INDEX_CODEGENNAMEGENERATOR_H

#include "clang/Basic/LLVM.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
  class ASTContext;
  class Decl;

namespace index {

class CodegenNameGenerator {
public:
  explicit CodegenNameGenerator(ASTContext &Ctx);
  ~CodegenNameGenerator();

  /// \returns true on failure to produce a name for the given decl, false on
  /// success.
  bool writeName(const Decl *D, raw_ostream &OS);

  /// Version of \c writeName function that returns a string.
  std::string getName(const Decl *D);

  /// This can return multiple mangled names when applicable, e.g. for C++
  /// constructors/destructors.
  std::vector<std::string> getAllManglings(const Decl *D);

private:
  struct Implementation;
  std::unique_ptr<Implementation> Impl;
};

} // namespace index
} // namespace clang

#endif // LLVM_CLANG_INDEX_CODEGENNAMEGENERATOR_H
