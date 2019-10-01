//===-- Generators.h - ClangDoc Generator ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Generator classes for converting declaration information into documentation
// in a specified format.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_GENERATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_GENERATOR_H

#include "Representation.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"

namespace clang {
namespace doc {

// Abstract base class for generators.
// This is expected to be implemented and exposed via the GeneratorRegistry.
class Generator {
public:
  virtual ~Generator() = default;

  // Write out the decl info in the specified format.
  virtual llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS) = 0;
};

typedef llvm::Registry<Generator> GeneratorRegistry;

llvm::Expected<std::unique_ptr<Generator>>
findGeneratorByName(llvm::StringRef Format);

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_GENERATOR_H
