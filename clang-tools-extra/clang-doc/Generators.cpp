//===---- Generator.cpp - Generator Registry ---------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Generators.h"

LLVM_INSTANTIATE_REGISTRY(clang::doc::GeneratorRegistry)

namespace clang {
namespace doc {

llvm::Expected<std::unique_ptr<Generator>>
findGeneratorByName(llvm::StringRef Format) {
  for (auto I = GeneratorRegistry::begin(), E = GeneratorRegistry::end();
       I != E; ++I) {
    if (I->getName() != Format)
      continue;
    return I->instantiate();
  }
  return llvm::make_error<llvm::StringError>("Can't find generator: " + Format,
                                             llvm::inconvertibleErrorCode());
}

// This anchor is used to force the linker to link in the generated object file
// and thus register the generators.
extern volatile int YAMLGeneratorAnchorSource;
extern volatile int MDGeneratorAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED YAMLGeneratorAnchorDest =
    YAMLGeneratorAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED MDGeneratorAnchorDest =
    MDGeneratorAnchorSource;

} // namespace doc
} // namespace clang
