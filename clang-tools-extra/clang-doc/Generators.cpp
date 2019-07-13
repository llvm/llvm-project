//===-- Generators.cpp - Generator Registry ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Enum conversion

std::string getAccess(AccessSpecifier AS) {
  switch (AS) {
  case AccessSpecifier::AS_public:
    return "public";
  case AccessSpecifier::AS_protected:
    return "protected";
  case AccessSpecifier::AS_private:
    return "private";
  case AccessSpecifier::AS_none:
    return {};
  }
  llvm_unreachable("Unknown AccessSpecifier");
}

std::string getTagType(TagTypeKind AS) {
  switch (AS) {
  case TagTypeKind::TTK_Class:
    return "class";
  case TagTypeKind::TTK_Union:
    return "union";
  case TagTypeKind::TTK_Interface:
    return "interface";
  case TagTypeKind::TTK_Struct:
    return "struct";
  case TagTypeKind::TTK_Enum:
    return "enum";
  }
  llvm_unreachable("Unknown TagTypeKind");
}

// This anchor is used to force the linker to link in the generated object file
// and thus register the generators.
extern volatile int YAMLGeneratorAnchorSource;
extern volatile int MDGeneratorAnchorSource;
extern volatile int HTMLGeneratorAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED YAMLGeneratorAnchorDest =
    YAMLGeneratorAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED MDGeneratorAnchorDest =
    MDGeneratorAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED HTMLGeneratorAnchorDest =
    HTMLGeneratorAnchorSource;

} // namespace doc
} // namespace clang
