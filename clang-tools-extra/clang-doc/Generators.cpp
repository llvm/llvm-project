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
  for (const auto &Generator : GeneratorRegistry::entries()) {
    if (Generator.getName() != Format)
      continue;
    return Generator.instantiate();
  }
  return createStringError(llvm::inconvertibleErrorCode(),
                           "can't find generator: " + Format);
}

// Enum conversion

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

llvm::Error Generator::createResources(ClangDocContext &CDCtx) {
  return llvm::Error::success();
}

// A function to add a reference to Info in Idx.
// Given an Info X with the following namespaces: [B,A]; a reference to X will
// be added in the children of a reference to B, which should be also a child of
// a reference to A, where A is a child of Idx.
//   Idx
//    |-- A
//        |--B
//           |--X
// If the references to the namespaces do not exist, they will be created. If
// the references already exist, the same one will be used.
void Generator::addInfoToIndex(Index &Idx, const doc::Info *Info) {
  // Index pointer that will be moving through Idx until the first parent
  // namespace of Info (where the reference has to be inserted) is found.
  Index *I = &Idx;
  // The Namespace vector includes the upper-most namespace at the end so the
  // loop will start from the end to find each of the namespaces.
  for (const auto &R : llvm::reverse(Info->Namespace)) {
    // Look for the current namespace in the children of the index I is
    // pointing.
    auto It = llvm::find(I->Children, R.USR);
    if (It != I->Children.end()) {
      // If it is found, just change I to point the namespace reference found.
      I = &*It;
    } else {
      // If it is not found a new reference is created
      I->Children.emplace_back(R.USR, R.Name, R.RefType, R.Path);
      // I is updated with the reference of the new namespace reference
      I = &I->Children.back();
    }
  }
  // Look for Info in the vector where it is supposed to be; it could already
  // exist if it is a parent namespace of an Info already passed to this
  // function.
  auto It = llvm::find(I->Children, Info->USR);
  if (It == I->Children.end()) {
    // If it is not in the vector it is inserted
    I->Children.emplace_back(Info->USR, Info->extractName(), Info->IT,
                             Info->Path);
  } else {
    // If it not in the vector we only check if Path and Name are not empty
    // because if the Info was included by a namespace it may not have those
    // values.
    if (It->Path.empty())
      It->Path = Info->Path;
    if (It->Name.empty())
      It->Name = Info->extractName();
  }
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
