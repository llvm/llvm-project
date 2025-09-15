//===--- CodeGenOptions.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/CodeGenOptions.h"

namespace clang {

CodeGenOptions::CodeGenOptions() {
#define CODEGENOPT(Name, Bits, Default, Compatibility) Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default, Compatibility)              \
  set##Name(Default);
#include "clang/Basic/CodeGenOptions.def"

  RelocationModel = llvm::Reloc::PIC_;
}

void CodeGenOptions::resetNonModularOptions(StringRef ModuleFormat) {
  // FIXME: Replace with C++20 `using enum CodeGenOptions::CompatibilityKind`.
  using CK = CompatibilityKind;

  // First reset benign codegen and debug options.
#define CODEGENOPT(Name, Bits, Default, Compatibility)                         \
  if constexpr (CK::Compatibility == CK::Benign)                               \
    Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default, Compatibility)              \
  if constexpr (CK::Compatibility == CK::Benign)                               \
    set##Name(Default);
#include "clang/Basic/CodeGenOptions.def"

  // Conditionally reset debug options that only matter when the debug info is
  // emitted into the PCM (-gmodules).
  if (ModuleFormat == "raw" && !DebugTypeExtRefs) {
#define DEBUGOPT(Name, Bits, Default, Compatibility)                           \
  if constexpr (CK::Compatibility != CK::Benign)                               \
    Name = Default;
#define VALUE_DEBUGOPT(Name, Bits, Default, Compatibility)                     \
  if constexpr (CK::Compatibility != CK::Benign)                               \
    Name = Default;
#define ENUM_DEBUGOPT(Name, Type, Bits, Default, Compatibility)                \
  if constexpr (CK::Compatibility != CK::Benign)                               \
    set##Name(Default);
#include "clang/Basic/DebugOptions.def"
  }

  RelocationModel = llvm::Reloc::PIC_;
}

}  // end namespace clang
