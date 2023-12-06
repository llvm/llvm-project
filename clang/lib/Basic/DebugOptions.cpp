//===--- DebugOptions.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DebugOptions.h"

namespace clang {

DebugOptions::DebugOptions() {
#define DEBUGOPT(Name, Bits, Default) Name = Default;
#define ENUM_DEBUGOPT(Name, Type, Bits, Default) set##Name(Default);
#include "clang/Basic/DebugOptions.def"
}

void DebugOptions::resetNonModularOptions(llvm::StringRef ModuleFormat) {
  // First reset all debug options that can always be reset, because they never
  // affect the PCM.
#define DEBUGOPT(Name, Bits, Default)
#define BENIGN_DEBUGOPT(Name, Bits, Default) Name = Default;
#define BENIGN_VALUE_DEBUGOPT(Name, Bits, Default) Name = Default;
#define BENIGN_ENUM_DEBUGOPT(Name, Type, Bits, Default) set##Name(Default);
#include "clang/Basic/DebugOptions.def"

  // Conditionally reset debug options that only matter when the debug info is
  // emitted into the PCM (-gmodules).
  if (ModuleFormat == "raw" && !DebugTypeExtRefs) {
#define DEBUGOPT(Name, Bits, Default) Name = Default;
#define ENUM_DEBUGOPT(Name, Type, Bits, Default) set##Name(Default);
#define BENIGN_DEBUGOPT(Name, Bits, Default)
#define BENIGN_VALUE_DEBUGOPT(Name, Bits, Default)
#define BENIGN_ENUM_DEBUGOPT(Name, Type, Bits, Default)
#include "clang/Basic/DebugOptions.def"
  }
}

} // end namespace clang
