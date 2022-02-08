//===- lld/Common/Driver.h - Linker Driver Emulator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COMMON_DRIVER_H
#define LLD_COMMON_DRIVER_H

#include "lld/Common/CommonLinkerContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
struct SafeReturn {
  int ret;
  bool canRunAgain;
};

// Generic entry point when using LLD as a library, safe for re-entry, supports
// crash recovery. Returns a general completion code and a boolean telling
// whether it can be called again. In some cases, a crash could corrupt memory
// and re-entry would not be possible anymore. Use exitLld() in that case to
// properly exit your application and avoid intermittent crashes on exit caused
// by cleanup.
SafeReturn safeLldMain(int argc, const char **argv, llvm::raw_ostream &stdoutOS,
                       llvm::raw_ostream &stderrOS);

namespace coff {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace mingw {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace elf {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace macho {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace wasm {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

// Temporary API that forces global state cleanup between explicit calls to
// drivers above. DO NOT USE - this will be replaced by safeLldMain(). See
// discussion in https://reviews.llvm.org/D119049.
void cleanup();
} // namespace lld

#endif
