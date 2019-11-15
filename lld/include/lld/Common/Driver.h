//===- lld/Common/Driver.h - Linker Driver Emulator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COMMON_DRIVER_H
#define LLD_COMMON_DRIVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace coff {
bool link(llvm::ArrayRef<const char *> args, bool canExitEarly,
          llvm::raw_ostream &stdout, llvm::raw_ostream &stderr);
}

namespace mingw {
bool link(llvm::ArrayRef<const char *> args, bool canExitEarly,
          llvm::raw_ostream &stdout, llvm::raw_ostream &stderr);
}

namespace elf {
bool link(llvm::ArrayRef<const char *> args, bool canExitEarly,
          llvm::raw_ostream &stdout, llvm::raw_ostream &stderr);
}

namespace mach_o {
bool link(llvm::ArrayRef<const char *> args, bool canExitEarly,
          llvm::raw_ostream &stdout, llvm::raw_ostream &stderr);
}

namespace wasm {
bool link(llvm::ArrayRef<const char *> args, bool canExitEarly,
          llvm::raw_ostream &stdout, llvm::raw_ostream &stderr);
}
}

#endif
