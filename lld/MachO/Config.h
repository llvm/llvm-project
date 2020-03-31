//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONFIG_H
#define LLD_MACHO_CONFIG_H

#include "llvm/ADT/StringRef.h"

namespace lld {
namespace macho {

class Symbol;

struct Configuration {
  llvm::StringRef outputFile;
  Symbol *entry;
};

extern Configuration *config;

} // namespace macho
} // namespace lld

#endif
