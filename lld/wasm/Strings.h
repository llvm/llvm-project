//===- Strings.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_STRINGS_H
#define LLD_WASM_STRINGS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace lld {
namespace wasm {

std::string displayName(llvm::StringRef Name);

} // namespace wasm
} // namespace lld

#endif
