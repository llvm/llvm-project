//===------------------------------ Utils.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions required by both the module build daemon (server) and clang
// invocation (client)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_UTILS_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_UTILS_H

#include "llvm/Support/Error.h"
#include <string>

namespace cc1modbuildd {

void writeError(llvm::Error Err, std::string Msg);
std::string getFullErrorMsg(llvm::Error Err, std::string Msg);
llvm::Error makeStringError(llvm::Error Err, std::string Msg);

} // namespace cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_UTILS_H