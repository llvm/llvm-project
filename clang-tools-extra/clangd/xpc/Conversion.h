//===--- Conversion.h - LSP data (de-)serialization through XPC -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_XPC_XPCJSONCONVERSIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_XPC_XPCJSONCONVERSIONS_H

#include "llvm/Support/JSON.h"
#include <xpc/xpc.h>

namespace clang {
namespace clangd {

xpc_object_t jsonToXpc(const llvm::json::Value &JSON);
llvm::json::Value xpcToJson(const xpc_object_t &XPCObject);

} // namespace clangd
} // namespace clang

#endif
