//===--- Conversion.h - LSP data (de-)serialization through XPC -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
