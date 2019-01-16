//===--- Conversion.cpp - LSP data (de-)serialization through XPC - C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "xpc/Conversion.h"
#include "Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include <string>
#include <vector>

using namespace llvm;
namespace clang {
namespace clangd {

xpc_object_t jsonToXpc(const json::Value &JSON) {
  const char *const Key = "LSP";
  xpc_object_t PayloadObj = xpc_string_create(llvm::to_string(JSON).c_str());
  return xpc_dictionary_create(&Key, &PayloadObj, 1);
}

json::Value xpcToJson(const xpc_object_t &XPCObject) {
  if (xpc_get_type(XPCObject) == XPC_TYPE_DICTIONARY) {
    const char *const LSP = xpc_dictionary_get_string(XPCObject, "LSP");
    auto Json = json::parse(llvm::StringRef(LSP));
    if (Json)
      return *Json;
    else
      elog("JSON parse error: {0}", toString(Json.takeError()));
  }
  return json::Value(nullptr);
}

} // namespace clangd
} // namespace clang
