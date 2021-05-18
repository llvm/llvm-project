//===-- SBLanguageRuntime.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLanguageRuntime.h"
#include "SBReproducerPrivate.h"
#include "lldb/Target/Language.h"

using namespace lldb;
using namespace lldb_private;

lldb::LanguageType
SBLanguageRuntime::GetLanguageTypeFromString(const char *string) {
  LLDB_RECORD_STATIC_METHOD(lldb::LanguageType, SBLanguageRuntime,
                            GetLanguageTypeFromString, (const char *), string);

  return Language::GetLanguageTypeFromString(llvm::StringRef(string));
}

const char *
SBLanguageRuntime::GetNameForLanguageType(lldb::LanguageType language) {
  LLDB_RECORD_STATIC_METHOD(const char *, SBLanguageRuntime,
                            GetNameForLanguageType, (lldb::LanguageType),
                            language);

  return Language::GetNameForLanguageType(language);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBLanguageRuntime>(Registry &R) {
  LLDB_REGISTER_STATIC_METHOD(lldb::LanguageType, SBLanguageRuntime,
                              GetLanguageTypeFromString, (const char *));
  LLDB_REGISTER_STATIC_METHOD(const char *, SBLanguageRuntime,
                              GetNameForLanguageType, (lldb::LanguageType));
}

}
}
