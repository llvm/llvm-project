//===-- SWIG Interface for SBLanguageRuntime --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBLanguageRuntime
{
public:
    static lldb::LanguageType
    GetLanguageTypeFromString (const char *string);
    
    static const char *
    GetNameForLanguageType (lldb::LanguageType language);
};

} // namespace lldb
