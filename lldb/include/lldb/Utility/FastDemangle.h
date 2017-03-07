//===-- FastDemangle.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_FastDemangle_h_
#define liblldb_FastDemangle_h_

#include <cstddef>

#include <functional>

namespace lldb_private {

char *FastDemangle(const char *mangled_name);

char *
FastDemangle(const char *mangled_name, size_t mangled_name_length,
             std::function<void(const char *s)> primitive_type_hook = nullptr);
}

#endif
