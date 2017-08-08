//===-- CxaDemangle.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CxaDemangle_h_
#define liblldb_CxaDemangle_h_

namespace lldb_private {

char *__cxa_demangle(const char *mangled_name, char *buf, size_t *n,
                     int *status);
}

#endif
