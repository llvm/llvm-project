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

namespace lldb_private {

char *FastDemangle(const char *mangled_name);

char *FastDemangle(const char *mangled_name, long mangled_name_length);
}

#endif
