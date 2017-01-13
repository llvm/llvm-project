//===----------------------------- config.h -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
//  Defines macros used within the libc++abi project.
//
//===----------------------------------------------------------------------===//


#ifndef LIBCXXABI_CONFIG_H
#define LIBCXXABI_CONFIG_H

#include <unistd.h>

// Set this in the CXXFLAGS when you need it, because otherwise we'd have to
// #if !defined(__linux__) && !defined(__APPLE__) && ...
// and so-on for *every* platform.
#ifndef LIBCXXABI_BAREMETAL
#  define LIBCXXABI_BAREMETAL 0
#endif

// The default terminate handler attempts to demangle uncaught exceptions, which
// causes extra I/O and demangling code to be pulled in.
// Set this to make the terminate handler default to a silent alternative.
#ifndef LIBCXXABI_SILENT_TERMINATE
#  define LIBCXXABI_SILENT_TERMINATE 0
#endif

#endif // LIBCXXABI_CONFIG_H
