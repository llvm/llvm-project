//===-- lldb-python.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_python_h_
#define LLDB_lldb_python_h_

// Python.h needs to be included before any system headers in order to avoid redefinition of macros

#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#endif // LLDB_DISABLE_PYTHON

#endif  // LLDB_lldb_python_h_
