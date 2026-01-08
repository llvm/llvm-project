// RUN: not %clang_cc1 -fsyntax-only -triple i686-win32 %s 2>&1 | FileCheck %s

#include "Inputs\success.h"

// CHECK: error: 'Inputs\success.h' file not found
// CHECK: #include "Inputs\success.h"
// CHECK:          ^

// This test is really checking that we *don't* replace backslashes with slashes
// on non-Windows unless -fms-extensions is passed. It won't fail in this way on
// Windows because the filesystem will interpret the backslash as a directory
// separator.
// UNSUPPORTED: system-windows, system-cygwin
