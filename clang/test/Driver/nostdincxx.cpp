// RUN: not %clangxx -nostdinc %s 2>&1 | FileCheck %s
// RUN: not %clangxx -nostdinc++ %s 2>&1 | FileCheck %s
// RUN: not %clangxx -nostdlibinc %s 2>&1 | FileCheck %s
// RUN: not %clangxx -triple x86_64-unknown-unknown-gnu -fsyntax-only -nostdinc -nostdinc++ %s 2>&1 | FileCheck /dev/null --implicit-check-not=-Wunused-command-line-argument
// CHECK: file not found
#include <vector> 

// MSVC and PS4 have C++ headers in the same directory as C headers.
// UNSUPPORTED: ms-sdk, ps4
