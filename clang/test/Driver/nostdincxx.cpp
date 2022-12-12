// RUN: not %clangxx -nostdinc %s 2>&1 | FileCheck %s
// RUN: not %clangxx -nostdinc++ %s 2>&1 | FileCheck %s
// RUN: not %clangxx -nostdlibinc %s 2>&1 | FileCheck %s
// RUN: not %clangxx --target=x86_64-unknown-unknown-gnu -fsyntax-only -nostdinc -nostdinc++ %s 2>&1 | FileCheck /dev/null --implicit-check-not=-Wunused-command-line-argument
// CHECK: 'vector' file not found
#include <vector>

// MSVC, PS4, PS5 have C++ headers in the same directory as C headers.
// UNSUPPORTED: ms-sdk, target={{.*-(ps4|ps5)}}
