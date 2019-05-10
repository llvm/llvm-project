// RUN: not %clangxx -nostdinc++ %s 2>&1 | FileCheck %s
// CHECK: file not found
#include <vector> 

// MSVC and PS4 have C++ headers in the same directory as C headers.
// UNSUPPORTED: ms-sdk, ps4
