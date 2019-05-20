// RUN: not %clangxx -nostdinc++ %s 2>&1 | FileCheck %s
// CHECK: file not found
#include <vector> 

// MSVC has C++ headers in same directory as C headers.
// REQUIRES: non-ms-sdk
// REQUIRES: non-ps4-sdk
