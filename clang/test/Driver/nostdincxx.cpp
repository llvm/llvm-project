// RUN: %clangxx -nostdinc++ %s 2>&1 | FileCheck %s
// XFAIL: win32
// CHECK: file not found
#include <vector> 
