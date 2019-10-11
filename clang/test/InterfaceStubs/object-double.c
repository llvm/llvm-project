// RUN: not %clang -o - -emit-interface-stubs %s %S/object.c 2>&1 | FileCheck %s
// Need to encode more type info or weak vs strong symbol resolution in llvm-ifs
// XFAIL: *
// CHECK: error: Interface Stub: Size Mismatch
float data = 42.0;