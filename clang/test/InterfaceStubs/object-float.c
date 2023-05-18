// RUN: not %clang -fvisibility=default -o - -emit-interface-stubs %s %S/object.c 2>&1 | FileCheck %s
// CHECK: error: Interface Stub: Size Mismatch
double data = 42.0;
