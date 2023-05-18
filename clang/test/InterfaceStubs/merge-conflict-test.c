// RUN: not %clang -fvisibility=default -o libfoo.so -emit-interface-stubs %s %S/driver-test.c 2>&1 | FileCheck %s
// CHECK: error: Interface Stub: Type Mismatch
int foo;
