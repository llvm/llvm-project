// RUN: %clang_cc1 -emit-pch -o %t %s

// Test that serialization/deserialization of a DynamicAllocLValue
// variant of APValue does not crash.

#ifndef HEADER
#define HEADER

struct A {  int *p; };
const A &w = A{ new int(10) };

#endif
