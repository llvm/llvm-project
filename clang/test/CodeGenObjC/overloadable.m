// rdar://6657613
// RUN: %clang_cc1 -emit-llvm %s -o %t

@class C;

// RUN: grep _Z1fP11objc_object %t | count 1
void __attribute__((overloadable)) f(id c) { }

// RUN: grep _Z1fP1C %t | count 1
void __attribute__((overloadable)) f(C *c) { }
