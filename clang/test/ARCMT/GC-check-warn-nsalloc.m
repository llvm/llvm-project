// RUN: %clang_cc1 -arcmt-check -verify -no-ns-alloc-error -triple x86_64-apple-darwin10 -fobjc-gc-only %s
// RUN: %clang_cc1 -arcmt-check -verify -no-ns-alloc-error -triple x86_64-apple-darwin10 -fobjc-gc-only -x objective-c++ %s
// DISABLE: mingw32
// rdar://10532541
// XFAIL: *

typedef unsigned NSUInteger;
void *__strong NSAllocateCollectable(NSUInteger size, NSUInteger options);

void test1() {
  NSAllocateCollectable(100, 0); // expected-warning {{call returns pointer to GC managed memory; it will become unmanaged in ARC}}
}
