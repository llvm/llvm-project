// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

void foo(int x)
{
  _Atomic(int) i = 0;
  _Atomic(short) j = 0;
  // Check that multiply / divides on atomics produce a cmpxchg loop
  i *= 2;
  // CHECK: mul nsw i32
  // CHECK: cmpxchg i32*
  i /= 2;
  // CHECK: sdiv i32
  // CHECK: cmpxchg i32*
  j /= x;
  // CHECK: sdiv i32
  // CHECK: cmpxchg i16*

  // These should be emitting atomicrmw instructions, but they aren't yet
  i += 2; // CHECK: cmpxchg
  i -= 2; // CHECK: cmpxchg
  i++; // CHECK: cmpxchg
  i--; // CHECK: cmpxchg
}
