//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-implicit-check-not=artificial")
  // CHECK: frame #0: 0x{{[0-9a-f]+}} a.out`sink() at main.cpp:[[@LINE-1]]:4 [opt]
  // CHECK-NEXT: frame #1: 0x{{[0-9a-f]+}} a.out`func3{{.*}} [opt] [artificial]
  // CHECK-NEXT: frame #2: 0x{{[0-9a-f]+}} a.out`func2{{.*}} [opt]
  // CHECK-NEXT: frame #3: 0x{{[0-9a-f]+}} a.out`func1{{.*}} [opt] [artificial]
  // CHECK-NEXT: frame #4: 0x{{[0-9a-f]+}} a.out`main{{.*}} [opt]
}

void __attribute__((noinline)) func3() { sink(); /* tail */ }

void __attribute__((disable_tail_calls, noinline)) func2() { func3(); /* regular */ }

void __attribute__((noinline)) func1() { func2(); /* tail */ }

int __attribute__((disable_tail_calls)) main() {
  func1(); /* regular */
  return 0;
}
