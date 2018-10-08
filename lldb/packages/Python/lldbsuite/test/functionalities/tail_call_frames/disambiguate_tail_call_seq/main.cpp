//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-implicit-check-not=artificial")
  // CHECK: frame #0: 0x{{[0-9a-f]+}} a.out`sink() at main.cpp:[[@LINE-1]]:4 [opt]
  // CHECK-NEXT: func3{{.*}} [opt] [artificial]
  // CHECK-NEXT: func1{{.*}} [opt] [artificial]
  // CHECK-NEXT: main{{.*}} [opt]
}

void __attribute__((noinline)) func3() { sink(); /* tail */ }

void __attribute__((noinline)) func2() { sink(); /* tail */ }

void __attribute__((noinline)) func1() { func3(); /* tail */ }

int __attribute__((disable_tail_calls)) main(int argc, char **) {
  // The sequences `main -> func1 -> f{2,3} -> sink` are both plausible. Test
  // that lldb picks the latter sequence.
  func1();
  return 0;
}
