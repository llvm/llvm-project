//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

volatile int x;

void __attribute__((noinline)) sink2() {
  x++; //% self.filecheck("bt", "main.cpp", "-check-prefix=FROM-FUNC1")
  // FROM-FUNC1: frame #0: 0x{{[0-9a-f]+}} a.out`sink{{.*}} at main.cpp:[[@LINE-1]]:{{.*}} [opt]
  // FROM-FUNC1-NEXT: sink({{.*}} [opt]
  // FROM-FUNC1-NEXT: func1{{.*}} [opt] [artificial]
  // FROM-FUNC1-NEXT: main{{.*}} [opt]
}

void __attribute__((noinline)) sink(bool called_from_main) {
  if (called_from_main) {
    x++; //% self.filecheck("bt", "main.cpp", "-check-prefix=FROM-MAIN")
    // FROM-MAIN: frame #0: 0x{{[0-9a-f]+}} a.out`sink{{.*}} at main.cpp:[[@LINE-1]]:{{.*}} [opt]
    // FROM-MAIN-NEXT: main{{.*}} [opt]
  } else {
    sink2();
  }
}

void __attribute__((noinline)) func1() { sink(false); /* tail */ }

int __attribute__((disable_tail_calls)) main(int argc, char **) {
  // When func1 tail-calls sink, make sure that the former appears in the
  // backtrace.
  sink(true);
  func1();
  return 0;
}
