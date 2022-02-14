// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s
// PR10414

// The synthetic global made by the CFE for big initializer should be marked
// constant.

void bar();
void foo(void) {
  // CHECK: private unnamed_addr constant
  char Blah[] = "asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd";
  bar(Blah);
}
