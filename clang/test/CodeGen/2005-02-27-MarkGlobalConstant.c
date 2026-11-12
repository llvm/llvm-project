// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -emit-llvm -o - | FileCheck %s %}
// PR10414

// The synthetic global made by the CFE for big initializer should be marked
// constant.

void bar(char *);
void foo(void) {
  // CHECK: private unnamed_addr constant
  char Blah[] = "asdlfkajsdlfkajsd;lfkajds;lfkjasd;flkajsd;lkfja;sdlkfjasd";
  bar(Blah);
}
