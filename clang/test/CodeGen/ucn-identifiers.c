// RUN: %clang_cc1 %s -emit-llvm -o /dev/null
// RUN: %clang_cc1 %s -emit-llvm -o /dev/null -x c++
// This file contains UTF-8; please do not fix!


extern void \u00FCber(int);
extern void \U000000FCber(int); // redeclaration, no warning

void goodCalls(void) {
  \u00FCber(0);
  \u00fcber(1);
  über(2);
  \U000000FCber(3);
}
