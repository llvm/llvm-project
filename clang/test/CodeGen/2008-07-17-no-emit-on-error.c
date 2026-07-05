// RUN: rm -f %t1.bc
// RUN: %clang_cc1 -DPASS %s -emit-llvm-bc -o %t1.bc
// RUN: opt %t1.bc -disable-output
// RUN: rm -f %t1.bc
// RUN: not %clang_cc1 %s -emit-llvm-bc -o %t1.bc
// RUN: not opt %t1.bc -disable-output

void f(void) {
}

#ifndef PASS
void g(void) {
  *10;
}
#endif
