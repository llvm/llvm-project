// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O3 -mframe-pointer=non-leaf -pg -S -o - %s | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O3 -mframe-pointer=all -pg -S -o - %s | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O3 -mframe-pointer=non-leaf -pg -S -o - %s | \
// RUN:   FileCheck %s

// Test that the frame pointer is kept when compiling with
// profiling.

//CHECK: pushq %rbp
int main(void)
{
  return 0;
}
