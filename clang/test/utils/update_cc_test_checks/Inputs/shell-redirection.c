/// Check that shell redirections in the RUN line are handled
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm < %s 2>/dev/null | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -disable-O0-optnone -o - %s 2>&1 | opt -S -passes=mem2reg \
// RUN:   | FileCheck %s --check-prefix=MEM2REG

int test(int a, int b) {
  return a + b;
}
