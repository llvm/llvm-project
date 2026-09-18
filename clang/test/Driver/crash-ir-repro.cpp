// RUN: %clang -S -emit-llvm -o %t.ll %s
// RUN: not %crash_opt %clang -S -DCRASH %s -o %t.ll 2>&1 | FileCheck %s

// CHECK: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.cpp
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.sh

#ifdef CRASH
#pragma clang __debug parser_crash
#endif

int main() {
  return 0;
}
