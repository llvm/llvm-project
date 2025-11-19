// RUN: %clang -S -emit-llvm -o %t.ll %s
// RUN: not %clang -S -DCRASH %s %t.ll 2>&1 | FileCheck %s

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.cpp
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.ll
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.sh

#ifdef CRASH
#pragma clang __debug parser_crash
#endif

int main() {
  return 0;
}
