// RUN: %clang -S -emit-llvm -o %t.ll %s
// RUN: not %clang -S -DCRASH %s -o %t.ll 2>&1 | FileCheck %s

// TODO(boomanaiden154): This test case causes clang to raise a signal when
// running under ubsan, but not in normal build configurations. This should
// be fixed.
// UNSUPPORTED: ubsan, hwasan

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.cpp
// CHECK-NEXT: clang: note: diagnostic msg: {{.*}}.sh

#ifdef CRASH
#pragma clang __debug parser_crash
#endif

int main() {
  return 0;
}
