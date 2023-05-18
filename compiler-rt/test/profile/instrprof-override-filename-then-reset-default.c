// RUN: rm -rf %t.d
// RUN: mkdir -p %t.d
// RUN: cd %t.d
// RUN: %clang_profgen -O3 %s -o %t.out
// RUN: %run %t.out %t.d/bad.profraw
// RUN: llvm-profdata merge -o %t.d/default.profdata %t.d/default.profraw
// RUN: %clang_profuse=%t.d/default.profdata -o - -S -emit-llvm %s | FileCheck %s


void __llvm_profile_set_filename(const char *);
int main(int argc, const char *argv[]) {
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc < 2)
    return 1;
  __llvm_profile_set_filename(argv[1]);
  __llvm_profile_set_filename(0);
  return 0;
}
// CHECK: ![[PD1]] = !{!"branch_weights", i32 1, i32 2}
