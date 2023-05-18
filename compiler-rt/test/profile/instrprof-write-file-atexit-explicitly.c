// RUN: %clang_profgen -o %t -O3 %s
// RUN: %run %t %t.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s

int __llvm_profile_runtime = 0;
int __llvm_profile_register_write_file_atexit(void);
void __llvm_profile_set_filename(const char *);
int main(int argc, const char *argv[]) {
  __llvm_profile_register_write_file_atexit();
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc < 2)
    return 1;
  __llvm_profile_set_filename(argv[1]);
  return 0;
}
// CHECK: ![[PD1]] = !{!"branch_weights", i32 1, i32 2}
