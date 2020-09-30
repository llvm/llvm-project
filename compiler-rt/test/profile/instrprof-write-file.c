// RUN: rm -rf %t1.profraw %t2.profraw
// RUN: %clang_profgen -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t1.profraw %run %t %t2.profraw
// RUN: llvm-profdata merge -o %t1.profdata %t1.profraw
// RUN: %clang_profuse=%t1.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=CHECK1 --check-prefix=CHECK
// RUN: llvm-profdata merge -o %t2.profdata %t2.profraw
// RUN: %clang_profuse=%t2.profdata -o - -S -emit-llvm %s | FileCheck %s --check-prefix=CHECK2 --check-prefix=CHECK

int __llvm_profile_write_file(void);
void __llvm_profile_set_filename(const char *);
int foo(int);
int main(int argc, const char *argv[]) {
  // CHECK-LABEL: define {{.*}} @main(
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc < 2)
    return 1;

  // Write out the profile.
  __llvm_profile_write_file();

  // Change the profile.
  int Ret = foo(0);

  // It'll write out again at exit; change the filename so we get two files.
  __llvm_profile_set_filename(argv[1]);
  return Ret;
}
int foo(int X) {
  // CHECK-LABEL: define {{.*}} @foo(
  // CHECK1: br i1 %{{.*}}, label %{{.*}}, label %{{[^,]+$}}
  // CHECK2: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD2:[0-9]+]]
  return X <= 0 ? -X : X;
}
// CHECK: ![[PD1]] = !{!"branch_weights", i64 1, i64 2}
// CHECK2: ![[PD2]] = !{!"branch_weights", i64 2, i64 1}
