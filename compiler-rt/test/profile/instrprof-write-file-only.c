// RUN: %clang_profgen -o %t -O3 %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s

int __llvm_profile_runtime = 0;
void __llvm_profile_initialize_file(void);
int __llvm_profile_write_file(void);
void __llvm_profile_set_filename(const char *);
int foo(int);
int main(int argc, const char *argv[]) {
  // CHECK-LABEL: define {{.*}} @main(
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc > 42)
    return 1;

  // Since the runtime has been suppressed, initialize the file name, as the
  // writing will fail below as the file name has not been specified.
  __llvm_profile_initialize_file();

  // Write out the profile.
  __llvm_profile_write_file();

  // Change the profile.
  return foo(0);
}
int foo(int X) {
  // There should be no profiling information for @foo, since it was called
  // after the profile was written (and the atexit was suppressed by defining
  // profile_runtime).
  // CHECK-LABEL: define {{.*}} @foo(
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{[^,]+$}}
  return X <= 0 ? -X : X;
}
// CHECK: ![[PD1]] = !{!"branch_weights", i64 1, i64 2}
