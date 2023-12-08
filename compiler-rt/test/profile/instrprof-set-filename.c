// RUN: rm -f %t.profraw
//
// 1. Test that __llvm_profile_set_filename has higher precedence than
//    the default path.
// RUN: %clang_profgen -o %t -O3 %s
// RUN: %run %t %t.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s
// RUN: rm %t.profraw
// RUN: rm %t.profdata
// 2. Test that __llvm_profile_set_filename has higher precedence than
//    environment variable
// RUN: env LLVM_PROFILE_FILE=%t.env.profraw %run %t %t.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s
// RUN: rm %t.profraw
// RUN: rm %t.profdata
// 3. Test that __llvm_profile_set_filename has higher precedence than
//    the command line.
// RUN: %clang_profgen=%t.cmd.profraw -o %t-cmd -O3 %s
// RUN: %run %t-cmd %t.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s
// RUN: rm %t.profraw
// RUN: rm %t.profdata
// 4. Test that command line has high precedence than the default path
// RUN: %clang_profgen=%t.cmd.profraw -DNO_API -o %t-cmd -O3 %s
// RUN: %run %t-cmd %t.profraw
// RUN: llvm-profdata merge -o %t.cmd.profdata %t.cmd.profraw
// RUN: %clang_profuse=%t.cmd.profdata -o - -S -emit-llvm %s | FileCheck %s
// RUN: rm %t.cmd.profraw
// RUN: rm %t.cmd.profdata
// 5. Test that the environment variable has higher precedence than
//    the command line.
// RUN: env LLVM_PROFILE_FILE=%t.env.profraw %run %t-cmd %t.profraw
// RUN: llvm-profdata merge -o %t.env.profdata %t.env.profraw
// RUN: %clang_profuse=%t.env.profdata -o - -S -emit-llvm %s | FileCheck %s
// RUN: rm %t.env.profraw
// RUN: rm %t.env.profdata
// RUN: rm %t %t-cmd

#ifdef CALL_SHARED
extern void func(int);
#endif
void __llvm_profile_set_filename(const char *);
int main(int argc, const char *argv[]) {
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  if (argc < 2)
    return 1;
#ifndef NO_API
  __llvm_profile_set_filename(argv[1]);
#endif

#ifdef CALL_SHARED
  func(1);
#endif
  return 0;
}
// CHECK: ![[PD1]] = !{!"branch_weights", i32 1, i32 2}
// SHARED: Total functions: 2
