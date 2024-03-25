// Testing profile generate.
// RUN: %clang_profgen %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFGEN
// RUN: %clang_pgogen %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFGEN

// Testing profile use. Generate some profile file first.
// RUN: rm -rf rawprof.profraw
// RUN: %clang_profgen -o %t1 %s
// RUN: %run %t1
// RUN: llvm-profdata merge -o %t1.profdata rawprof.profraw
// RUN: %clang_profuse=%t1.profdata %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFUSE
// RUN: rm -rf rawprof.profraw
// RUN: %clang_pgogen -o %t2 %s
// RUN: %run %t2
// RUN: llvm-profdata merge -o %t2.profdata rawprof.profraw
// RUN: %clang_pgouse=%t2.profdata %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFUSE
#include "profile/instr_prof_interface.h"

__attribute__((noinline)) int bar() { return 4; }

int foo() {
  __llvm_profile_reset_counters();
  // PROFGEN: call void @__llvm_profile_reset_counters()
  // PROFUSE-NOT: call void @__llvm_profile_reset_counters()
  return bar();
}

// PROFUSE-NOT: declare void @__llvm_profile_reset_counters()

int main() {
  int z = foo() + 3;
  __llvm_profile_set_filename("rawprof.profraw");
  // PROFGEN: call void @__llvm_profile_set_filename(ptr noundef @{{.*}})
  // PROFUSE-NOT: call void @__llvm_profile_set_filename(ptr noundef @{{.*}})
  if (__llvm_profile_dump())
    return 2;
  // PROFGEN: %{{.*}} = call {{(signext )*}}i32 @__llvm_profile_dump()
  // PROFUSE-NOT: %{{.*}} = call {{(signext )*}}i32 @__llvm_profile_dump()
  return z + bar() - 11;
}

// PROFUSE-NOT: declare void @__llvm_profile_set_filename(ptr noundef)
// PROFUSE-NOT: declare signext i32 @__llvm_profile_dump()
