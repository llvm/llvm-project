// RUN: %clang_profgen %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFGEN
// RUN: %clang_profgen -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: %clang_profuse=%t.profdata %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFUSE
#include "profile/instr_prof_interface.h"

__attribute__((noinline)) int bar() { return 4; }

int foo() {
  __llvm_profile_reset_counters();
  // PROFGEN: call void @__llvm_profile_reset_counters()
  // PROFUSE-NOT: call void @__llvm_profile_reset_counters()
  return bar();
}

int main() {
  int z = foo() + 3;
  __llvm_profile_dump();
  // PROFGEN: %call1 = call signext i32 @__llvm_profile_dump()
  // PROFUSE-NOT: %call1 = call signext i32 @__llvm_profile_dump()
  __llvm_orderfile_dump();
  // PROFGEN: %call2 = call signext i32 @__llvm_orderfile_dump()
  // PROFUSE-NOT: %call2 = call signext i32 @__llvm_orderfile_dump()
  return z + bar() - 11;
}
