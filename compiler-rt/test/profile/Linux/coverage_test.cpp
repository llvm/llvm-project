// RUN: %clang_profgen -fuse-ld=gold -O2 -fdata-sections -ffunction-sections -fcoverage-mapping -Wl,--gc-sections -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -filename-equivalence 2>&1 | FileCheck %s
// BFD linker older than 2.26 has a bug that per-func profile data will be wrongly garbage collected when GC is turned on. We only do end-to-end test here without GC:
// RUN: %clang_profgen -O2  -fcoverage-mapping  -o %t.2 %s
// RUN: env LLVM_PROFILE_FILE=%t.2.profraw %run %t.2
// RUN: llvm-profdata merge -o %t.2.profdata %t.2.profraw
// RUN: llvm-cov show %t.2 -instr-profile %t.2.profdata -filename-equivalence 2>&1 | FileCheck %s
// Check covmap is not garbage collected when GC is turned on with BFD linker. Due to the bug mentioned above, we can only
// do the check with objdump:
// RUN: %clang_profgen -O2  -fcoverage-mapping -Wl,--gc-sections -o %t.3 %s
// RUN: llvm-objdump -h %t.3 | FileCheck --check-prefix COVMAP %s
// Check PIE option
// RUN: %clang_profgen -fuse-ld=gold -O2 -fdata-sections -ffunction-sections -fPIE -pie -fcoverage-mapping -Wl,--gc-sections -o %t.pie %s
// RUN: env LLVM_PROFILE_FILE=%t.pie.profraw %run %t.pie
// RUN: llvm-profdata merge -o %t.pie.profdata %t.pie.profraw
// RUN: llvm-cov show %t.pie -instr-profile %t.pie.profdata -filename-equivalence 2>&1 | FileCheck %s

void foo(bool cond) { // CHECK:  1| [[@LINE]]|void foo(
  if (cond) {         // CHECK:  1| [[@LINE]]|  if (cond) {
  }                   // CHECK:  0| [[@LINE]]|  }
}                     // CHECK:  1| [[@LINE]]|}
void bar() {          // CHECK:  1| [[@LINE]]|void bar() {
}                     // CHECK:  1| [[@LINE]]|}
void func() {         // CHECK:  0| [[@LINE]]|void func(
}                     // CHECK:  0| [[@LINE]]|}
int main() {          // CHECK:  1| [[@LINE]]|int main(
  foo(false);         // CHECK:  1| [[@LINE]]|  foo(
  bar();              // CHECK:  1| [[@LINE]]|  bar(
  return 0;           // CHECK:  1| [[@LINE]]|  return
}                     // CHECK:  1| [[@LINE]]|}

// COVMAP: __llvm_covmap {{.*}}

