// RUN: %clang_profgen -fuse-ld=gold -O2 -fdata-sections -ffunction-sections -fprofile-instr-generate -fcoverage-mapping -Wl,--gc-sections -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -filename-equivalence 2>&1 | FileCheck %s
// Testing PIE
// RUN: %clang_profgen -fuse-ld=gold -O2 -fdata-sections -ffunction-sections -fPIE -pie -fprofile-instr-generate -fcoverage-mapping -Wl,--gc-sections -o %t.pie %s
// RUN: env LLVM_PROFILE_FILE=%t.pie.profraw %run %t.pie
// RUN: llvm-profdata merge -o %t.pie.profdata %t.pie.profraw
// RUN: llvm-cov show %t.pie -instr-profile %t.pie.profdata -filename-equivalence 2>&1 | FileCheck %s
// testing m32
// RUN: %clang_profgen -fuse-ld=gold -O2 -fdata-sections -ffunction-sections -fPIE -pie -fprofile-instr-generate -fcoverage-mapping -m32 -Wl,--gc-sections -o %t.32.pie %s
// RUN: env LLVM_PROFILE_FILE=%t.32.pie.profraw %run %t.32.pie
// RUN: llvm-profdata merge -o %t.32.pie.profdata %t.32.pie.profraw
// RUN: llvm-cov show %t.32.pie -instr-profile %t.32.pie.profdata -filename-equivalence 2>&1 | FileCheck %s

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





