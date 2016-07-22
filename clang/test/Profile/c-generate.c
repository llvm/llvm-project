// Check that the -fprofile-instrument-path= form works.
// RUN: %clang_cc1 -main-file-name c-generate.c %s -o - -emit-llvm -fprofile-instrument=clang -fprofile-instrument-path=c-generate-test.profraw | FileCheck %s --check-prefix=PROF-INSTR-PATH
// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=none | FileCheck %s --check-prefix=PROF-INSTR-NONE
// RUN: not %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=garbage 2>&1 | FileCheck %s --check-prefix=PROF-INSTR-GARBAGE
//
// PROF-INSTR-PATH: constant [24 x i8] c"c-generate-test.profraw\00"
//
// PROF-INSTR-NONE-NOT: @__profn_main
// PROF-INSTR-GARBAGE: invalid PGO instrumentor in argument '-fprofile-instrument=garbage'

int main(void) {
  return 0;
}
