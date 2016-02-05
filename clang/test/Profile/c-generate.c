// Check that the -fprofile-instrument-path= form works.
// RUN: %clang_cc1 -main-file-name c-generate.c %s -o - -emit-llvm -fprofile-instrument=clang -fprofile-instrument-path=c-generate-test.profraw | FileCheck %s --check-prefix=PROF-INSTR-PATH
// RUN: %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=none | FileCheck %s --check-prefix=PROF-INSTR-NONE
// RUN: not %clang_cc1 %s -o - -emit-llvm -fprofile-instrument=garbage 2>&1 | FileCheck %s --check-prefix=PROF-INSTR-GARBAGE
//
// PROF-INSTR-PATH: private constant [24 x i8] c"c-generate-test.profraw\00"
// PROF-INSTR-PATH: call void @__llvm_profile_override_default_filename(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @0, i32 0, i32 0))
// PROF-INSTR-PATH: declare void @__llvm_profile_override_default_filename(i8*)
//
// PROF-INSTR-NONE-NOT: @__profn_main
// PROF-INSTR-GARBAGE: invalid PGO instrumentor in argument '-fprofile-instrument=garbage'

int main(void) {
  return 0;
}
