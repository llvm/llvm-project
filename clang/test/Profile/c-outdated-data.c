// Test that outdated data is ignored.

// FIXME: It would be nice to use -verify here instead of FileCheck, but -verify
// doesn't play well with warnings that have no line number.

// RUN: llvm-profdata merge %S/Inputs/c-outdated-data.proftext -o %t.profdata
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-outdated-data.c %s -o /dev/null -emit-llvm -fprofile-instrument-use-path=%t.profdata 2>&1 | FileCheck %s -check-prefix=NO_MISSING
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-outdated-data.c %s -o /dev/null -emit-llvm -Wprofile-instr-missing -fprofile-instrument-use-path=%t.profdata 2>&1 | FileCheck %s -check-prefix=WITH_MISSING

// NO_MISSING: warning: profile data may be out of date: of 3 functions, 2 have mismatched data that will be ignored
// NO_MISSING-NOT: 1 has no data

// WITH_MISSING: warning: profile data may be out of date: of 3 functions, 2 have mismatched data that will be ignored
// WITH_MISSING: warning: profile data may be incomplete: of 3 functions, 1 has no data

void no_usable_data(void) {
  int i = 0;

  if (i) {}
}

void no_data(void) {
}

int main(int argc, const char *argv[]) {
  no_usable_data();
  return 0;
}
