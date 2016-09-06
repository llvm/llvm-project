// Test that outdated data is ignored.

// FIXME: It would be nice to use -verify here instead of FileCheck, but -verify
// doesn't play well with warnings that have no line number.

// RUN: llvm-profdata merge %S/Inputs/c-outdated-data.proftext -o %t.profdata
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-outdated-data.c %s -o /dev/null -emit-llvm -fprofile-instrument-use-path=%t.profdata -Wprofile-instr-dropped 2>&1 | FileCheck %s
// CHECK: warning: profile data may be out of date: of 3 functions, 1 has no data and 1 has mismatched data that will be ignored

void no_usable_data() {
  int i = 0;

  if (i) {}

#ifdef GENERATE_OUTDATED_DATA
  if (i) {}
#endif
}

#ifndef GENERATE_OUTDATED_DATA
void no_data() {
}
#endif

int main(int argc, const char *argv[]) {
  no_usable_data();
  return 0;
}
