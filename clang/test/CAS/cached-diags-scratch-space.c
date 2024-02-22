// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest %s > %t/casid

// Check that this doesn't crash and provides proper round-tripping.

// RUN: %clang -cc1 -triple x86_64-apple-macos12 -fsyntax-only \
// RUN:   -fconst-strings -Wincompatible-pointer-types-discards-qualifiers \
// RUN:   -serialize-diagnostic-file %t/regular.dia %s 2> %t/regular-diags.txt

// RUN: %clang -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job -emit-obj -o %t/output.o \
// RUN:   -fconst-strings -Wincompatible-pointer-types-discards-qualifiers \
// RUN:   -serialize-diagnostic-file %t/t1.dia %s 2> %t/diags1.txt

// RUN: %clang -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job -emit-obj -o %t/output.o \
// RUN:   -fconst-strings -Wincompatible-pointer-types-discards-qualifiers \
// RUN:   -serialize-diagnostic-file %t/t2.dia %s 2> %t/diags2.txt

// RUN: diff -u %t/regular-diags.txt %t/diags1.txt
// RUN: diff -u %t/regular-diags.txt %t/diags2.txt
// RUN: diff %t/regular.dia %t/t1.dia
// RUN: diff %t/regular.dia %t/t2.dia

#define STR(x) #x

void fn(char *x);
void test() {
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
  fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1)); fn(STR(something1));
}
