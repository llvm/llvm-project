// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false -triple x86_64-pc-linux-gnu %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false -triple i386-pc-linux-gnu  %s

int clang_analyzer_eval(int);

typedef struct {
  int a : 1;
  int b[2];
} BITFIELD_CAST;

void array_struct_bitfield_1() {
  BITFIELD_CAST ff = {0};
  BITFIELD_CAST *pff = &ff;
  clang_analyzer_eval(*((int *)pff + 1) == 0); // expected-warning{{TRUE}}
  ff.b[0] = 3;
  clang_analyzer_eval(*((int *)pff + 1) == 3); // expected-warning{{TRUE}}
}

int array_struct_bitfield_2() {
  BITFIELD_CAST ff = {0};
  BITFIELD_CAST *pff = &ff;
  int a = *((int *)pff + 2); // expected-warning{{Assigned value is garbage or undefined [core.uninitialized.Assign]}}
  return a;
}

typedef struct {
  unsigned int a : 1;
  unsigned int x : 31;
  unsigned int c : 1;
  int b[2];
} mystruct;

void array_struct_bitfield_3() {
  mystruct ff;
  mystruct *pff = &ff;
  ff.b[0] = 3;
  clang_analyzer_eval(*((int *)pff + 2) == 3); // expected-warning{{TRUE}}
}
