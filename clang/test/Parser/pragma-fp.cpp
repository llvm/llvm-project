// RUN: %clang_cc1 -std=c++11 -verify %s

void test_0(int *List, int Length) {
/* expected-error@+1 {{missing option; expected 'contract', 'reassociate' or 'exceptions'}} */
#pragma clang fp
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}
void test_1(int *List, int Length) {
/* expected-error@+1 {{invalid option 'blah'; expected 'contract', 'reassociate' or 'exceptions'}} */
#pragma clang fp blah
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test_3(int *List, int Length) {
/* expected-error@+1 {{expected '('}} */
#pragma clang fp contract on
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test_4(int *List, int Length) {
/* expected-error@+1 {{unexpected argument 'while' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}} */
#pragma clang fp contract(while)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test_5(int *List, int Length) {
/* expected-error@+1 {{unexpected argument 'maybe' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}} */
#pragma clang fp contract(maybe)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test_6(int *List, int Length) {
/* expected-error@+1 {{expected ')'}} */
#pragma clang fp contract(fast
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test_7(int *List, int Length) {
/* expected-warning@+1 {{extra tokens at end of '#pragma clang fp' - ignored}} */
#pragma clang fp contract(fast) *
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test_8(int *List, int Length) {
  for (int i = 0; i < Length; i++) {
    List[i] = i;
/* expected-error@+1 {{'#pragma clang fp' can only appear at file scope or at the start of a compound statement}} */
#pragma clang fp contract(fast)
  }
}

void test_9(float *dest, float a, float b) {
/* expected-error@+1 {{unexpected argument 'on' to '#pragma clang fp exceptions'; expected 'ignore', 'maytrap' or 'strict'}} */
#pragma clang fp exceptions(on)
  *dest = a + b;
}

void test_10(float *dest, float a, float b) {
#pragma clang fp exceptions(maytrap) contract(fast) reassociate(on)
  *dest = a + b;
}
