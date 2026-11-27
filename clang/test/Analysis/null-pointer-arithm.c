// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core

extern int *get_pointer();

int *test_add1(int offset) {
  int *p = get_pointer();
  if (p) {}
  return p + offset; // expected-warning{{Addition of a null pointer (from variable 'p') and a probably nonzero integer value (from variable 'offset') may result in undefined behavior}}
}

int *test_add2(int offset) {
  int *p = get_pointer();
  if (p) {}
  if (offset) {}
  return p + offset; // expected-warning{{Addition of a null pointer (from variable 'p') and a nonzero integer value (from variable 'offset') results in undefined behavior}}
}

int *test_add3(int offset) {
  int *p = get_pointer();
  if (p) {}
  if (offset != 0) return 0;
  return p + offset;
}

int *test_add4(int offset) {
  int *p = get_pointer();
  if (p) {}
  if (offset == 0) return 0;
  return p + offset; // expected-warning{{Addition of a null pointer (from variable 'p') and a nonzero integer value (from variable 'offset') results in undefined behavior}}
}

int *test_add5(int offset) {
  int *p = get_pointer();
  if (p) {}
  return offset + p; // expected-warning{{Addition of a probably nonzero integer value (from variable 'offset') and a null pointer (from variable 'p') may result in undefined behavior}}
}

int *test_sub1(int offset) {
  int *p = get_pointer();
  if (p) {}
  return p - offset; // expected-warning{{Subtraction of a null pointer (from variable 'p') and a probably nonzero integer value (from variable 'offset') may result in undefined behavior}}
}

int test_sub_p1() {
  int *p = get_pointer();
  if (p) {}
  return p - p;
}

int test_sub_p2() {
  int *p1 = get_pointer();
  int *p2 = get_pointer();
  if (p1) {}
  if (p2) {}
  return p1 - p2;
  // expected-warning@-1{{Subtraction of a non-null pointer (from variable 'p1') and a null pointer (from variable 'p2') results in undefined behavior}}
  // expected-warning@-2{{Subtraction of a null pointer (from variable 'p1') and a non-null pointer (from variable 'p2') results in undefined behavior}}
}

int test_sub_p3() {
  int *p1 = get_pointer();
  int *p2 = get_pointer();
  if (p1) {}
  return p1 - p2; // expected-warning{{Subtraction of a null pointer (from variable 'p1') and a probably non-null pointer (from variable 'p2') may result in undefined behavior}}
}

struct S {
  char *p;
  int offset;
};

char *test_struct(struct S s) {
  if (s.p) {}
  return s.p + s.offset; // expected-warning{{Addition of a null pointer (via field 'p') and a probably nonzero integer value (via field 'offset') may result in undefined behavior}}
}
