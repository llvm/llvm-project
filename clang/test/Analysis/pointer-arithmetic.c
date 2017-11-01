// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

int test1() {
  int *p = (int *)sizeof(int);
  p -= 1;
  return *p; // expected-warning {{Dereference of null pointer}}
}

int test2() {
  int *p = (int *)sizeof(int);
  p -= 2;
  p += 1;
  return *p; // expected-warning {{Dereference of null pointer}}
}

int test3() {
  int *p = (int *)sizeof(int);
  p++;
  p--;
  p--;
  return *p; // expected-warning {{Dereference of null pointer}}
}

int test4() {
  // This is a special case where pointer arithmetic is not calculated to
  // preserve useful warnings on dereferences of null pointers.
  int *p = 0;
  p += 1;
  return *p; // expected-warning {{Dereference of null pointer}}
}
