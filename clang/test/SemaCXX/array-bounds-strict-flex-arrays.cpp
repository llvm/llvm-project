// RUN: %clang_cc1 -verify -fstrict-flex-arrays=2 %s

// We cannot know for sure the size of a flexible array.
struct t {
  int f;
  int a[];
};
void test(t *s2) {
  s2->a[2] = 0; // no-warning
}

// Under -fstrict-flex-arrays `a` is not a flexible array.
struct t1 {
  int f;
  int a[1]; // expected-note {{array 'a' declared here}}
};
void test1(t1 *s2) {
  s2->a[2] = 0; // expected-warning {{array index 2 is past the end of the array (which contains 1 element)}}
}

// Under -fstrict-flex-arrays `a` is a flexible array.
struct t2 {
  int f;
  int a[0];
};
void test1(t2 *s2) {
  s2->a[2] = 0; // no-warning
}
