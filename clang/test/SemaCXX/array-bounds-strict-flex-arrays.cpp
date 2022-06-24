// RUN: %clang_cc1 -verify -fstrict-flex-arrays=3 %s

// We cannot know for sure the size of a flexible array.
void test() {
  struct {
    int f;
    int a[];
  } s2;
  s2.a[2] = 0; // no-warning
}

// Under -fstrict-flex-arrays `a` is not a flexible array.
void test1() {
  struct {
    int f;
    int a[1]; // expected-note {{declared here}}
  } s2;
  s2.a[2] = 0; // expected-warning 1 {{array index 2 is past the end of the array (which contains 1 element)}}
}
