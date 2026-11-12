// RUN: %clang_cc1 -verify=relaxed                    -fstrict-flex-arrays=1 %s
// RUN: %clang_cc1 -verify=relaxed,strict             -fstrict-flex-arrays=2 %s
// RUN: %clang_cc1 -verify=relaxed,strict,very-strict -fstrict-flex-arrays=3 %s

// We cannot know for sure the size of a flexible array.
struct t {
  int f;
  int a[];
};
void test(t *s2) {
  s2->a[2] = 0; // no-warning
}

// Under -fstrict-flex-arrays={1,2,3} `a` is not a flexible array
struct t0 {
  int f;
  int a[10]; // relaxed-note {{array 'a' declared here}}
};
void test0(t0 *s2) {
  s2->a[12] = 0; // relaxed-warning {{array index 12 is past the end of the array (that has type 'int[10]')}}
}


// Under -fstrict-flex-arrays=2 `a` is not a flexible array, but it is under -fstrict-flex-arrays=1
struct t1 {
  int f;
  int a[1]; // strict-note {{array 'a' declared here}}
};
void test1(t1 *s2) {
  s2->a[2] = 0; // strict-warning {{array index 2 is past the end of the array (that has type 'int[1]')}}
}

// Under -fstrict-flex-arrays={1,2} `a` is a flexible array, but not under -fstrict-flex-arrays=3.
struct t2 {
  int f;
  int a[0]; // very-strict-note {{array 'a' declared here}}
};
void test1(t2 *s2) {
  s2->a[2] = 0; // very-strict-warning {{array index 2 is past the end of the array (that has type 'int[0]')}}
}
