// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux \
// RUN:   -verify

// In this test we verify that each argument constraints are described properly.

// Check NotNullConstraint violation notes.
int __not_null(int *);
void test_not_null(int *x) {
  __not_null(nullptr); // \
  // expected-note{{The 1st argument should not be NULL}} \
  // expected-warning{{}}
}

// Check the BufferSizeConstraint violation notes.
using size_t = decltype(sizeof(int));
int __buf_size_arg_constraint_concrete(const void *); // size <= 10
int __buf_size_arg_constraint(const void *, size_t);  // size <= Arg1
int __buf_size_arg_constraint_mul(const void *, size_t, size_t); // size <= Arg1 * Arg2
void test_buffer_size(int x) {
  switch (x) {
  case 1: {
    char buf[9];
    __buf_size_arg_constraint_concrete(buf); // \
    // expected-note{{The size of the 1st argument should be equal to or greater than the value of 10}} \
    // expected-warning{{}}
    break;
  }
  case 2: {
    char buf[3];
    __buf_size_arg_constraint(buf, 4); // \
    // expected-note{{The size of the 1st argument should be equal to or greater than the value of the 2nd arg}} \
    // expected-warning{{}}
    break;
  }
  case 3: {
    char buf[3];
    __buf_size_arg_constraint_mul(buf, 4, 2); // \
    // expected-note{{The size of the 1st argument should be equal to or greater than the value of the 2nd argument times the 3rd argument}} \
    // expected-warning{{}}
    break;
  }
  }
}

// Check the RangeConstraint violation notes.
int __single_val_1(int);      // [1, 1]
int __range_1_2(int);         // [1, 2]
int __range_1_2__4_5(int);    // [1, 2], [4, 5]
void test_range(int x) {
    __single_val_1(2); // \
    // expected-note{{The 1st argument should be within the range [1, 1]}} \
    // expected-warning{{}}
}
// Do more specific check against the range strings.
void test_range_values(int x) {
  switch (x) {
    case 1:
      __single_val_1(2);      // expected-note{{[1, 1]}} \
                              // expected-warning{{}}
      break;
    case 2:
      __range_1_2(3);         // expected-note{{[1, 2]}} \
                              // expected-warning{{}}
      break;
    case 3:
      __range_1_2__4_5(3);    // expected-note{{[[1, 2], [4, 5]]}} \
                              // expected-warning{{}}
      break;
  }
}
// Do more specific check against the range kinds.
int __within(int);       // [1, 1]
int __out_of(int);       // [1, 1]
void test_range_kind(int x) {
  switch (x) {
    case 1:
      __within(2);       // expected-note{{within}} \
                         // expected-warning{{}}
      break;
    case 2:
      __out_of(1);       // expected-note{{out of}} \
                         // expected-warning{{}}
      break;
  }
}

