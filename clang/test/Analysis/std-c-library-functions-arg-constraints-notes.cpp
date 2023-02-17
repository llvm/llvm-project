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
  // expected-warning{{The 1st argument to '__not_null' should not be NULL}}
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
    // expected-warning{{The size of the 1st argument to '__buf_size_arg_constraint_concrete' should be equal to or greater than 10}}
    break;
  }
  case 2: {
    char buf[3];
    __buf_size_arg_constraint(buf, 4); // \
    // expected-warning{{The size of the 1st argument to '__buf_size_arg_constraint' should be equal to or greater than the value of the 2nd argument}}
    break;
  }
  case 3: {
    char buf[3];
    __buf_size_arg_constraint_mul(buf, 4, 2); // \
    // expected-warning{{The size of the 1st argument to '__buf_size_arg_constraint_mul' should be equal to or greater than the value of the 2nd argument times the 3rd argument}}
    break;
  }
  }
}

// Check the RangeConstraint violation notes.

int __single_val_0(int); // [0, 0]
int __single_val_1(int); // [1, 1]
int __range_1_2(int); // [1, 2]
int __range_m1_1(int); // [-1, 1]
int __range_m2_m1(int); // [-2, -1]
int __range_m10_10(int); // [-10, 10]
int __range_m1_inf(int); // [-1, inf]
int __range_0_inf(int); // [0, inf]
int __range_1_inf(int); // [1, inf]
int __range_minf_m1(int); // [-inf, -1]
int __range_minf_0(int); // [-inf, 0]
int __range_minf_1(int); // [-inf, 1]
int __range_1_2__4_6(int); // [1, 2], [4, 6]
int __range_1_2__4_inf(int); // [1, 2], [4, inf]

int __single_val_out_0(int); // [0, 0]
int __single_val_out_1(int); // [1, 1]
int __range_out_1_2(int); // [1, 2]
int __range_out_m1_1(int); // [-1, 1]
int __range_out_m2_m1(int); // [-2, -1]
int __range_out_m10_10(int); // [-10, 10]
int __range_out_m1_inf(int); // [-1, inf]
int __range_out_0_inf(int); // [0, inf]
int __range_out_1_inf(int); // [1, inf]
int __range_out_minf_m1(int); // [-inf, -1]
int __range_out_minf_0(int); // [-inf, 0]
int __range_out_minf_1(int); // [-inf, 1]
int __range_out_1_2__4_6(int); // [1, 2], [4, 6]
int __range_out_1_2__4_inf(int); // [1, 2], [4, inf]

void test_range_values(int x) {
  switch (x) {
  case 0:
    __single_val_0(1); // expected-warning{{should be zero}}
    break;
  case 1:
    __single_val_1(2); // expected-warning{{should be 1}}
    break;
  case 2:
    __range_1_2(3); // expected-warning{{should be 1 or 2}}
    break;
  case 3:
    __range_m1_1(3); // expected-warning{{should be between -1 and 1}}
    break;
  case 4:
    __range_m2_m1(1); // expected-warning{{should be -2 or -1}}
    break;
  case 5:
    __range_m10_10(11); // expected-warning{{should be between -10 and 10}}
    break;
  case 6:
    __range_m10_10(-11); // expected-warning{{should be between -10 and 10}}
    break;
  case 7:
    __range_1_2__4_6(3); // expected-warning{{should be 1 or 2 or 4, 5 or 6}}
    break;
  case 8:
    __range_1_2__4_inf(3); // expected-warning{{should be 1 or 2 or >= 4}}
    break;
  }
}

void test_range_values_inf(int x) {
  switch (x) {
  case 1:
    __range_m1_inf(-2); // expected-warning{{should be >= -1}}
    break;
  case 2:
    __range_0_inf(-1); // expected-warning{{should be >= 0}}
    break;
  case 3:
    __range_1_inf(0); // expected-warning{{should be > 0}}
    break;
  case 4:
    __range_minf_m1(0); // expected-warning{{should be < 0}}
    break;
  case 5:
    __range_minf_0(1); // expected-warning{{should be <= 0}}
    break;
  case 6:
    __range_minf_1(2); // expected-warning{{should be <= 1}}
    break;
  }
}

void test_range_values_out(int x) {
  switch (x) {
  case 0:
    __single_val_out_0(0); // expected-warning{{should be nonzero}}
    break;
  case 1:
    __single_val_out_1(1); // expected-warning{{should be not equal to 1}}
    break;
  case 2:
    __range_out_1_2(2); // expected-warning{{should be not 1 and not 2}}
    break;
  case 3:
    __range_out_m1_1(-1); // expected-warning{{should be not between -1 and 1}}
    break;
  case 4:
    __range_out_m2_m1(-2); // expected-warning{{should be not -2 and not -1}}
    break;
  case 5:
    __range_out_m10_10(0); // expected-warning{{should be not between -10 and 10}}
    break;
  case 6:
    __range_out_1_2__4_6(1); // expected-warning{{should be not 1 and not 2 and not between 4 and 6}}
    break;
  case 7:
    __range_out_1_2__4_inf(4); // expected-warning{{should be not 1 and not 2 and < 4}}
    break;
  }
}

void test_range_values_out_inf(int x) {
  switch (x) {
  case 1:
    __range_out_minf_m1(-1); // expected-warning{{should be >= 0}}
    break;
  case 2:
    __range_out_minf_0(0); // expected-warning{{should be > 0}}
    break;
  case 3:
    __range_out_minf_1(1); // expected-warning{{should be > 1}}
    break;
  case 4:
    __range_out_m1_inf(-1); // expected-warning{{should be < -1}}
    break;
  case 5:
    __range_out_0_inf(0); // expected-warning{{should be < 0}}
    break;
  case 6:
    __range_out_1_inf(1); // expected-warning{{should be <= 0}}
    break;
  }
}
