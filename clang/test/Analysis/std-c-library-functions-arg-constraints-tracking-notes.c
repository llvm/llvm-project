// Check the bugpath related to the reports.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-output=text \
// RUN:   -verify=bugpath

typedef typeof(sizeof(int)) size_t;

int __buf_size_arg_constraint(const void *, size_t);
void test_buf_size_concrete(void) {
  char buf[3];                       // bugpath-note{{'buf' initialized here}}
  int s = 4;                         // bugpath-note{{'s' initialized to 4}}
  __buf_size_arg_constraint(buf, s); // \
  // bugpath-warning{{The 1st argument to '__buf_size_arg_constraint' is a buffer with size 3 but should be a buffer with size equal to or greater than the value of the 2nd argument}} \
  // bugpath-note{{The 1st argument to '__buf_size_arg_constraint' is a buffer with size 3 but should be a buffer with size equal to or greater than the value of the 2nd argument}}
}

int __buf_size_arg_constraint_mul(const void *, size_t, size_t);
void test_buf_size_concrete_with_multiplication(void) {
  short buf[3];                               // bugpath-note{{'buf' initialized here}}
  int s1 = 4;                                 // bugpath-note{{'s1' initialized to 4}}
  int s2 = sizeof(short);                     // bugpath-note{{'s2' initialized to}}
  __buf_size_arg_constraint_mul(buf, s1, s2); // \
  // bugpath-warning{{The 1st argument to '__buf_size_arg_constraint_mul' is a buffer with size 6 but should be a buffer with size equal to or greater than the value of the 2nd argument (which is 4) times the 3rd argument (which is 2)}} \
  // bugpath-note{{The 1st argument to '__buf_size_arg_constraint_mul' is a buffer with size 6 but should be a buffer with size equal to or greater than the value of the 2nd argument (which is 4) times the 3rd argument (which is 2)}}
}
