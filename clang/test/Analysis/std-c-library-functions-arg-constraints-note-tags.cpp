// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux \
// RUN:   -analyzer-output=text \
// RUN:   -verify

template <typename T>
void clang_analyzer_express(T x);
void clang_analyzer_eval(bool);
int clang_analyzer_getExtent(void *);


// Check NotNullConstraint assumption notes.
int __not_null(int *);
int test_not_null_note(int *x, int y) {
  __not_null(x);      // expected-note{{Assuming the 1st argument to '__not_null' is not NULL}}
  if (x)              // expected-note{{'x' is non-null}} \
                      // expected-note{{Taking true branch}}
    if (!y)           // expected-note{{Assuming 'y' is 0}} \
                      // expected-note{{Taking true branch}}
      return 1 / y;   // expected-warning{{Division by zero}} \
                      // expected-note{{Division by zero}}

  return 0;
}

// Check the RangeConstraint assumption notes.
int __single_val_0(int);      // [0, 0]
int test_range_constraint_note(int x, int y) {
  __single_val_0(x);  // expected-note{{Assuming the 1st argument to '__single_val_0' is zero}}
  return y / x;       // expected-warning{{Division by zero}} \
                      // expected-note{{Division by zero}}
}

// Check the BufferSizeConstraint assumption notes.
int __buf_size_arg_constraint_concrete(const void *buf); // size of buf must be >= 10
void test_buffer_size_note(char *buf, int y) {
  __buf_size_arg_constraint_concrete(buf); // expected-note {{Assuming the size of the 1st argument to '__buf_size_arg_constraint_concrete' is equal to or greater than 10}}
  clang_analyzer_eval(clang_analyzer_getExtent(buf) >= 10); // expected-warning{{TRUE}} \
                                                            // expected-note{{TRUE}}

  // clang_analyzer_express marks the argument as interesting.
  clang_analyzer_express(buf); // expected-warning {{}} // the message does not really matter \
                               // expected-note {{}}
}

int __test_case_note();

int test_case_note_1(int y) {
  int x = __test_case_note(); // expected-note{{Function returns 0}} \
                              // expected-note{{'x' initialized here}}
  return y / x; // expected-warning{{Division by zero}} \
                // expected-note{{Division by zero}}
}

int test_case_note_2(int y) {
  int x = __test_case_note(); // expected-note{{Function returns 1}}
  return y / (x - 1); // expected-warning{{Division by zero}} \
                      // expected-note{{Division by zero}}
}
