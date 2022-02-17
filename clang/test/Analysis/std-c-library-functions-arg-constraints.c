// Check the basic reporting/warning and the application of constraints.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify=report

// Check the bugpath related to the reports.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-output=text \
// RUN:   -verify=bugpath

void clang_analyzer_eval(int);

int glob;

#define EOF -1

int isalnum(int);

void test_alnum_concrete(int v) {
  int ret = isalnum(256); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
  (void)ret;
}

void test_alnum_symbolic(int x) {
  int ret = isalnum(x);
  (void)ret;

  clang_analyzer_eval(EOF <= x && x <= 255); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{Left side of '&&' is true}} \
  // bugpath-note{{'x' is <= 255}}
}

void test_alnum_symbolic2(int x) {
  if (x > 255) { // \
    // bugpath-note{{Assuming 'x' is > 255}} \
    // bugpath-note{{Taking true branch}}

    int ret = isalnum(x); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // report-note{{}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{}} \
    // bugpath-note{{Function argument constraint is not satisfied}}

    (void)ret;
  }
}

int toupper(int);

void test_toupper_concrete(int v) {
  int ret = toupper(256); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
  (void)ret;
}

void test_toupper_symbolic(int x) {
  int ret = toupper(x);
  (void)ret;

  clang_analyzer_eval(EOF <= x && x <= 255); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{Left side of '&&' is true}} \
  // bugpath-note{{'x' is <= 255}}
}

void test_toupper_symbolic2(int x) {
  if (x > 255) { // \
    // bugpath-note{{Assuming 'x' is > 255}} \
    // bugpath-note{{Taking true branch}}

    int ret = toupper(x); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // report-note{{}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{}} \
    // bugpath-note{{Function argument constraint is not satisfied}}

    (void)ret;
  }
}

int tolower(int);

void test_tolower_concrete(int v) {
  int ret = tolower(256); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
  (void)ret;
}

void test_tolower_symbolic(int x) {
  int ret = tolower(x);
  (void)ret;

  clang_analyzer_eval(EOF <= x && x <= 255); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{Left side of '&&' is true}} \
  // bugpath-note{{'x' is <= 255}}
}

void test_tolower_symbolic2(int x) {
  if (x > 255) { // \
    // bugpath-note{{Assuming 'x' is > 255}} \
    // bugpath-note{{Taking true branch}}

    int ret = tolower(x); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // report-note{{}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{}} \
    // bugpath-note{{Function argument constraint is not satisfied}}

    (void)ret;
  }
}

int toascii(int);

void test_toascii_concrete(int v) {
  int ret = toascii(256); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
  (void)ret;
}

void test_toascii_symbolic(int x) {
  int ret = toascii(x);
  (void)ret;

  clang_analyzer_eval(EOF <= x && x <= 255); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{Left side of '&&' is true}} \
  // bugpath-note{{'x' is <= 255}}
}

void test_toascii_symbolic2(int x) {
  if (x > 255) { // \
    // bugpath-note{{Assuming 'x' is > 255}} \
    // bugpath-note{{Taking true branch}}

    int ret = toascii(x); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // report-note{{}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{}} \
    // bugpath-note{{Function argument constraint is not satisfied}}

    (void)ret;
  }
}

typedef struct FILE FILE;
typedef typeof(sizeof(int)) size_t;
size_t fread(void *restrict, size_t, size_t, FILE *restrict);
void test_notnull_concrete(FILE *fp) {
  fread(0, sizeof(int), 10, fp); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}
void test_notnull_symbolic(FILE *fp, int *buf) {
  fread(buf, sizeof(int), 10, fp);
  clang_analyzer_eval(buf != 0); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{'buf' is not equal to null}}
}
void test_notnull_symbolic2(FILE *fp, int *buf) {
  if (!buf)                          // bugpath-note{{Assuming 'buf' is null}} \
            // bugpath-note{{Taking true branch}}
    fread(buf, sizeof(int), 10, fp); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // report-note{{}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{}} \
    // bugpath-note{{Function argument constraint is not satisfied}}
}
typedef __WCHAR_TYPE__ wchar_t;
// This is one test case for the ARR38-C SEI-CERT rule.
void ARR38_C_F(FILE *file) {
  enum { BUFFER_SIZE = 1024 };
  wchar_t wbuf[BUFFER_SIZE]; // bugpath-note{{'wbuf' initialized here}}

  const size_t size = sizeof(*wbuf);   // bugpath-note{{'size' initialized to}}
  const size_t nitems = sizeof(wbuf);  // bugpath-note{{'nitems' initialized to}}

  // The 3rd parameter should be the number of elements to read, not
  // the size in bytes.
  fread(wbuf, size, nitems, file); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}

int __two_constrained_args(int, int);
void test_constraints_on_multiple_args(int x, int y) {
  // State split should not happen here. I.e. x == 1 should not be evaluated
  // FALSE.
  __two_constrained_args(x, y);
  clang_analyzer_eval(x == 1); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}}
  clang_analyzer_eval(y == 1); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}}
}

int __arg_constrained_twice(int);
void test_multiple_constraints_on_same_arg(int x) {
  __arg_constrained_twice(x);
  // Check that both constraints are applied and only one branch is there.
  clang_analyzer_eval(x < 1 || x > 2); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{Assuming 'x' is < 1}} \
  // bugpath-note{{Left side of '||' is true}}
}

int __variadic(void *stream, const char *format, ...);
void test_arg_constraint_on_variadic_fun(void) {
  __variadic(0, "%d%d", 1, 2); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}

int __buf_size_arg_constraint(const void *, size_t);
void test_buf_size_concrete(void) {
  char buf[3];                       // bugpath-note{{'buf' initialized here}}
  __buf_size_arg_constraint(buf, 4); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}
void test_buf_size_symbolic(int s) {
  char buf[3];
  __buf_size_arg_constraint(buf, s);
  clang_analyzer_eval(s <= 3); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{'s' is <= 3}}
}
void test_buf_size_symbolic_and_offset(int s) {
  char buf[3];
  __buf_size_arg_constraint(buf + 1, s);
  clang_analyzer_eval(s <= 2); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{'s' is <= 2}}
}

int __buf_size_arg_constraint_mul(const void *, size_t, size_t);
void test_buf_size_concrete_with_multiplication(void) {
  short buf[3];                                         // bugpath-note{{'buf' initialized here}}
  __buf_size_arg_constraint_mul(buf, 4, sizeof(short)); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}
void test_buf_size_symbolic_with_multiplication(size_t s) {
  short buf[3];
  __buf_size_arg_constraint_mul(buf, s, sizeof(short));
  clang_analyzer_eval(s * sizeof(short) <= 6); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}}
}
void test_buf_size_symbolic_and_offset_with_multiplication(size_t s) {
  short buf[3];
  __buf_size_arg_constraint_mul(buf + 1, s, sizeof(short));
  clang_analyzer_eval(s * sizeof(short) <= 4); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}}
}

// The minimum buffer size for this function is set to 10.
int __buf_size_arg_constraint_concrete(const void *);
void test_min_buf_size(void) {
  char buf[9];// bugpath-note{{'buf' initialized here}}
  __buf_size_arg_constraint_concrete(buf); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}
