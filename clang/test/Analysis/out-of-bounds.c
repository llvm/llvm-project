// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,security.ArrayBound,debug.ExprInspection -verify %s

void clang_analyzer_value(int);

// Tests doing an out-of-bounds access after the end of an array using:
// - constant integer index
// - constant integer size for buffer
void test1(int x) {
  int buf[100];
  buf[100] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ok(int x) {
  int buf[100];
  buf[99] = 1; // no-warning
}

const char test1_strings_underrun(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[-1]; // expected-warning{{Out of bound access to memory}}
}

const char test1_strings_overrun(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[1000];  // expected-warning{{Out of bound access to memory}}
}

const char test1_strings_ok(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[5]; // no-warning
}

// Tests doing an out-of-bounds access after the end of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test1_ptr(int x) {
  int buf[100];
  int *p = buf;
  p[101] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ptr_ok(int x) {
  int buf[100];
  int *p = buf;
  p[99] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test1_ptr_arith(int x) {
  int buf[100];
  int *p = buf;
  p = p + 100;
  p[0] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ptr_arith_ok(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[0] = 1; // no-warning
}

void test1_ptr_arith_bad(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[1] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ptr_arith_ok2(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[-1] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - constant integer index
// - constant integer size for buffer
void test2(int x) {
  int buf[100];
  buf[-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test2_ptr(int x) {
  int buf[100];
  int *p = buf;
  p[-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test2_ptr_arith(int x) {
  int buf[100];
  int *p = buf;
  --p;
  p[0] = 1; // expected-warning {{Out of bound access to memory preceding}}
}

// Tests doing an out-of-bounds access before the start of a multi-dimensional
// array using:
// - constant integer indices
// - constant integer sizes for the array
void test2_multi(int x) {
  int buf[100][100];
  buf[0][-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of a multi-dimensional
// array using:
// - constant integer indices
// - constant integer sizes for the array
void test2_multi_b(int x) {
  int buf[100][100];
  buf[-1][0] = 1; // expected-warning{{Out of bound access to memory}}
}

void test2_multi_ok(int x) {
  int buf[100][100];
  buf[0][0] = 1; // no-warning
}

void test3(int x) {
  int buf[100];
  if (x < 0)
    buf[x] = 1; // expected-warning{{Out of bound access to memory}}
}

void test4(int x) {
  int buf[100];
  if (x > 99)
    buf[x] = 1; // expected-warning{{Out of bound access to memory}}
}

// Don't warn when indexing below the start of a symbolic region's whose
// base extent we don't know.
int *get_symbolic(void);
void test_underflow_symbolic(void) {
  int *buf = get_symbolic();
  buf[-1] = 0; // no-warning
}

// But warn if we understand the internal memory layout of a symbolic region.
typedef struct {
  int id;
  char name[256];
} user_t;

user_t *get_symbolic_user(void);
char test_underflow_symbolic_2() {
  user_t *user = get_symbolic_user();
  return user->name[-1]; // expected-warning{{Out of bound access to memory}}
}

void test_incomplete_struct(void) {
  extern struct incomplete incomplete;
  int *p = (int *)&incomplete;
  p[1] = 42; // no-warning
}

void test_extern_void(void) {
  extern void v;
  int *p = (int *)&v;
  p[1] = 42; // no-warning
}

struct incomplete;
char test_comparison_with_extent_symbol(struct incomplete *p) {
  // Previously this was reported as a (false positive) overflow error because
  // the extent symbol of the area pointed by `p` was an unsigned and the '-1'
  // was converted to its type by `evalBinOpNN`.
  return ((char *)p)[-1]; // no-warning
}

int table[256], small_table[128];
int test_cast_to_unsigned(signed char x) {
  unsigned char y = x;
  if (x >= 0)
    return x;
  // FIXME: Here the analyzer ignores the signed -> unsigned cast, and manages to
  // load a negative value from an unsigned variable. This causes an underflow
  // report, which is an ugly false positive.
  // The underlying issue is tracked by Github ticket #39492.
  clang_analyzer_value(y); // expected-warning {{8s:{ [-128, -1] } }}
  return table[y]; // expected-warning {{Out of bound access to memory preceding}}
}

int test_cast_to_unsigned_overflow(signed char x) {
  unsigned char y = x;
  if (x >= 0)
    return x;
  // A variant of 'test_cast_to_unsigned' where the correct behavior would be
  // an overflow report (because the negative values are cast to `unsigned
  // char` values that are too large).
  // FIXME: See comment in 'test_cast_to_unsigned'.
  clang_analyzer_value(y); // expected-warning {{8s:{ [-128, -1] } }}
  return small_table[y]; // expected-warning {{Out of bound access to memory preceding}}
}

int test_negative_offset_with_unsigned_idx(void) {
  // An example where the subscript operator uses an unsigned index, but the
  // underflow report is still justified. (We should try to keep this if we
  // silence false positives like the one in 'test_cast_to_unsigned'.)
  int *p = table - 10;
  unsigned idx = 2u;
  return p[idx]; // expected-warning {{Out of bound access to memory preceding}}
}
