// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,debug.ExprInspection -verify %s

// Test functions that are called "memcpy" but aren't the memcpy
// we're looking for. Unfortunately, this test cannot be put into
// a namespace. The out-of-class weird memcpy needs to be recognized
// as a normal C function for the test to make sense.
typedef __typeof(sizeof(int)) size_t;
void *memcpy(void *, const void *, size_t);

int sprintf(char *str, const char *format, ...);
int snprintf(char *str, size_t size, const char *format, ...);

void clang_analyzer_warnIfReached();

struct S {
  static S s1, s2;

  // A weird overload within the class that accepts a structure reference
  // instead of a pointer.
  void memcpy(void *, const S &, size_t);
  void test_in_class_weird_memcpy() {
    memcpy(this, s2, 1); // no-crash
  }
};

// A similarly weird overload outside of the class.
void *memcpy(void *, const S &, size_t);

void test_out_of_class_weird_memcpy() {
  memcpy(&S::s1, S::s2, 1); // no-crash
}

template<typename... Args>
void log(const char* fmt, const Args&... args) {
  char buf[100] = {};
  auto f = snprintf;
  auto g = sprintf;
  int n = 0;
  n += f(buf, 99, fmt, args...); // no-crash: The CalleeDecl is a VarDecl, but it's okay.
  n += g(buf, fmt, args...); // no-crash: Same.
  (void)n;
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void test_gh_74269_no_crash() {
  log("%d", 1);
}
