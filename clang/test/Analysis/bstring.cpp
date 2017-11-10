// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DUSE_BUILTINS -analyzer-checker=core,unix.cstring,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DVARIANT -analyzer-checker=core,unix.cstring,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DUSE_BUILTINS -DVARIANT -analyzer-checker=core,unix.cstring,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s

#include "Inputs/system-header-simulator-cxx.h"
#include "Inputs/system-header-simulator-for-malloc.h"

// This provides us with four possible mempcpy() definitions.
// See also comments in bstring.c.

#ifdef USE_BUILTINS
#define BUILTIN(f) __builtin_##f
#else /* USE_BUILTINS */
#define BUILTIN(f) f
#endif /* USE_BUILTINS */

#ifdef VARIANT

#define __mempcpy_chk BUILTIN(__mempcpy_chk)
void *__mempcpy_chk(void *__restrict__ s1, const void *__restrict__ s2,
                    size_t n, size_t destlen);

#define mempcpy(a,b,c) __mempcpy_chk(a,b,c,(size_t)-1)

#else /* VARIANT */

#define mempcpy BUILTIN(mempcpy)
void *mempcpy(void *__restrict__ s1, const void *__restrict__ s2, size_t n);

#endif /* VARIANT */

void clang_analyzer_eval(int);

int *testStdCopyInvalidatesBuffer(std::vector<int> v) {
  int n = v.size();
  int *buf = (int *)malloc(n * sizeof(int));

  buf[0] = 66;

  // Call to copy should invalidate buf.
  std::copy(v.begin(), v.end(), buf);

  int i = buf[0];

  clang_analyzer_eval(i == 66); // expected-warning {{UNKNOWN}}

  return buf;
}

int *testStdCopyBackwardInvalidatesBuffer(std::vector<int> v) {
  int n = v.size();
  int *buf = (int *)malloc(n * sizeof(int));
  
  buf[0] = 66;

  // Call to copy_backward should invalidate buf.
  std::copy_backward(v.begin(), v.end(), buf + n);

  int i = buf[0];

  clang_analyzer_eval(i == 66); // expected-warning {{UNKNOWN}}

  return buf;
}

namespace pr34460 {
short a;
class b {
  int c;
  long g;
  void d() {
    int e = c;
    f += e;
    mempcpy(f, &a, g);
  }
  unsigned *f;
};
}
