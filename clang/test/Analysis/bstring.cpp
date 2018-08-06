// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring,unix.Malloc,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DUSE_BUILTINS -analyzer-checker=core,unix.cstring,unix.Malloc,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DVARIANT -analyzer-checker=core,unix.cstring,alpha.unix.cstring,unix.Malloc,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DUSE_BUILTINS -DVARIANT -analyzer-checker=core,unix.cstring,alpha.unix.cstring,unix.Malloc,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DSUPPRESS_OUT_OF_BOUND -analyzer-checker=core,unix.cstring,unix.Malloc,alpha.unix.cstring.BufferOverlap,alpha.unix.cstring.NotNullTerminated,debug.ExprInspection -analyzer-store=region -verify %s

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

void *memset(void *dest, int ch, std::size_t count);
namespace memset_non_pod {
class Base {
public:
  int b_mem;
  Base() : b_mem(1) {}
};

class Derived : public Base {
public:
  int d_mem;
  Derived() : d_mem(2) {}
};

void memset1_inheritance() {
  Derived d;
  memset(&d, 0, sizeof(Derived));
  clang_analyzer_eval(d.b_mem == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(d.d_mem == 0); // expected-warning{{TRUE}}
}

#ifdef SUPPRESS_OUT_OF_BOUND
void memset2_inheritance_field() {
  Derived d;
  memset(&d.d_mem, 0, sizeof(Derived));
  clang_analyzer_eval(d.b_mem == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d.d_mem == 0); // expected-warning{{UNKNOWN}}
}

void memset3_inheritance_field() {
  Derived d;
  memset(&d.b_mem, 0, sizeof(Derived));
  clang_analyzer_eval(d.b_mem == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(d.d_mem == 0); // expected-warning{{TRUE}}
}
#endif

void memset4_array_nonpod_object() {
  Derived array[10];
  clang_analyzer_eval(array[1].b_mem == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(array[1].d_mem == 2); // expected-warning{{UNKNOWN}}
  memset(&array[1], 0, sizeof(Derived));
  clang_analyzer_eval(array[1].b_mem == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(array[1].d_mem == 0); // expected-warning{{UNKNOWN}}
}

void memset5_array_nonpod_object() {
  Derived array[10];
  clang_analyzer_eval(array[1].b_mem == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(array[1].d_mem == 2); // expected-warning{{UNKNOWN}}
  memset(array, 0, sizeof(array));
  clang_analyzer_eval(array[1].b_mem == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[1].d_mem == 0); // expected-warning{{TRUE}}
}

void memset6_new_array_nonpod_object() {
  Derived *array = new Derived[10];
  clang_analyzer_eval(array[2].b_mem == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(array[2].d_mem == 2); // expected-warning{{UNKNOWN}}
  memset(array, 0, 10 * sizeof(Derived));
  clang_analyzer_eval(array[2].b_mem == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[2].d_mem == 0); // expected-warning{{TRUE}}
  delete[] array;
}

void memset7_placement_new() {
  Derived *d = new Derived();
  clang_analyzer_eval(d->b_mem == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(d->d_mem == 2); // expected-warning{{TRUE}}

  memset(d, 0, sizeof(Derived));
  clang_analyzer_eval(d->b_mem == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(d->d_mem == 0); // expected-warning{{TRUE}}

  Derived *d1 = new (d) Derived();
  clang_analyzer_eval(d1->b_mem == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(d1->d_mem == 2); // expected-warning{{TRUE}}

  memset(d1, 0, sizeof(Derived));
  clang_analyzer_eval(d->b_mem == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(d->d_mem == 0); // expected-warning{{TRUE}}
}

class BaseVirtual {
public:
  int b_mem;
  virtual int get() { return 1; }
};

class DerivedVirtual : public BaseVirtual {
public:
  int d_mem;
};

#ifdef SUPPRESS_OUT_OF_BOUND
void memset8_virtual_inheritance_field() {
  DerivedVirtual d;
  memset(&d.b_mem, 0, sizeof(Derived));
  clang_analyzer_eval(d.b_mem == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d.d_mem == 0); // expected-warning{{UNKNOWN}}
}
#endif
} // namespace memset_non_pod

#ifdef SUPPRESS_OUT_OF_BOUND
void memset1_new_array() {
  int *array = new int[10];
  memset(array, 0, 10 * sizeof(int));
  clang_analyzer_eval(array[2] == 0); // expected-warning{{TRUE}}
  memset(array + 1, 'a', 10 * sizeof(9));
  clang_analyzer_eval(array[2] == 0); // expected-warning{{UNKNOWN}}
  delete[] array;
}
#endif
