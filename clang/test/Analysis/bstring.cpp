// DEFINE: %{analyzer} = %clang_analyze_cc1 \
// DEFINE:     -analyzer-checker=core \
// DEFINE:     -analyzer-checker=unix.cstring \
// DEFINE:     -analyzer-checker=unix.Malloc \
// DEFINE:     -analyzer-checker=debug.ExprInspection \
// DEFINE:     -analyzer-config eagerly-assume=false

// All alpha.unix.cstring subcheckers enabled (OutOfBounds sinks at OOB memset).
// RUN: %{analyzer} -analyzer-checker=alpha.unix.cstring \
// RUN:     -verify=expected,oob,uninit %s
// RUN: %{analyzer} -analyzer-checker=alpha.unix.cstring \
// RUN:     -DUSE_BUILTINS -verify=expected,oob,uninit %s
// RUN: %{analyzer} -analyzer-checker=alpha.unix.cstring \
// RUN:     -DVARIANT -verify=expected,oob,uninit %s
// RUN: %{analyzer} -analyzer-checker=alpha.unix.cstring \
// RUN:     -DUSE_BUILTINS -DVARIANT -verify=expected,oob,uninit %s

// OutOfBounds disabled: OOB memset doesn't sink, analysis continues.
// RUN: %{analyzer} \
// RUN:     -analyzer-checker=alpha.unix.cstring.BufferOverlap \
// RUN:     -analyzer-checker=unix.cstring.NotNullTerminated \
// RUN:     -verify=expected,no-oob %s

// UninitializedRead enabled without OutOfBounds: verifies that
// UninitializedRead works independently of OutOfBounds.
// RUN: %{analyzer} \
// RUN:     -analyzer-checker=alpha.unix.cstring.UninitializedRead \
// RUN:     -verify=expected,no-oob,uninit %s

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

void memset2_inheritance_field() {
  Derived d;
  // FIXME: OOB memset on a derived field with sizeof(Derived).
  // Current behavior: with 'oob' the analysis sinks; with 'no-oob' it
  // continues and the evals produce UNKNOWN.
  // Expected behavior: the OOB error is fatal regardless of whether the
  // OutOfBounds checker frontend is enabled, so the evals should be unreachable
  // in all configurations. The 'no-oob' expectations below should be removed.
  memset(&d.d_mem, 0, sizeof(Derived)); // oob-warning{{overflows the destination buffer}}
  clang_analyzer_eval(d.b_mem == 0); // no-oob-warning{{UNKNOWN}}
  clang_analyzer_eval(d.d_mem == 0); // no-oob-warning{{UNKNOWN}}
}

void memset3_inheritance_field() {
  Derived d;
  // FIXME: memset on the base field with sizeof(Derived). By the letter of
  // the standard this is UB, but practically this only touches memory it is
  // supposed to with the above class definitions.
  // Current behavior: with 'oob' the analysis sinks. With 'no-oob' the fields are
  // treated as correctly set to 0.
  // Expected behavior: same as memset2. The OOB error should be fatal in all
  // configurations and the 'no-oob' expectations should be removed.
  memset(&d.b_mem, 0, sizeof(Derived)); // oob-warning{{overflows the destination buffer}}
  clang_analyzer_eval(d.b_mem == 0); // no-oob-warning{{TRUE}}
  clang_analyzer_eval(d.d_mem == 0); // no-oob-warning{{TRUE}}
}

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

void memset8_virtual_inheritance_field() {
  DerivedVirtual d;
  // FIXME: Same as memset3, but the base has a virtual function. In typical
  // implementations &d.b_mem differs from &d because the vtable pointer
  // precedes the first member, so this may also write past the object's
  // extent.
  // Current behavior: with 'oob' the analysis sinks. With 'no-oob' the fields
  // are treated as UNKNOWN.
  // Expected behavior: same as memset2. The OOB error should be fatal in all
  // configurations and the 'no-oob' expectations should be removed.
  memset(&d.b_mem, 0, sizeof(Derived)); // oob-warning{{overflows the destination buffer}}
  clang_analyzer_eval(d.b_mem == 0); // no-oob-warning{{UNKNOWN}}
  clang_analyzer_eval(d.d_mem == 0); // no-oob-warning{{UNKNOWN}}
}

} // namespace memset_non_pod

void memset1_new_array() {
  int *array = new int[10];
  memset(array, 0, 10 * sizeof(int));
  clang_analyzer_eval(array[2] == 0); // expected-warning{{TRUE}}
  // FIXME: OOB memset on a heap array.
  // Current behavior: with 'oob' the analysis sinks. With 'no-oob' it
  // continues and the eval produces UNKNOWN.
  // Expected behavior: same as memset2. The OOB error should be fatal in all
  // configurations and the 'no-oob' expectation should be removed.
  memset(array + 1, 'a', 10 * sizeof(9)); // oob-warning{{overflows the destination buffer}}
  clang_analyzer_eval(array[2] == 0); // no-oob-warning{{UNKNOWN}}
  delete[] array;
}

void memmove_uninit_without_outofbound() {
  int src[4];
  int dst[4];
  // This test verifies that UninitializedRead produces warnings even when
  // OutOfBounds is disabled. Previously, CheckBufferAccess would early-return
  // before reaching checkInit() when OutOfBounds was disabled, suppressing
  // UninitializedRead as a side effect.
  memmove(dst, src, sizeof(src)); // uninit-warning{{The first element of the 2nd argument is undefined}}
                                  // uninit-note@-1{{Other elements might also be undefined}}
}
