// ParamTLS has limited size. Everything that does not fit is considered fully
// initialized.

// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=2 -O0 %s -o %t && %run %t
//
// AArch64 and LoongArch64 fail with:
// void f801(S<801>): Assertion `__msan_test_shadow(&s, sizeof(s)) == -1' failed
// XFAIL: target={{(aarch64|loongarch64).*}}
// When passing huge structs by value, SystemZ uses pointers, therefore this
// test in its present form is unfortunately not applicable.
// ABI says: "A struct or union of any other size <snip>. Replace such an
// argument by a pointer to the object, or to a copy where necessary to enforce
// call-by-value semantics."
// XFAIL: target=s390x{{.*}}

#include <sanitizer/msan_interface.h>
#include <assert.h>

// This test assumes that ParamTLS size is 800 bytes.

// This test passes poisoned values through function argument list.
// In case of overflow, argument is unpoisoned.
#define OVERFLOW(x) assert(__msan_test_shadow(&x, sizeof(x)) == -1)
// In case of no overflow, it is still poisoned.
#define NO_OVERFLOW(x) assert(__msan_test_shadow(&x, sizeof(x)) == 0)

#if defined(__x86_64__)
// In x86_64, if argument is partially outside tls, it is considered completely
// unpoisoned
#define PARTIAL_OVERFLOW(x) OVERFLOW(x)
#else
// In other archs, bigger arguments are splitted in multiple IR arguments, so
// they are considered poisoned till tls limit. Checking last byte of such arg:
#define PARTIAL_OVERFLOW(x) assert(__msan_test_shadow((char *)(&(x) + 1) - 1, 1) == -1)
#endif


template<int N>
struct S {
  char x[N];
};

void f100(S<100> s) {
  NO_OVERFLOW(s);
}

void f800(S<800> s) {
  NO_OVERFLOW(s);
}

void f801(S<801> s) {
  PARTIAL_OVERFLOW(s);
}

void f1000(S<1000> s) {
  PARTIAL_OVERFLOW(s);
}

void f_many(int a, double b, S<800> s, int c, double d) {
  NO_OVERFLOW(a);
  NO_OVERFLOW(b);
  PARTIAL_OVERFLOW(s);
  OVERFLOW(c);
  OVERFLOW(d);
}

// -8 bytes for "int a", aligned by 8
// -2 to make "int c" a partial fit
void f_many2(int a, S<800 - 8 - 2> s, int c, double d) {
  NO_OVERFLOW(a);
  NO_OVERFLOW(s);
  PARTIAL_OVERFLOW(c);
  OVERFLOW(d);
}

int main(void) {
  S<100> s100;
  S<800> s800;
  S<801> s801;
  S<1000> s1000;
  f100(s100);
  f800(s800);
  f801(s801);
  f1000(s1000);

  int i;
  double d;
  f_many(i, d, s800, i, d);

  S<800 - 8 - 2> s788;
  f_many2(i, s788, i, d);
  return 0;
}
