// RUN: %clangxx -fsanitize=float-cast-overflow %s -o %t
// RUN: %run %t _
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t 0 2>&1 | FileCheck %s --check-prefix=CHECK-0
// RUN: %run %t 1 2>&1 | FileCheck %s --check-prefix=CHECK-1
// RUN: %run %t 2 2>&1 | FileCheck %s --check-prefix=CHECK-2
// RUN: %run %t 3 2>&1 | FileCheck %s --check-prefix=CHECK-3
// RUN: %run %t 4 2>&1 | FileCheck %s --check-prefix=CHECK-4
// RUN: %run %t 5 2>&1 | FileCheck %s --check-prefix=CHECK-5
// RUN: %run %t 6 2>&1 | FileCheck %s --check-prefix=CHECK-6
// FIXME: %run %t 7 2>&1 | FileCheck %s --check-prefix=CHECK-7
// FIXME: not %run %t 8 2>&1 | FileCheck %s --check-prefix=CHECK-8
// RUN: not %run %t 9 2>&1 | FileCheck %s --check-prefix=CHECK-9

// This test assumes float and double are IEEE-754 single- and double-precision.

#if defined(__APPLE__)
# include <machine/endian.h>
# define BYTE_ORDER __DARWIN_BYTE_ORDER
# define BIG_ENDIAN __DARWIN_BIG_ENDIAN
# define LITTLE_ENDIAN __DARWIN_LITTLE_ENDIAN
#elif defined(__FreeBSD__) || defined(__NetBSD__)
# include <sys/endian.h>
# ifndef BYTE_ORDER
#  define BYTE_ORDER _BYTE_ORDER
# endif
# ifndef BIG_ENDIAN
#  define BIG_ENDIAN _BIG_ENDIAN
# endif
# ifndef LITTLE_ENDIAN
#  define LITTLE_ENDIAN _LITTLE_ENDIAN
# endif
#elif defined(__sun__) && defined(__svr4__)
// Solaris provides _BIG_ENDIAN/_LITTLE_ENDIAN selector in sys/types.h.
# include <sys/types.h>
# define BIG_ENDIAN 4321
# define LITTLE_ENDIAN 1234
# if defined(_BIG_ENDIAN)
#  define BYTE_ORDER BIG_ENDIAN
# else
#  define BYTE_ORDER LITTLE_ENDIAN
# endif
#elif defined(_WIN32)
# define BYTE_ORDER 0
# define BIG_ENDIAN 1
# define LITTLE_ENDIAN 0
#else
# include <endian.h>
# define BYTE_ORDER __BYTE_ORDER
# define BIG_ENDIAN __BIG_ENDIAN
# define LITTLE_ENDIAN __LITTLE_ENDIAN
#endif  // __APPLE__
#include <stdint.h>
#include <stdio.h>
#include <string.h>

float Inf;
float NaN;

int main(int argc, char **argv) {
  float MaxFloatRepresentableAsInt = 0x7fffff80;
  (int)MaxFloatRepresentableAsInt; // ok
  (int)-MaxFloatRepresentableAsInt; // ok

  float MinFloatRepresentableAsInt = -0x7fffffff - 1;
  (int)MinFloatRepresentableAsInt; // ok

  float MaxFloatRepresentableAsUInt = 0xffffff00u;
  (unsigned int)MaxFloatRepresentableAsUInt; // ok

#ifdef __SIZEOF_INT128__
  unsigned __int128 FloatMaxAsUInt128 = -((unsigned __int128)1 << 104);
  (void)(float)FloatMaxAsUInt128; // ok
#endif

  float NearlyMinusOne = -0.99999;
  unsigned Zero = NearlyMinusOne; // ok

  // Build a '+Inf'.
#if BYTE_ORDER == LITTLE_ENDIAN
  unsigned char InfVal[] = { 0x00, 0x00, 0x80, 0x7f };
#else
  unsigned char InfVal[] = { 0x7f, 0x80, 0x00, 0x00 };
#endif
  float Inf;
  memcpy(&Inf, InfVal, 4);

  // Build a 'NaN'.
#if BYTE_ORDER == LITTLE_ENDIAN
  unsigned char NaNVal[] = { 0x01, 0x00, 0x80, 0x7f };
#else
  unsigned char NaNVal[] = { 0x7f, 0x80, 0x00, 0x01 };
#endif
  float NaN;
  memcpy(&NaN, NaNVal, 4);

  double DblInf = (double)Inf; // ok

  switch (argv[1][0]) {
    // FIXME: Produce a source location for these checks and test for it here.

    // Floating point -> integer overflow.
  case '0': {
    // Note that values between 0x7ffffe00 and 0x80000000 may or may not
    // successfully round-trip, depending on the rounding mode.
    // CHECK-0: {{.*}}cast-overflow.cpp:[[@LINE+1]]:27: runtime error: 2.14748{{.*}} is outside the range of representable values of type 'int'
    static int test_int = MaxFloatRepresentableAsInt + 0x80;
    // CHECK-0: SUMMARY: {{.*}}Sanitizer: float-cast-overflow {{.*}}cast-overflow.cpp:[[@LINE-1]]
    return 0;
    }
  case '1': {
    // CHECK-1: {{.*}}cast-overflow.cpp:[[@LINE+1]]:27: runtime error: -2.14748{{.*}} is outside the range of representable values of type 'int'
    static int test_int = MinFloatRepresentableAsInt - 0x100;
    return 0;
  }
  case '2': {
    // CHECK-2: {{.*}}cast-overflow.cpp:[[@LINE+2]]:37: runtime error: -1 is outside the range of representable values of type 'unsigned int'
    volatile float f = -1.0;
    volatile unsigned u = (unsigned)f;
    return 0;
  }
  case '3': {
    // CHECK-3: {{.*}}cast-overflow.cpp:[[@LINE+1]]:37: runtime error: 4.2949{{.*}} is outside the range of representable values of type 'unsigned int'
    static int test_int = (unsigned)(MaxFloatRepresentableAsUInt + 0x100);
    return 0;
  }

  case '4': {
    // CHECK-4: {{.*}}cast-overflow.cpp:[[@LINE+1]]:27: runtime error: {{.*}} is outside the range of representable values of type 'int'
    static int test_int = Inf;
    return 0;
  }
  case '5': {
    // CHECK-5: {{.*}}cast-overflow.cpp:[[@LINE+1]]:27: runtime error: {{.*}} is outside the range of representable values of type 'int'
    static int test_int = NaN;
    return 0;
  }

    // Integer -> floating point overflow.
  case '6': {
    // CHECK-6: cast-overflow.cpp:[[@LINE+2]]:{{27: runtime error: 3.40282e\+38 is outside the range of representable values of type 'int'| __int128 not supported}}
#if defined(__SIZEOF_INT128__) && !defined(_WIN32)
    static int test_int = (float)(FloatMaxAsUInt128 + 1);
    return 0;
#else
    // Print the same line as the check above. That way the test is robust to
    // line changes around it
    printf("%s:%d: __int128 not supported", __FILE__, __LINE__ - 5);
    return 0;
#endif
  }
  // FIXME: The backend cannot lower __fp16 operations on x86 yet.
  //case '7':
  //  (__fp16)65504; // ok
  //  // CHECK-7: runtime error: 65505 is outside the range of representable values of type '__fp16'
  //  return (__fp16)65505;

    // Floating point -> floating point overflow.
  case '8':
    // CHECK-8: {{.*}}cast-overflow.cpp:[[@LINE+1]]:19: runtime error: 1e+39 is outside the range of representable values of type 'float'
    return (float)1e39;
  case '9':
    volatile long double ld = 300.0;
    // CHECK-9: {{.*}}cast-overflow.cpp:[[@LINE+1]]:14: runtime error: 300 is outside the range of representable values of type 'char'
    char c = ld;
    return c;
  }
}
