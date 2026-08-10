// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

/* WG14 N1464: ???
 * Creation of complex value
 */

// The paper is about the CMPLX macros, which Clang supports via the
// __builtin_complex builtin function. Test the basic functionality.
_Static_assert(__builtin_complex(5.0, 2.0) == 5.0 + 1.0j * 2.0, "");
_Static_assert(__builtin_complex(5.0f, 2.0f) == 5.0f + 1.0j * 2.0f, "");
_Static_assert(__builtin_complex(5.0L, 2.0L) == 5.0L + 1.0j * 2.0L, "");

// Test the edge case involving NaN or infinity.
#define INF(type) (type)__builtin_inf()
#define NAN(type) (type)__builtin_nan("")
_Static_assert(__builtin_complex(5.0f, INF(float)) != 5.0f + 1.0j * INF(float), "");
_Static_assert(__builtin_complex(5.0, INF(double)) != 5.0 + 1.0j * INF(double), "");
_Static_assert(__builtin_complex(5.0L, INF(long double)) != 5.0L + 1.0j * INF(long double), "");
_Static_assert(__builtin_complex(5.0f, NAN(float)) != 5.0f + 1.0j * NAN(float), "");
_Static_assert(__builtin_complex(5.0, NAN(double)) != 5.0 + 1.0j * NAN(double), "");
_Static_assert(__builtin_complex(5.0L, NAN(long double)) != 5.0L + 1.0j * NAN(long double), "");
