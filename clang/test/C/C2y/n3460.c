// RUN: %clang_cc1 -verify -std=c2y -Wall %s

/* WG14 N3460: Clang 12
 * Complex operators
 *
 * This moves some Annex G requirements into the main body of the standard.
 */

// CMPLX(0.0, inf) * 2.0, the result should be CMPLX(0.0, inf), not CMPLX(nan, inf)
static_assert(__builtin_complex(0.0, __builtin_inf()) * 2.0 ==
              __builtin_complex(0.0, __builtin_inf()));

// CMPLX(0.0, 1.0) * -0.0 is CMPLX(-0.0, -0.0), not CMPLX(-0.0, +0.0)
static_assert(__builtin_complex(0.0, 1.0) * -0.0 ==
              __builtin_complex(-0.0, -0.0));

// Testing for -0.0 is a pain because -0.0 == +0.0, so forcefully generate a
// diagnostic and check the note.
static_assert(__builtin_complex(0.0, 1.0) * -0.0 == 1); /* expected-error {{static assertion failed due to requirement '__builtin_complex(0., 1.) * -0. == 1'}} \
                                                           expected-note {{expression evaluates to '(-0 + -0i) == 1'}}
                                                         */

// CMPLX(0.0, inf) / 2.0, the result should be CMPLX(0.0, inf),
// not CMPLX(nan, inf)
static_assert(__builtin_complex(0.0, __builtin_inf()) / 2.0 ==
              __builtin_complex(0.0, __builtin_inf()));

// CMPLX(2.0, 3.0) * 2.0, the result should be CMPLX(4.0, 6.0)
static_assert(__builtin_complex(2.0, 3.0) * 2.0 ==
              __builtin_complex(4.0, 6.0));

// CMPLX(2.0, 4.0) / 2.0, the result should be CMPLX(1.0, 2.0)
static_assert(__builtin_complex(2.0, 4.0) / 2.0 ==
              __builtin_complex(1.0, 2.0));

// CMPLX(2.0, 3.0) * CMPLX(4.0, 5.0), the result should be
// CMPLX(8.0 - 15.0, 12.0 + 10.0)
static_assert(__builtin_complex(2.0, 3.0) * __builtin_complex(4.0, 5.0) ==
              __builtin_complex(-7.0, 22.0));

// CMPLX(2.0, 3.0) / CMPLX(4.0, 5.0), the result should be
// CMPLX((8.0 + 15.0)/(4.0^2 + 5.0^2), (12.0 - 10.0)/(4.0^2 + 5.0^2))
static_assert(__builtin_complex(2.0, 3.0) / __builtin_complex(4.0, 5.0) ==
              __builtin_complex(23.0 / 41.0, 2.0 / 41.0));


// 2.0 / CMPLX(2.0, 4.0), the result should be
// CMPLX(4.0 /(2.0^2 + 4.0^2), -8.0/(2.0^2 + 4.0^2))
static_assert(2.0 / __builtin_complex(2.0, 4.0) ==
              __builtin_complex(4.0 / 20.0, -8.0 / 20.0));

