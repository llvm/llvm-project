// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify
//
// Test the constant folding of builtin complex numbers.

static_assert((0.0 + 0.0j) == (0.0 + 0.0j));
static_assert((0.0 + 0.0j) != (0.0 + 0.0j)); // expected-error {{static assertion}} \
                                             // expected-note {{evaluates to}}

static_assert((0.0 + 0.0j) == 0.0);
static_assert(0.0 == (0.0 + 0.0j));
static_assert(0.0 == 0.0j);
static_assert((0.0 + 1.0j) != 0.0);
static_assert(1.0 != (0.0 + 0.0j));
static_assert(0.0 != 1.0j);

// Walk around the complex plane stepping between angular differences and
// equality.
static_assert((1.0 + 0.0j) == (0.0 + 0.0j)); // expected-error {{static assertion}} \
                                             // expected-note {{evaluates to}}
static_assert((1.0 + 0.0j) == (1.0 + 0.0j));
static_assert((1.0 + 1.0j) == (1.0 + 0.0j)); // expected-error {{static assertion}} \
                                             // expected-note {{evaluates to}}
static_assert((1.0 + 1.0j) == (1.0 + 1.0j));
static_assert((0.0 + 1.0j) == (1.0 + 1.0j)); // expected-error {{static assertion}} \
                                             // expected-note {{evaluates to}}
static_assert((0.0 + 1.0j) == (0.0 + 1.0j));
static_assert((-1.0 + 1.0j) == (0.0 + 1.0j)); // expected-error {{static assertion}} \
                                              // expected-note {{evaluates to}}
static_assert((-1.0 + 1.0j) == (-1.0 + 1.0j));
static_assert((-1.0 + 0.0j) == (-1.0 + 1.0j)); // expected-error {{static assertion}} \
                                               // expected-note {{evaluates to}}
static_assert((-1.0 + 0.0j) == (-1.0 + 0.0j));
static_assert((-1.0 - 1.0j) == (-1.0 + 0.0j)); // expected-error {{static assertion}} \
                                               // expected-note {{evaluates to}}
static_assert((-1.0 - 1.0j) == (-1.0 - 1.0j));
static_assert((0.0 - 1.0j) == (-1.0 - 1.0j)); // expected-error {{static assertion}} \
                                              // expected-note {{evaluates to}}
static_assert((0.0 - 1.0j) == (0.0 - 1.0j));
static_assert((1.0 - 1.0j) == (0.0 - 1.0j)); // expected-error {{static assertion}} \
                                             // expected-note {{evaluates to}}
static_assert((1.0 - 1.0j) == (1.0 - 1.0j));

// Test basic mathematical folding of both complex and real operands.
static_assert(((1.0 + 0.5j) + (0.25 - 0.75j)) == (1.25 - 0.25j));
static_assert(((1.0 + 0.5j) + 0.25) == (1.25 + 0.5j));
static_assert((1.0 + (0.25 - 0.75j)) == (1.25 - 0.75j));

static_assert(((1.0 + 0.5j) - (0.25 - 0.75j)) == (0.75 + 1.25j));
static_assert(((1.0 + 0.5j) - 0.25) == (0.75 + 0.5j));
static_assert((1.0 - (0.25 - 0.75j)) == (0.75 + 0.75j));

static_assert(((1.25 + 0.5j) * (0.25 - 0.75j)) == (0.6875 - 0.8125j));
static_assert(((1.25 + 0.5j) * 0.25) == (0.3125 + 0.125j));
static_assert((1.25 * (0.25 - 0.75j)) == (0.3125 - 0.9375j));

static_assert(((1.25 + 0.5j) / (0.25 - 0.75j)) == (-0.1 + 1.7j));
static_assert(((1.25 + 0.5j) / 0.25) == (5.0 + 2.0j));
static_assert((1.25 / (0.25 - 0.75j)) == (0.5 + 1.5j));

// Test that infinities are preserved, don't turn into NaNs, and do form zeros
// when the divisor.
static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) * 1.0)) == 1);
static_assert(__builtin_isinf_sign(__imag__((1.0 + __builtin_inf() * 1.0j) * 1.0)) == 1);
static_assert(__builtin_isinf_sign(__real__(1.0 * (__builtin_inf() + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__imag__(1.0 * (1.0 + __builtin_inf() * 1.0j))) == 1);

static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) * (1.0 + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__real__((1.0 + 1.0j) * (__builtin_inf() + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) * (__builtin_inf() + 1.0j))) == 1);

static_assert(__builtin_isinf_sign(__real__((1.0 + __builtin_inf() * 1.0j) * (1.0 + 1.0j))) == -1);
static_assert(__builtin_isinf_sign(__imag__((1.0 + __builtin_inf() * 1.0j) * (1.0 + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__real__((1.0 + 1.0j) * (1.0 + __builtin_inf() * 1.0j))) == -1);
static_assert(__builtin_isinf_sign(__imag__((1.0 + 1.0j) * (1.0 + __builtin_inf() * 1.0j))) == 1);

static_assert(__builtin_isinf_sign(__real__((1.0 + __builtin_inf() * 1.0j) * (1.0 + __builtin_inf() * 1.0j))) == -1);
static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + __builtin_inf() * 1.0j) * (__builtin_inf() + __builtin_inf() * 1.0j))) == -1);

static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) / (1.0 + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__imag__(1.0 + (__builtin_inf() * 1.0j) / (1.0 + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__imag__((__builtin_inf() + __builtin_inf() * 1.0j) / (1.0 + 1.0j))) == 1);
static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) / 1.0)) == 1);
static_assert(__builtin_isinf_sign(__imag__(1.0 + (__builtin_inf() * 1.0j) / 1.0)) == 1);
static_assert(__builtin_isinf_sign(__imag__((__builtin_inf() + __builtin_inf() * 1.0j) / 1.0)) == 1);

static_assert(((1.0 + 1.0j) / (__builtin_inf() + 1.0j)) == (0.0 + 0.0j));
static_assert(((1.0 + 1.0j) / (1.0 + __builtin_inf() * 1.0j)) == (0.0 + 0.0j));
static_assert(((1.0 + 1.0j) / (__builtin_inf() + __builtin_inf() * 1.0j)) == (0.0 + 0.0j));
static_assert(((1.0 + 1.0j) / __builtin_inf()) == (0.0 + 0.0j));

static_assert(__builtin_isinf_sign(__real__((1.0 + 1.0j) / (0.0 + 0.0j))) == 1);
static_assert(__builtin_isinf_sign(__real__((1.0 + 1.0j) / 0.0)) == 1);

static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) / (0.0 + 0.0j))) == 1);
static_assert(__builtin_isinf_sign(__imag__((1.0 + __builtin_inf() * 1.0j) / (0.0 + 0.0j))) == 1);
static_assert(__builtin_isinf_sign(__imag__((__builtin_inf() + __builtin_inf() * 1.0j) / (0.0 + 0.0j))) == 1);
static_assert(__builtin_isinf_sign(__real__((__builtin_inf() + 1.0j) / 0.0)) == 1);
static_assert(__builtin_isinf_sign(__imag__((1.0 + __builtin_inf() * 1.0j) / 0.0)) == 1);
static_assert(__builtin_isinf_sign(__imag__((__builtin_inf() + __builtin_inf() * 1.0j) / 0.0)) == 1);
