// RUN: %clang_cc1 -verify %s

/* WG14 N736: yes
 * preprocessor arithmetic done in intmax_t/uintmax_t
 */

// There is not a standard requirement that this relationships holds. If these
// asserts fail, it means we have another test scenario to consider.
_Static_assert(__INTMAX_MAX__ == __LONG_LONG_MAX__,
               "intmax_t is not the same width as long long?");
_Static_assert((-__INTMAX_MAX__ - 1) == (-__LONG_LONG_MAX__ - 1LL),
               "intmax_t is not the same width as long long?");
_Static_assert(__UINTMAX_MAX__ == (__LONG_LONG_MAX__ * 2ULL + 1ULL),
               "uintmax_t is not the same width as unsigned long long?");

// Test that arithmetic on the largest positive signed intmax_t works.
#if 9223372036854775807LL + 0LL != 9223372036854775807LL
#error "uh oh"
#endif

// Same for negative.
#if -9223372036854775807LL - 1LL + 0LL != -9223372036854775807LL - 1LL
#error "uh oh"
#endif

// Then test the same for unsigned
#if 18446744073709551615ULL + 0ULL != 18446744073709551615ULL
#error "uh oh"
#endif

// Test that unsigned overflow causes silent wraparound.
#if 18446744073709551615ULL + 1ULL != 0 // Silently wraps to 0.
#error "uh oh"
#endif

#if 0ULL - 1ULL != 18446744073709551615ULL // Silently wraps to 0xFFFF'FFFF'FFFF'FFFF.
#error "uh oh"
#endif

// Now test that signed arithmetic that pushes us over a limit is properly
// diagnosed.
#if 9223372036854775807LL + 1LL // expected-warning {{integer overflow in preprocessor expression}}
#endif

#if -9223372036854775807LL - 2LL // expected-warning {{integer overflow in preprocessor expression}}
#endif

