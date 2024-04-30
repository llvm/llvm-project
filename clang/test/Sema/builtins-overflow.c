// RUN: %clang_cc1 -fsyntax-only -verify %s

#if __has_feature(__builtin_add_overflow)
#warning defined as expected
// expected-warning@-1 {{defined as expected}}
#endif

void test(void) {
  unsigned r;
  const char * c;
  float f;
  const unsigned q;

  __builtin_add_overflow();  // expected-error {{too few arguments to function call, expected 3, have 0}}
  __builtin_add_overflow(1, 1, 1, 1);  // expected-error {{too many arguments to function call, expected 3, have 4}}

  __builtin_add_overflow(c, 1, &r);  // expected-error {{operand argument to overflow builtin must be an integer type ('const char *' invalid)}}
  __builtin_add_overflow(1, c, &r);  // expected-error {{operand argument to overflow builtin must be an integer type ('const char *' invalid)}}
  __builtin_add_overflow(1, 1, 3);  // expected-error {{result argument to overflow builtin must be a pointer to a non-const integer type ('int' invalid)}}
  __builtin_add_overflow(1, 1, &f);  // expected-error {{result argument to overflow builtin must be a pointer to a non-const integer type ('float *' invalid)}}
  __builtin_add_overflow(1, 1, &q);  // expected-error {{result argument to overflow builtin must be a pointer to a non-const integer type ('const unsigned int *' invalid)}}

  {
    _BitInt(128) x = 1;
    _BitInt(128) y = 1;
    _BitInt(128) result;
    _Bool status = __builtin_mul_overflow(x, y, &result); // expect ok
  }
#if __BITINT_MAXWIDTH__ > 128
  {
    unsigned _BitInt(129) x = 1;
    unsigned _BitInt(129) y = 1;
    unsigned _BitInt(129) result;
    _Bool status = __builtin_mul_overflow(x, y, &result); // expect ok
  }
  {
    _BitInt(129) x = 1;
    _BitInt(129) y = 1;
    _BitInt(129) result;
    _Bool status = __builtin_mul_overflow(x, y, &result); // expected-error {{__builtin_mul_overflow does not support 'signed _BitInt' operands of more than 128 bits}}
  }
#endif
}
