// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

typedef __INT32_TYPE__ int32_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __INT_FAST32_TYPE__ int_fast32_t;
typedef __UINT_FAST32_TYPE__ uint_fast32_t;

int printf(const char *restrict, ...);
int scanf(const char *restrict, ...);

void t1(int32_t i32, uint32_t u32, int_fast32_t if32, uint_fast32_t uf32, int32_t *i_ptr, int_fast32_t *if_ptr, double *d_ptr) {
  printf("%w32d", i32);
  printf("%w32i", i32);
  printf("%w32u", u32);
  printf("%w32x", u32);
  printf("%w32b", u32);
  printf("%wf32d", if32);
  printf("%wf32u", uf32);
  printf("%wf32B", uf32);

  printf("%w32d", 1.0);  // expected-warning{{format specifies type 'int32_t' (aka 'int') but the argument has type 'double'}}
  printf("%w32u", 1.0);  // expected-warning{{format specifies type 'uint32_t' (aka 'unsigned int') but the argument has type 'double'}}
  printf("%wf32d", 1.0); // expected-warning{{format specifies type 'int_fast32_t' (aka 'int') but the argument has type 'double'}}
  printf("%wf32u", 1.0); // expected-warning{{format specifies type 'uint_fast32_t' (aka 'unsigned int') but the argument has type 'double'}}
  printf("%w18446744073709551616d", i32); // expected-warning{{invalid conversion specifier '1'}}

  printf("%w32n", i_ptr);
  printf("%wf32n", if_ptr);
  printf("%w32n", d_ptr); // expected-warning{{format specifies type 'int32_t *' (aka 'int *') but the argument has type 'double *'}}
}

void t2(int32_t *i_ptr, uint32_t *u_ptr, int_fast32_t *if_ptr, uint_fast32_t *uf_ptr, double *d_ptr) {
  scanf("%w32d", i_ptr);
  scanf("%w32i", i_ptr);
  scanf("%w32u", u_ptr);
  scanf("%w32x", u_ptr);
  scanf("%w32b", u_ptr);
  scanf("%wf32d", if_ptr);
  scanf("%wf32u", uf_ptr);

  scanf("%w32d", d_ptr);  // expected-warning{{format specifies type 'int32_t *' (aka 'int *') but the argument has type 'double *'}}
  scanf("%w32u", d_ptr);  // expected-warning{{format specifies type 'uint32_t *' (aka 'unsigned int *') but the argument has type 'double *'}}
  scanf("%wf32d", d_ptr); // expected-warning{{format specifies type 'int_fast32_t *' (aka 'int *') but the argument has type 'double *'}}
  scanf("%wf32u", d_ptr); // expected-warning{{format specifies type 'uint_fast32_t *' (aka 'unsigned int *') but the argument has type 'double *'}}
}

void t3(const char *fmt) __attribute__((format_matches(printf, 1, "%w32d"))); // expected-note{{comparing with this specifier}}
void t4(void) {
  t3("%w32d");
  t3("%w64d"); // expected-warning{{format specifier 'w64d' is incompatible with 'w32d'}}
}
