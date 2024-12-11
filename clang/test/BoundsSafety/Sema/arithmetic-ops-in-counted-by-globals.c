

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <ptrcheck.h>

int len;
int *__counted_by(len - 2) buf;
// expected-error@-1{{negative count value of -2 for 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single')}}
int *__counted_by_or_null(len - 2) buf_n;

int len2 = 2;
int *__counted_by(len2 - 2) buf2;
int *__counted_by_or_null(len2 - 2) buf2_n;

int len3 = 0;
int *__counted_by(len3 - 2) buf3;
// expected-error@-1{{negative count value of -2 for 'buf3' of type 'int *__single __counted_by(len3 - 2)' (aka 'int *__single')}}
int *__counted_by_or_null(len3 - 2) buf3_n;

int len4 = 0;
int *__counted_by(len4 + 2) buf4;
// expected-error@-1{{implicitly initializing 'buf4' of type 'int *__single __counted_by(len4 + 2)' (aka 'int *__single') and count value of 2 with null always fails}}
int *__counted_by_or_null(len4 + 2) buf4_n;

int len5 = 0;
int *__counted_by(len5 + 2) buf5 = 0;
// expected-error@-1{{initializing 'buf5' of type 'int *__single __counted_by(len5 + 2)' (aka 'int *__single') and count value of 2 with null always fails}}
int *__counted_by_or_null(len5 + 2) buf5_n = 0;

int len5_1;
int *__counted_by(len5_1 + 2) buf5_1 = 0;
// expected-error@-1{{initializing 'buf5_1' of type 'int *__single __counted_by(len5_1 + 2)' (aka 'int *__single') and count value of 2 with null always fails}}
int *__counted_by_or_null(len5_1 + 2) buf5_1_n = 0;

int *len6;
int *__counted_by(*len6) buf6;
// expected-error@-1{{dereference operator in '__counted_by' is only allowed for function parameters}}
int *__counted_by_or_null(*len6) buf6_n;
// expected-error@-1{{dereference operator in '__counted_by_or_null' is only allowed for function parameters}}

int *len6_1;
int *__counted_by(*len6_1 + 2) buf6_1;
// expected-error@-1{{invalid argument expression to bounds attribute}}
int *__counted_by_or_null(*len6_1 + 2) buf6_1_n;
// expected-error@-1{{invalid argument expression to bounds attribute}}

int len7_1;
int len7_2;
int *__counted_by(len7_1 * 2 + len7_2 * 4) buf7;
int *__counted_by_or_null(len7_1 * 2 + len7_2 * 4) buf7_n;

int len8_1;
int len8_2;
int *__counted_by(len8_1 * 2 + len8_2 * 4 + 1) buf8;
// expected-error@-1{{implicitly initializing 'buf8' of type 'int *__single __counted_by(len8_1 * 2 + len8_2 * 4 + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
int *__counted_by_or_null(len8_1 * 2 + len8_2 * 4 + 1) buf8_n;

int len9_1 = 1;
int len9_2 = 3;
int *__counted_by(len9_1 * len9_2) buf9;
// expected-error@-1{{implicitly initializing 'buf9' of type 'int *__single __counted_by(len9_1 * len9_2)' (aka 'int *__single') and count value of 3 with null always fails}}
int *__counted_by_or_null(len9_1 * len9_2) buf9_n;

int len10_1;
int len10_2 = 2;
int *__counted_by(len10_1 * len10_2) buf10;
int *__counted_by_or_null(len10_1 * len10_2) buf10_n;

int len11_1;
int len11_2 = 2;
int *__counted_by(len11_1 * len11_2 + 1) buf11;
// expected-error@-1{{implicitly initializing 'buf11' of type 'int *__single __counted_by(len11_1 * len11_2 + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
int *__counted_by_or_null(len11_1 * len11_2 + 1) buf11_n;

// don't complain about extern variables
extern int ext_len;
extern int *__counted_by(ext_len - 2) ext_buf;
