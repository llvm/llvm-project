

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <ptrcheck.h>

int len;
void *__sized_by(len - 2) buf;
// expected-error@-1{{negative size value of -2 for 'buf' of type 'void *__single __sized_by(len - 2)' (aka 'void *__single')}}
void *__sized_by_or_null(len - 2) buf_n;

int len2 = 2;
void *__sized_by(len2 - 2) buf2;
void *__sized_by_or_null(len2 - 2) buf2_n;

int len3 = 0;
void *__sized_by(len3 - 2) buf3;
// expected-error@-1{{negative size value of -2 for 'buf3' of type 'void *__single __sized_by(len3 - 2)' (aka 'void *__single')}}
void *__sized_by_or_null(len3 - 2) buf3_n;

int len4 = 0;
void *__sized_by(len4 + 2) buf4;
// expected-error@-1{{implicitly initializing 'buf4' of type 'void *__single __sized_by(len4 + 2)' (aka 'void *__single') and size value of 2 with null always fails}}
void *__sized_by_or_null(len4 + 2) buf4_n;

int len5 = 0;
void *__sized_by(len5 + 2) buf5 = 0;
// expected-error@-1{{initializing 'buf5' of type 'void *__single __sized_by(len5 + 2)' (aka 'void *__single') and size value of 2 with null always fails}}
void *__sized_by_or_null(len5 + 2) buf5_n = 0;

int len5_1;
void *__sized_by(len5_1 + 2) buf5_1 = 0;
// expected-error@-1{{initializing 'buf5_1' of type 'void *__single __sized_by(len5_1 + 2)' (aka 'void *__single') and size value of 2 with null always fails}}
void *__sized_by_or_null(len5_1 + 2) buf5_1_n = 0;

int *len6;
void *__sized_by(*len6) buf6;
// expected-error@-1{{dereference operator in '__sized_by' is only allowed for function parameters}}
void *__sized_by_or_null(*len6) buf6_n;
// expected-error@-1{{dereference operator in '__sized_by_or_null' is only allowed for function parameters}}

int *len6_1;
void *__sized_by(*len6_1 + 2) buf6_1;
// expected-error@-1{{invalid argument expression to bounds attribute}}
void *__sized_by_or_null(*len6_1 + 2) buf6_1_n;
// expected-error@-1{{invalid argument expression to bounds attribute}}

int len7_1;
int len7_2;
void *__sized_by(len7_1 * 2 + len7_2 * 4) buf7;
void *__sized_by_or_null(len7_1 * 2 + len7_2 * 4) buf7_n;

int len8_1;
int len8_2;
void *__sized_by(len8_1 * 2 + len8_2 * 4 + 1) buf8;
// expected-error@-1{{implicitly initializing 'buf8' of type 'void *__single __sized_by(len8_1 * 2 + len8_2 * 4 + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
void *__sized_by_or_null(len8_1 * 2 + len8_2 * 4 + 1) buf8_n;

int len9_1 = 1;
int len9_2 = 3;
void *__sized_by(len9_1 * len9_2) buf9;
// expected-error@-1{{implicitly initializing 'buf9' of type 'void *__single __sized_by(len9_1 * len9_2)' (aka 'void *__single') and size value of 3 with null always fails}}
void *__sized_by_or_null(len9_1 * len9_2) buf9_n;

int len10_1;
int len10_2 = 2;
void *__sized_by(len10_1 * len10_2) buf10;
void *__sized_by_or_null(len10_1 * len10_2) buf10_n;

int len11_1;
int len11_2 = 2;
void *__sized_by(len11_1 * len11_2 + 1) buf11;
// expected-error@-1{{implicitly initializing 'buf11' of type 'void *__single __sized_by(len11_1 * len11_2 + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
void *__sized_by_or_null(len11_1 * len11_2 + 1) buf11_n;

// don't complain about extern variables
extern int ext_len;
extern int *__sized_by(ext_len - 2) ext_buf;
