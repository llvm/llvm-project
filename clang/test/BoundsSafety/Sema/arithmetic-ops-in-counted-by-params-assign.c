

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <stddef.h>
#include <ptrcheck.h>


/* --------------------
        counted_by
   -------------------- */
void param_with_count(int *__counted_by(len - 2) buf, int len) {
  // expected-error@+1{{negative count value of -2 for 'buf' of type 'int *__single __counted_by(len - 2)' (aka 'int *__single')}}
  buf = 0;
  len = 0;
}

void inout_count(int *__counted_by(*len) buf, int *len);

void inout_count_buf(int *__counted_by(*len) *buf, int *len);

void pass_argument_to_inout_count_buf(int *__counted_by(len - 1) buf, int len) {
  // expected-error@+1{{parameter 'buf' with '__counted_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_count_buf(&buf, &len);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len - 1' in argument to function}}
  inout_count(buf, &len);

  int arr[10];
  int len2 = 9;
  int *__counted_by(len2 + 1) buf2 = arr;
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_count_buf(&buf2, &len2);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_count(buf2, &len2);

  int len3 = 1;
  int *__counted_by(len3 - 1) buf3 = arr;
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'buf2' as an indirect parameter; must also pass 'len2' or its address because the type of 'buf2', 'int *__single __counted_by(len2 + 1)' (aka 'int *__single'), refers to 'len2'}}
  inout_count_buf(&buf2, &len3);
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'len3' as an indirect parameter; must also pass 'buf3' or its address because the type of 'buf3', 'int *__single __counted_by(len3 - 1)' (aka 'int *__single'), refers to 'len3'}}
  inout_count(buf2, &len3);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len3 - 1' in argument to function}}
  inout_count(buf3, &len3);
}

/* --------------------
         sized_by
   -------------------- */
void param_with_size(int *__sized_by(len - 2) buf, int len) {
  // expected-error@+1{{negative size value of -2 for 'buf' of type 'int *__single __sized_by(len - 2)' (aka 'int *__single')}}
  buf = 0;
  len = 0;
}

void inout_size(int *__sized_by(*len) buf, int *len);

void inout_size_buf(int *__sized_by(*len) *buf, int *len);

void pass_argument_to_inout_size_buf(int *__sized_by(len - 1) buf, int len) {
  // expected-error@+1{{parameter 'buf' with '__sized_by' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_size_buf(&buf, &len);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len - 1' in argument to function}}
  inout_size(buf, &len);

  int arr[10];
  int len2 = 9;
  int *__sized_by(len2 + 1) buf2 = arr;
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_size_buf(&buf2, &len2);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_size(buf2, &len2);

  int len3 = 1;
  int *__sized_by(len3 - 1) buf3 = arr;
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'buf2' as an indirect parameter; must also pass 'len2' or its address because the type of 'buf2', 'int *__single __sized_by(len2 + 1)' (aka 'int *__single'), refers to 'len2'}}
  inout_size_buf(&buf2, &len3);
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'len3' as an indirect parameter; must also pass 'buf3' or its address because the type of 'buf3', 'int *__single __sized_by(len3 - 1)' (aka 'int *__single'), refers to 'len3'}}
  inout_size(buf2, &len3);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len3 - 1' in argument to function}}
  inout_size(buf3, &len3);
}

/* --------------------
    counted_by_or_null
   -------------------- */
void inout_count_nullable(int *__counted_by_or_null(*len) buf, int *len);

void inout_count_nullable_buf(int *__counted_by_or_null(*len) *buf, int *len);

void pass_argument_to_inout_count_nullable_buf(int *__counted_by_or_null(len - 1) buf, int len) {
  // expected-error@+1{{parameter 'buf' with '__counted_by_or_null' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_count_buf(&buf, &len);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len - 1' in argument to function}}
  inout_count_nullable(buf, &len);

  int arr[10];
  int len2 = 9;
  int *__counted_by_or_null(len2 + 1) buf2 = arr;
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_count_buf(&buf2, &len2);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_count_nullable(buf2, &len2);

  int len3 = 1;
  int *__counted_by_or_null(len3 - 1) buf3 = arr;
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'buf2' as an indirect parameter; must also pass 'len2' or its address because the type of 'buf2', 'int *__single __counted_by_or_null(len2 + 1)' (aka 'int *__single'), refers to 'len2'}}
  inout_count_buf(&buf2, &len3);
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'len3' as an indirect parameter; must also pass 'buf3' or its address because the type of 'buf3', 'int *__single __counted_by_or_null(len3 - 1)' (aka 'int *__single'), refers to 'len3'}}
  inout_count_nullable(buf2, &len3);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len3 - 1' in argument to function}}
  inout_count_nullable(buf3, &len3);
}

/* --------------------
     sized_by_or_null
   -------------------- */
void inout_size_nullable(int *__sized_by_or_null(*len) buf, int *len);

void inout_size_nullable_buf(int *__sized_by_or_null(*len) *buf, int *len);

void pass_argument_to_inout_size_nullable_buf(int *__sized_by_or_null(len - 1) buf, int len) {
  // expected-error@+1{{parameter 'buf' with '__sized_by_or_null' attribute depending on an indirect count is implicitly read-only and cannot be passed as an indirect argument}}
  inout_size_nullable_buf(&buf, &len);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len - 1' in argument to function}}
  inout_size_nullable(buf, &len);

  int arr[10];
  int len2 = 9;
  int *__sized_by_or_null(len2 + 1) buf2 = arr;
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_size_nullable_buf(&buf2, &len2);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  inout_size_nullable(buf2, &len2);

  int len3 = 1;
  int *__sized_by_or_null(len3 - 1) buf3 = arr;
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'buf2' as an indirect parameter; must also pass 'len2' or its address because the type of 'buf2', 'int *__single __sized_by_or_null(len2 + 1)' (aka 'int *__single'), refers to 'len2'}}
  inout_size_nullable_buf(&buf2, &len3);
  // expected-error@+2{{incompatible count expression '*len' vs. 'len2 + 1' in argument to function}}
  // expected-error@+1{{passing address of 'len3' as an indirect parameter; must also pass 'buf3' or its address because the type of 'buf3', 'int *__single __sized_by_or_null(len3 - 1)' (aka 'int *__single'), refers to 'len3'}}
  inout_size_nullable(buf2, &len3);
  // expected-error@+1{{incompatible count expression '*len' vs. 'len3 - 1' in argument to function}}
  inout_size_nullable(buf3, &len3);
}
