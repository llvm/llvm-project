

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,legacy %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fbounds-safety-bringup-missing-checks=indirect_count_update -verify=expected,legacy,extra %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fbounds-safety-bringup-missing-checks=compound_literal_init -verify=expected,legacy,cli %s
#include <ptrcheck.h>

void consume_cb(int* __counted_by(3) p);
void consume_cbon(int* __counted_by_or_null(3) p);

struct cb {
  int count;
  int* __counted_by(count) buf;
};

struct cbon {
  int count;
  int* __counted_by_or_null(count) buf;
};

void side_effect(void);

int global_arr [2] = {0};
// expected-note@+2 4{{__counted_by attribute is here}}
// extra-note@+1 2{{__counted_by attribute is here}}
int*__counted_by(2) global_cb = global_arr;
// expected-note@+2 4{{__counted_by_or_null attribute is here}}
// extra-note@+1 2{{__counted_by_or_null attribute is here}}
int*__counted_by_or_null(2) global_cbon = global_arr;

const int const_size = 2;
// expected-note@+2 4{{__counted_by attribute is here}}
// extra-note@+1 2{{__counted_by attribute is here}}
int*__counted_by(const_size) global_cb_const_qual_count = global_arr;
// expected-note@+2 4{{__counted_by_or_null attribute is here}}
// extra-note@+1 2{{__counted_by_or_null attribute is here}}
int*__counted_by_or_null(const_size) global_cbon_const_qual_count = global_arr;

// expected-note@+2 4{{__counted_by attribute is here}}
// extra-note@+1 2{{__counted_by attribute is here}}
int*__counted_by(1+1) global_cb_opo = global_arr;
// expected-note@+2 4{{__counted_by_or_null attribute is here}}
// extra-note@+1 2{{__counted_by_or_null attribute is here}}
int*__counted_by_or_null(1+1) global_cbon_opo = global_arr;

// legacy-note@+3 1{{__counted_by attribute is here}}
// expected-note@+2 8{{__counted_by attribute is here}}
// extra-note@+1 3{{__counted_by attribute is here}}
int* __counted_by(2) test_cb(int* __counted_by(3) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__counted_by attribute is here}}
  // extra-note@+1 2{{__counted_by attribute is here}}
  int* __counted_by(2) local_cb = p;
  local_cb = p; // OK
  side_effect();
  ++local_cb; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cb++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --local_cb; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local_cb--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local_cb += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cb -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_cb++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++local_cb = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *--local_cb = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *local_cb-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local = local_cb + 1; // OK because `local_cb` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_cb++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  ++global_cb; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cb--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  --global_cb; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  global_cb += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cb -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_cb++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++global_cb = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *global_cb-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *--global_cb = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  global_cb = global_cb + 1; // OK because `global_cb` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}

  p *= 1; // expected-error{{invalid operands to binary expression}}
  p *= 2; // expected-error{{invalid operands to binary expression}}
  p |= 2; // expected-error{{invalid operands to binary expression}}
  p &= 2; // expected-error{{invalid operands to binary expression}}
  p ^= 2; // expected-error{{invalid operands to binary expression}}
  p %= 2; // expected-error{{invalid operands to binary expression}}
  p <<= 0; // expected-error{{invalid operands to binary expression}}
  p >>= 0; // expected-error{{invalid operands to binary expression}}

  // Increment in other contexts
  // legacy-error@+2{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
  // legacy-note@+1{{previously assigned here}}
  p++, p = local; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}

  side_effect();

  consume_cb(++p); // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  struct cb S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  S = (struct cb){.buf = p++}; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
}

// legacy-note@+3 1{{__counted_by attribute is here}}
// expected-note@+2 8{{__counted_by attribute is here}}
// extra-note@+1 3{{__counted_by attribute is here}}
int* __counted_by(2) test_cb_constant_fold_count(int* __counted_by(2+1) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__counted_by attribute is here}}
  // extra-note@+1 2{{__counted_by attribute is here}}
  int* __counted_by(1+1) local_cb = p;
  local_cb = p; // OK
  side_effect();
  ++local_cb; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cb++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --local_cb; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local_cb--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local_cb += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cb -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_cb++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++local_cb = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *--local_cb = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *local_cb-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local = local_cb + 1; // OK because `local_cb` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_cb_opo++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  ++global_cb_opo; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cb_opo--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  --global_cb_opo; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  global_cb_opo += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cb_opo -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_cb_opo++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++global_cb_opo = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *global_cb_opo-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *--global_cb_opo = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  global_cb_opo = global_cb_opo + 1; // OK because `global_cb` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}

  p *= 1; // expected-error{{invalid operands to binary expression}}
  p *= 2; // expected-error{{invalid operands to binary expression}}
  p |= 2; // expected-error{{invalid operands to binary expression}}
  p &= 2; // expected-error{{invalid operands to binary expression}}
  p ^= 2; // expected-error{{invalid operands to binary expression}}
  p %= 2; // expected-error{{invalid operands to binary expression}}
  p <<= 0; // expected-error{{invalid operands to binary expression}}
  p >>= 0; // expected-error{{invalid operands to binary expression}}

  // Increment in other contexts
  // legacy-error@+2{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
  // legacy-note@+1{{previously assigned here}}
  p++, p = local; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}

  side_effect();

  consume_cb(++p); // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  struct cb S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  S = (struct cb){.buf = p++}; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
}

// legacy-note@+3 1{{__counted_by attribute is here}}
// expected-note@+2 8{{__counted_by attribute is here}}
// extra-note@+1 3{{__counted_by attribute is here}}
int* __counted_by(size) test_cb_const_qualified_size(const int size, int* __counted_by(size) p) {
  int* local;
  // Modify local var
  const int local_size = 2;
  // expected-note@+2 4{{__counted_by attribute is here}}
  // extra-note@+1 2{{__counted_by attribute is here}}
  int* __counted_by(local_size) local_cb = p;
  side_effect();
  local_cb = p; // OK

  side_effect();
  ++local_cb; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cb++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --local_cb; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local_cb--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  local_cb += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cb -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_cb++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++local_cb = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *--local_cb = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *local_cb-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  local = local_cb + 1; // OK because `local_cb` gets promoted to a __bidi_indexable first.
  side_effect();

  // Modify global
  global_cb_const_qual_count++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  ++global_cb_const_qual_count; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --global_cb_const_qual_count; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  global_cb_const_qual_count += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cb_const_qual_count-= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_cb_const_qual_count++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++global_cb_const_qual_count = 1; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *global_cb_const_qual_count-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *--global_cb_const_qual_count = 1; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  global_cb_const_qual_count = global_cb_const_qual_count + 1; // OK because `global_cb` gets promoted to a __bidi_indexable first.


  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__counted_by' attributed pointer with constant count 'size' always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__counted_by' attributed pointer with constant count 'size' always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__counted_by' pointer always traps}}


  p *= 1; // expected-error{{invalid operands to binary expression}}
  p *= 2; // expected-error{{invalid operands to binary expression}}
  p |= 2; // expected-error{{invalid operands to binary expression}}
  p &= 2; // expected-error{{invalid operands to binary expression}}
  p ^= 2; // expected-error{{invalid operands to binary expression}}
  p %= 2; // expected-error{{invalid operands to binary expression}}
  p <<= 0; // expected-error{{invalid operands to binary expression}}
  p >>= 0; // expected-error{{invalid operands to binary expression}}

  // Increment in other contexts
  // legacy-error@+2{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
  // legacy-note@+1{{previously assigned here}}
  p++, p = local; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}

  side_effect();

  consume_cb(++p); // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
  struct cb S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
  };
  // cli-error@+1{{initalizer for '__counted_by' pointer with side effects is not yet supported}}
  S = (struct cb){.buf = p++}; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__counted_by' attributed pointer with constant count 'size' always traps}}
}

// legacy-note@+3 1{{__counted_by_or_null attribute is here}}
// expected-note@+2 8{{__counted_by_or_null attribute is here}}
// extra-note@+1 3{{__counted_by_or_null attribute is here}}
int* __counted_by_or_null(2) test_cbon(int* __counted_by_or_null(3) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__counted_by_or_null attribute is here}}
  // extra-note@+1 2{{__counted_by_or_null attribute is here}}
  int* __counted_by_or_null(2) local_cbon = p;
  local_cbon = p; // OK
  side_effect();
  ++local_cbon; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cbon++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --local_cbon; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local_cbon--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local_cbon += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cbon -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_cbon++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++local_cbon = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *--local_cbon = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *local_cbon-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local = local_cbon + 1; // OK because `local_cbon` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_cbon++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  ++global_cbon; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cbon--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  --global_cbon; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  global_cbon += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cbon -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_cbon++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++global_cbon = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *global_cbon-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *--global_cbon = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  global_cbon = global_cbon + 1; // OK because `global_cbon` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}

  p *= 1; // expected-error{{invalid operands to binary expression}}
  p *= 2; // expected-error{{invalid operands to binary expression}}
  p |= 2; // expected-error{{invalid operands to binary expression}}
  p &= 2; // expected-error{{invalid operands to binary expression}}
  p ^= 2; // expected-error{{invalid operands to binary expression}}
  p %= 2; // expected-error{{invalid operands to binary expression}}
  p <<= 0; // expected-error{{invalid operands to binary expression}}
  p >>= 0; // expected-error{{invalid operands to binary expression}}

  // Increment in other contexts
  // legacy-error@+2{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
  // legacy-note@+1{{previously assigned here}}
  p++, p = local; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}

  side_effect();

  consume_cbon(++p); // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  struct cbon S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  S = (struct cbon){.buf = p++}; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
}

// legacy-note@+3 1{{__counted_by_or_null attribute is here}}
// expected-note@+2 8{{__counted_by_or_null attribute is here}}
// extra-note@+1 3{{__counted_by_or_null attribute is here}}
int* __counted_by_or_null(2) test_cbon_constant_fold_count(int* __counted_by_or_null(2+1) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__counted_by_or_null attribute is here}}
  // extra-note@+1 2{{__counted_by_or_null attribute is here}}
  int* __counted_by_or_null(1+1) local_cbon = p;
  local_cbon = p; // OK
  side_effect();
  ++local_cbon; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cbon++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --local_cbon; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local_cbon--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local_cbon += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cbon -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_cbon++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++local_cbon = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *--local_cbon = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *local_cbon-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local = local_cbon + 1; // OK because `local_cbon` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_cbon_opo++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  ++global_cbon_opo; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cbon_opo--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  --global_cbon_opo; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  global_cbon_opo += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cbon_opo -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_cbon_opo++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++global_cbon_opo = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *global_cbon_opo-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *--global_cbon_opo = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  global_cbon_opo = global_cbon_opo + 1; // OK because `global_cbon` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}

  p *= 1; // expected-error{{invalid operands to binary expression}}
  p *= 2; // expected-error{{invalid operands to binary expression}}
  p |= 2; // expected-error{{invalid operands to binary expression}}
  p &= 2; // expected-error{{invalid operands to binary expression}}
  p ^= 2; // expected-error{{invalid operands to binary expression}}
  p %= 2; // expected-error{{invalid operands to binary expression}}
  p <<= 0; // expected-error{{invalid operands to binary expression}}
  p >>= 0; // expected-error{{invalid operands to binary expression}}

  // Increment in other contexts
  // legacy-error@+2{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
  // legacy-note@+1{{previously assigned here}}
  p++, p = local; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}

  side_effect();

  consume_cbon(++p); // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  struct cbon S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  S = (struct cbon){.buf = p++}; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 3 always traps}}
}

// legacy-note@+3 1{{__counted_by_or_null attribute is here}}
// expected-note@+2 8{{__counted_by_or_null attribute is here}}
// extra-note@+1 3{{__counted_by_or_null attribute is here}}
int* __counted_by_or_null(size) test_cbon_const_qualified_size(const int size, int* __counted_by_or_null(size) p) {
  int* local;
  // Modify local var
  const int local_size = 2;
  // expected-note@+2 4{{__counted_by_or_null attribute is here}}
  // extra-note@+1 2{{__counted_by_or_null attribute is here}}
  int* __counted_by_or_null(local_size) local_cbon = p;
  side_effect();
  local_cbon = p; // OK

  side_effect();
  ++local_cbon; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cbon++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --local_cbon; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local_cbon--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  local_cbon += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  local_cbon -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_cbon++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++local_cbon = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *--local_cbon = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *local_cbon-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  local = local_cbon + 1; // OK because `local_cbon` gets promoted to a __bidi_indexable first.
  side_effect();

  // Modify global
  global_cbon_const_qual_count++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  ++global_cbon_const_qual_count; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  --global_cbon_const_qual_count; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  global_cbon_const_qual_count += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  global_cbon_const_qual_count-= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_cbon_const_qual_count++ = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();
  *++global_cbon_const_qual_count = 1; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count of 2 always traps}}
  side_effect();

  *global_cbon_const_qual_count-- = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *--global_cbon_const_qual_count = 1; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  global_cbon_const_qual_count = global_cbon_const_qual_count + 1; // OK because `global_cbon` gets promoted to a __bidi_indexable first.


  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__counted_by_or_null' pointer always traps}}


  p *= 1; // expected-error{{invalid operands to binary expression}}
  p *= 2; // expected-error{{invalid operands to binary expression}}
  p |= 2; // expected-error{{invalid operands to binary expression}}
  p &= 2; // expected-error{{invalid operands to binary expression}}
  p ^= 2; // expected-error{{invalid operands to binary expression}}
  p %= 2; // expected-error{{invalid operands to binary expression}}
  p <<= 0; // expected-error{{invalid operands to binary expression}}
  p >>= 0; // expected-error{{invalid operands to binary expression}}

  // Increment in other contexts
  // legacy-error@+2{{multiple consecutive assignments to a dynamic count pointer 'p' must be simplified; keep only one of the assignments}}
  // legacy-note@+1{{previously assigned here}}
  p++, p = local; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}

  side_effect();

  consume_cbon(++p); // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  struct cbon S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
  };
  // cli-error@+1{{initalizer for '__counted_by_or_null' pointer with side effects is not yet supported}}
  S = (struct cbon){.buf = p++}; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__counted_by_or_null' attributed pointer with constant count 'size' always traps}}
}

// Warning diagnostic tests

void downgrade_to_warning(int* __counted_by(4) ptr) { // expected-note{{__counted_by attribute is here}}
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wbounds-safety-externally-counted-ptr-arith-constant-count"
  ++ptr; // expected-warning{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 4 always traps}}
#pragma clang diagnostic pop
}

void downgrade_to_ignored(int* __counted_by(4) ptr) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbounds-safety-externally-counted-ptr-arith-constant-count"
  ++ptr; // ok
}
