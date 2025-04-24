

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,legacy %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fbounds-safety-bringup-missing-checks=indirect_count_update -verify=expected,legacy,extra %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fbounds-safety-bringup-missing-checks=compound_literal_init -verify=expected,legacy,cli %s
#include <ptrcheck.h>

void consume_sb(int* __sized_by(3) p);
void consume_sbon(int* __sized_by_or_null(3) p);

struct sb {
  int count;
  int* __sized_by(count) buf;
};

struct sbon {
  int count;
  int* __sized_by_or_null(count) buf;
};

void side_effect(void);

int global_arr [2] = {0};
// expected-note@+2 4{{__sized_by attribute is here}}
// extra-note@+1 2{{__sized_by attribute is here}}
int*__sized_by(2) global_sb = global_arr;
// expected-note@+2 4{{__sized_by_or_null attribute is here}}
// extra-note@+1 2{{__sized_by_or_null attribute is here}}
int*__sized_by_or_null(2) global_sbon = global_arr;

const int const_size = 2;
// expected-note@+2 4{{__sized_by attribute is here}}
// extra-note@+1 2{{__sized_by attribute is here}}
int*__sized_by(const_size) global_sb_const_qual_count = global_arr;
// expected-note@+2 4{{__sized_by_or_null attribute is here}}
// extra-note@+1 2{{__sized_by_or_null attribute is here}}
int*__sized_by_or_null(const_size) global_sbon_const_qual_count = global_arr;

// expected-note@+2 4{{__sized_by attribute is here}}
// extra-note@+1 2{{__sized_by attribute is here}}
int*__sized_by(1+1) global_sb_opo = global_arr;
// expected-note@+2 4{{__sized_by_or_null attribute is here}}
// extra-note@+1 2{{__sized_by_or_null attribute is here}}
int*__sized_by_or_null(1+1) global_sbon_opo = global_arr;

// legacy-note@+3 1{{__sized_by attribute is here}}
// expected-note@+2 8{{__sized_by attribute is here}}
// extra-note@+1 3{{__sized_by attribute is here}}
int* __sized_by(2) test_sb(int* __sized_by(3) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__sized_by attribute is here}}
  // extra-note@+1 2{{__sized_by attribute is here}}
  int* __sized_by(2) local_sb = p;
  local_sb = p; // OK
  side_effect();
  ++local_sb; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sb++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --local_sb; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local_sb--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local_sb += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sb -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_sb++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++local_sb = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *--local_sb = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *local_sb-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local = local_sb + 1; // OK because `local_sb` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_sb++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  ++global_sb; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sb--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  --global_sb; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  global_sb += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sb -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_sb++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++global_sb = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *global_sb-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *--global_sb = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  global_sb = global_sb + 1; // OK because `global_sb` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}

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
  p++, p = local; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}

  side_effect();

  consume_sb(++p); // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  struct sb S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  S = (struct sb){.buf = p++}; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
}

// legacy-note@+3 1{{__sized_by attribute is here}}
// expected-note@+2 8{{__sized_by attribute is here}}
// extra-note@+1 3{{__sized_by attribute is here}}
int* __sized_by(2) test_sb_constant_fold_count(int* __sized_by(2+1) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__sized_by attribute is here}}
  // extra-note@+1 2{{__sized_by attribute is here}}
  int* __sized_by(1+1) local_sb = p;
  local_sb = p; // OK
  side_effect();
  ++local_sb; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sb++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --local_sb; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local_sb--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local_sb += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sb -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_sb++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++local_sb = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *--local_sb = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *local_sb-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local = local_sb + 1; // OK because `local_sb` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_sb_opo++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  ++global_sb_opo; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sb_opo--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  --global_sb_opo; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  global_sb_opo += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sb_opo -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_sb_opo++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++global_sb_opo = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *global_sb_opo-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *--global_sb_opo = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  global_sb_opo = global_sb_opo + 1; // OK because `global_sb` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}

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
  p++, p = local; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}

  side_effect();

  consume_sb(++p); // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  struct sb S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  S = (struct sb){.buf = p++}; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size of 3 always traps}}
}

// legacy-note@+3 1{{__sized_by attribute is here}}
// expected-note@+2 8{{__sized_by attribute is here}}
// extra-note@+1 3{{__sized_by attribute is here}}
int* __sized_by(size) test_sb_const_qualified_size(const int size, int* __sized_by(size) p) {
  int* local;
  // Modify local var
  const int local_size = 2;
  // expected-note@+2 4{{__sized_by attribute is here}}
  // extra-note@+1 2{{__sized_by attribute is here}}
  int* __sized_by(local_size) local_sb = p;
  side_effect();
  local_sb = p; // OK

  side_effect();
  ++local_sb; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sb++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --local_sb; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local_sb--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  local_sb += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sb -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_sb++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++local_sb = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *--local_sb = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *local_sb-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  local = local_sb + 1; // OK because `local_sb` gets promoted to a __bidi_indexable first.
  side_effect();

  // Modify global
  global_sb_const_qual_count++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  ++global_sb_const_qual_count; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --global_sb_const_qual_count; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  global_sb_const_qual_count += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sb_const_qual_count-= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_sb_const_qual_count++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++global_sb_const_qual_count = 1; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *global_sb_const_qual_count-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *--global_sb_const_qual_count = 1; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  global_sb_const_qual_count = global_sb_const_qual_count + 1; // OK because `global_sb` gets promoted to a __bidi_indexable first.


  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__sized_by' attributed pointer with constant size 'size' always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__sized_by' attributed pointer with constant size 'size' always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__sized_by' pointer always traps}}


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
  p++, p = local; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}

  side_effect();

  consume_sb(++p); // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
  struct sb S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
  };
  // cli-error@+1{{initalizer for '__sized_by' pointer with side effects is not yet supported}}
  S = (struct sb){.buf = p++}; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__sized_by' attributed pointer with constant size 'size' always traps}}
}

// legacy-note@+3 1{{__sized_by_or_null attribute is here}}
// expected-note@+2 8{{__sized_by_or_null attribute is here}}
// extra-note@+1 3{{__sized_by_or_null attribute is here}}
int* __sized_by_or_null(2) test_sbon(int* __sized_by_or_null(3) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__sized_by_or_null attribute is here}}
  // extra-note@+1 2{{__sized_by_or_null attribute is here}}
  int* __sized_by_or_null(2) local_sbon = p;
  local_sbon = p; // OK
  side_effect();
  ++local_sbon; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sbon++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --local_sbon; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local_sbon--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local_sbon += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sbon -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_sbon++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++local_sbon = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *--local_sbon = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *local_sbon-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local = local_sbon + 1; // OK because `local_sbon` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_sbon++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  ++global_sbon; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sbon--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  --global_sbon; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  global_sbon += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sbon -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_sbon++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++global_sbon = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *global_sbon-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *--global_sbon = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  global_sbon = global_sbon + 1; // OK because `global_sbon` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}

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
  p++, p = local; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}

  side_effect();

  consume_sbon(++p); // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  struct sbon S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  S = (struct sbon){.buf = p++}; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
}

// legacy-note@+3 1{{__sized_by_or_null attribute is here}}
// expected-note@+2 8{{__sized_by_or_null attribute is here}}
// extra-note@+1 3{{__sized_by_or_null attribute is here}}
int* __sized_by_or_null(2) test_sbon_constant_fold_count(int* __sized_by_or_null(2+1) p) {
  int* local;

  // Modify local var
  // expected-note@+2 4{{__sized_by_or_null attribute is here}}
  // extra-note@+1 2{{__sized_by_or_null attribute is here}}
  int* __sized_by_or_null(1+1) local_sbon = p;
  local_sbon = p; // OK
  side_effect();
  ++local_sbon; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sbon++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --local_sbon; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local_sbon--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local_sbon += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sbon -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_sbon++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++local_sbon = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *--local_sbon = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *local_sbon-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local = local_sbon + 1; // OK because `local_sbon` gets promoted to a __bidi_indexable first.
  side_effect();


  // Modify global
  global_sbon_opo++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  ++global_sbon_opo; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sbon_opo--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  --global_sbon_opo; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  global_sbon_opo += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sbon_opo -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_sbon_opo++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++global_sbon_opo = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *global_sbon_opo-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *--global_sbon_opo = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  global_sbon_opo = global_sbon_opo + 1; // OK because `global_sbon` gets promoted to a __bidi_indexable first.

  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}

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
  p++, p = local; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}

  side_effect();

  consume_sbon(++p); // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  struct sbon S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
  };
  // cli-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  S = (struct sbon){.buf = p++}; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 3 always traps}}
}

// legacy-note@+3 1{{__sized_by_or_null attribute is here}}
// expected-note@+2 8{{__sized_by_or_null attribute is here}}
// extra-note@+1 3{{__sized_by_or_null attribute is here}}
int* __sized_by_or_null(size) test_sbon_const_qualified_size(const int size, int* __sized_by_or_null(size) p) {
  int* local;
  // Modify local var
  const int local_size = 2;
  // expected-note@+2 4{{__sized_by_or_null attribute is here}}
  // extra-note@+1 2{{__sized_by_or_null attribute is here}}
  int* __sized_by_or_null(local_size) local_sbon = p;
  side_effect();
  local_sbon = p; // OK

  side_effect();
  ++local_sbon; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sbon++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --local_sbon; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local_sbon--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  local_sbon += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  local_sbon -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *local_sbon++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++local_sbon = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *--local_sbon = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *local_sbon-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  local = local_sbon + 1; // OK because `local_sbon` gets promoted to a __bidi_indexable first.
  side_effect();

  // Modify global
  global_sbon_const_qual_count++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  ++global_sbon_const_qual_count; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  --global_sbon_const_qual_count; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  global_sbon_const_qual_count += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  global_sbon_const_qual_count-= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *global_sbon_const_qual_count++ = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();
  *++global_sbon_const_qual_count = 1; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size of 2 always traps}}
  side_effect();

  *global_sbon_const_qual_count-- = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *--global_sbon_const_qual_count = 1; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  global_sbon_const_qual_count = global_sbon_const_qual_count + 1; // OK because `global_sbon` gets promoted to a __bidi_indexable first.


  // Modify param
  ++p; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  side_effect();
  p++; // expected-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  side_effect();
  --p; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  p--; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  p += 1; // expected-error{{compound addition-assignment on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  side_effect();
  p -= 1; // expected-error{{compound subtraction-assignment on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  side_effect();

  // Only when -fbounds-safety-bringup-missing-checks=indirect_count_update
  *++p = 0; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  side_effect();
  *p++ = 0; // extra-error{{positive pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}

  *--p = 0; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}
  side_effect();
  *p-- = 0; // expected-error{{negative pointer arithmetic on '__sized_by_or_null' pointer always traps}}


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
  p++, p = local; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  side_effect();
  *(++p) = 0; // extra-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}

  side_effect();

  consume_sbon(++p); // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  struct sbon S = {
    .count = 2,
    // legacy-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    .buf = ++p // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
  };
  // cli-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
  S = (struct sbon){.buf = p++}; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}

  side_effect();

  return ++p; // expected-error{{pointer arithmetic on '__sized_by_or_null' attributed pointer with constant size 'size' always traps}}
}

// Warning diagnostic tests

void downgrade_to_warning(int* __sized_by(4) ptr) { // expected-note{{__sized_by attribute is here}}
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wbounds-safety-externally-counted-ptr-arith-constant-count"
  ++ptr; // expected-warning{{positive pointer arithmetic on '__sized_by' attributed pointer with constant size of 4 always traps}}
#pragma clang diagnostic pop
}

void downgrade_to_ignored(int* __sized_by(4) ptr) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbounds-safety-externally-counted-ptr-arith-constant-count"
  ++ptr; // ok
}
