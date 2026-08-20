// RUN: %clang_cc1 -fsyntax-only -Wcounted-by-static-allocation -Wno-gnu-variable-sized-type-not-at-end -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-gnu-variable-sized-type-not-at-end -verify=quiet %s
// quiet-no-diagnostics

#define __counted_by(f)  __attribute__((counted_by(f)))

struct flex {
  int count;
  char fam[] __counted_by(count); // expected-note 7 {{'__counted_by' is intended for dynamically allocated objects, where the size of the allocation can be computed from the count}}
};

// Global definition with an initializer.
struct flex global_init = {.count = 10}; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with static storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}

// Tentative definitions are diagnosed once per object.
struct flex tentative; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with static storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}
struct flex tentative;

// Thread-local storage.
_Thread_local struct flex tls_var; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with thread storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}

// Arrays of such structs are still fixed-size allocations.
struct flex array_of_flex[4]; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with static storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}

// Non-defining declarations are not diagnosed; the definition is.
extern struct flex external_obj;

// Pointers to such structs are fine; the pointee can be heap-allocated.
struct flex *ptr;
struct flex **ptr_ptr;

// A flexible array member without a count annotation is not diagnosed.
struct plain_flex {
  int count;
  char fam[];
};
struct plain_flex plain_global = {.count = 10};

// Nested struct whose FAM is the last field: diagnosed (the FAM is at the tail).
struct outer {
  int x;
  struct flex inner;
};
struct outer outer_global; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with static storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}

// FAM struct not the last field, so the FAM is not at the tail: not diagnosed.
struct middle_fam {
  int x;
  struct flex inner;
  int foo;
};
struct middle_fam middle_global;

// Unions are not descended into (no counted_by in unions): not diagnosed.
struct union_holder {
  int count;
  union {
    struct flex inner;
    long long y;
  };
};
struct union_holder union_global;

// A union has no tail field, so neither member order is diagnosed.
union union_flex_last {
  long long y;
  struct flex inner;
};
union union_flex_last union_last_global;

union union_flex_first {
  struct flex inner;
  long long y;
};
union union_flex_first union_first_global;

// FAM reached through a C11 anonymous struct: diagnosed.
struct anon_holder {
  int count;
  struct {
    int pad;
    char fam[] __counted_by(count); // expected-note {{'__counted_by' is intended for dynamically allocated objects, where the size of the allocation can be computed from the count}}
  };
};
struct anon_holder anon_global; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with static storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}

void func(void) {
  struct flex local; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with automatic storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}
  static struct flex static_local; // expected-warning {{flexible array member 'fam' with '__counted_by' attribute in an object with static storage duration; the count in 'count' cannot grow or shrink the fixed-size allocation}}
  struct flex *local_ptr;
  (void)local;
  (void)static_local;
  (void)local_ptr;
}
