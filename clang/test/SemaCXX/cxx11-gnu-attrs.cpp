// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++11 -verify %s

// Error cases.

[[gnu::this_attribute_does_not_exist]] int unknown_attr;
// expected-warning@-1 {{unknown attribute 'this_attribute_does_not_exist' ignored}}
int [[gnu::unused]] attr_on_type;
// expected-error@-1 {{'unused' attribute cannot be applied to types}}
int *[[gnu::unused]] attr_on_ptr;
// expected-warning@-1 {{attribute 'unused' ignored, because it cannot be applied to a type}}
[[gnu::fastcall]] void pr17424_1();
// expected-warning@-1 {{calling convention 'fastcall' ignored for this target}}
[[gnu::fastcall]] [[gnu::stdcall]] void pr17424_2();
// expected-warning@-1 {{calling convention 'fastcall' ignored for this target}}
// expected-warning@-2 {{calling convention 'stdcall' ignored for this target}}
[[gnu::fastcall]] __stdcall void pr17424_3();
// expected-warning@-1 {{calling convention 'fastcall' ignored for this target}}
// expected-warning@-2 {{calling convention '__stdcall' ignored for this target}}
[[gnu::fastcall]] void pr17424_4() [[gnu::stdcall]];
// expected-warning@-1 {{calling convention 'fastcall' ignored for this target}}
// expected-warning@-2 {{attribute 'stdcall' ignored, because it cannot be applied to a type}}
void pr17424_5 [[gnu::fastcall]]();
// expected-warning@-1 {{calling convention 'fastcall' ignored for this target}}

// Valid cases.

void aliasb [[gnu::alias("_Z6alias1v")]] ();
void alias1() {}
void aliasa [[gnu::alias("_Z6alias1v")]] ();

extern struct PR22493Ty {
} PR22493 [[gnu::alias("_ZN7pcrecpp2RE6no_argE")]];

[[gnu::aligned(8)]] int aligned;
void aligned_fn [[gnu::aligned(32)]] ();
struct [[gnu::aligned(8)]] aligned_struct {};

void always_inline [[gnu::always_inline]] ();

__thread int tls_model [[gnu::tls_model("local-exec")]];

void cleanup(int *p) {
  int n [[gnu::cleanup(cleanup)]];
}

void deprecated1 [[gnu::deprecated]] (); // expected-note {{here}}
[[gnu::deprecated("custom message")]] void deprecated2(); // expected-note {{here}}
void deprecated3() {
  deprecated1(); // expected-warning {{deprecated}}
  deprecated2(); // expected-warning {{custom message}}
}

[[gnu::naked(1,2,3)]] void naked(); // expected-error {{takes no arguments}}

void nonnull [[gnu::nonnull]] (); // expected-warning {{applied to function with no pointer arguments}}

// [[gnu::noreturn]] appertains to a declaration, and marks the innermost
// function declarator in that declaration as being noreturn.
int noreturn [[gnu::noreturn]]; // expected-warning {{'noreturn' only applies to function types}}
int noreturn_fn_1();
int noreturn_fn_2() [[gnu::noreturn]]; // expected-warning {{cannot be applied to a type}}
int noreturn_fn_3 [[gnu::noreturn]] ();
[[gnu::noreturn]] int noreturn_fn_4();
int (*noreturn_fn_ptr_1 [[gnu::noreturn]])() = &noreturn_fn_1; // expected-error {{cannot initialize}}
int (*noreturn_fn_ptr_2 [[gnu::noreturn]])() = &noreturn_fn_3;
[[gnu::noreturn]] int (*noreturn_fn_ptr_3)() = &noreturn_fn_1; // expected-error {{cannot initialize}}
[[gnu::noreturn]] int (*noreturn_fn_ptr_4)() = &noreturn_fn_3;

struct [[gnu::packed]] packed { char c; int n; };
static_assert(sizeof(packed) == sizeof(char) + sizeof(int), "not packed");
