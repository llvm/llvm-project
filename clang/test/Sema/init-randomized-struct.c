// RUN: %clang_cc1 -triple=x86_64-unknown-linux -frandomize-layout-seed=1234567890abcded \
// RUN:  -verify -fsyntax-only -Werror %s

// NOTE: The current seed (1234567890abcded) is specifically chosen because it
// uncovered a bug in diagnostics. With it the randomization of "t9" places the
// "a" element at the end of the record. When that happens, the clang complains
// about excessive initializers, which is confusing, because there aren't
// excessive initializers. It should instead complain about using a
// non-designated initializer on a raqndomized struct.

// Initializing a randomized structure requires a designated initializer,
// otherwise the element ordering will be off. The only exceptions to this rule
// are:
//
//    - A structure with only one element, and
//    - A structure initialized with "{0}".
//
// These are well-defined situations where the field ordering doesn't affect
// the result.

typedef void (*func_ptr)();

void foo(void);
void bar(void);
void baz(void);
void gaz(void);

struct test {
  func_ptr a;
  func_ptr b;
  func_ptr c;
  func_ptr d;
  func_ptr e;
  func_ptr f;
  func_ptr g;
} __attribute__((randomize_layout));

struct test t1 = {}; // This should be fine per WG14 N2900 (in C23) + our extension handling of it in earlier modes
struct test t2 = { 0 }; // This should also be fine per C99 6.7.8p19
struct test t3 = { .f = baz, .b = bar, .g = gaz, .a = foo }; // Okay
struct test t4 = { .a = foo, bar, baz }; // expected-error {{a randomized struct can only be initialized with a designated initializer}}

struct other_test {
  func_ptr a;
  func_ptr b[3];
  func_ptr c;
} __attribute__((randomize_layout));

struct other_test t5 = { .a = foo, .b[0] = foo }; // Okay
struct other_test t6 = { .a = foo, .b[0] = foo, bar, baz }; // Okay
struct other_test t7 = { .a = foo, .b = { foo, bar, baz } }; // Okay
struct other_test t8 = { baz, bar, gaz, foo }; // expected-error {{a randomized struct can only be initialized with a designated initializer}}
struct other_test t9 = { .a = foo, .b[0] = foo, bar, baz, gaz }; // expected-error {{a randomized struct can only be initialized with a designated initializer}}

struct empty_test {
} __attribute__((randomize_layout));

struct empty_test t10 = {}; // Okay

struct degen_test {
  func_ptr a;
} __attribute__((randomize_layout));

struct degen_test t11 = { foo }; // Okay

struct static_assert_test {
  int f;
  _Static_assert(sizeof(int) == 4, "oh no!");
} __attribute__((randomize_layout));

struct static_assert_test t12 = { 42 }; // Okay

struct enum_decl_test {
  enum e { BORK = 42, FORK = 9 } f;
} __attribute__((randomize_layout));

struct enum_decl_test t13 = { BORK }; // Okay

struct mixed {
  int a;
  short b;
  unsigned c;
  char d;
} __attribute__((randomize_layout));

struct mixed t14 = { 7 }; // expected-error {{a randomized struct can only be initialized with a designated initializer}}
struct mixed t15 = { .b = 8 }; // Okay

// This should be autodetected as randomized.
struct funcs {
  func_ptr a;
  func_ptr b;
  func_ptr c;
  func_ptr d;
};

struct funcs t16 = { .c = foo }; // Okay
struct funcs t17 = { foo }; // expected-error {{a randomized struct can only be initialized with a designated initializer}}

// This should be forced off.
struct funcs_unshuffled {
  func_ptr a;
  func_ptr b;
  func_ptr c;
  func_ptr d;
} __attribute__((no_randomize_layout));

struct funcs_unshuffled t18 = { .d = foo }; // Okay
struct funcs_unshuffled t19 = { foo }; // Okay

// This is still all function pointers.
// https://github.com/llvm/llvm-project/issues/138355
struct funcs_composite {
  func_ptr a;
  func_ptr b;
  struct funcs inner;
  func_ptr c;
  func_ptr d;
};

struct funcs_composite t20 = { .a = foo }; // Okay
struct funcs_composite t21 = { .inner.c = foo }; // Okay
struct funcs_composite t22 = { foo }; // expected-error {{a randomized struct can only be initialized with a designated initializer}}
