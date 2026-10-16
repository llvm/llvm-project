// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fsyntax-only -verify %s

#if !__has_attribute(copy) || !__has_attribute(__copy__)
#error copy is not supported
#endif
#if __STDC_VERSION__ >= 202311L
#if !__has_c_attribute(gnu::copy)
#error gnu::copy is not supported
#endif
#endif

void source(int *, int *) __attribute__((nonnull(2)));
void copied(int *, int *) __attribute__((copy(source)));
void address(int *, int *) __attribute__((__copy__(&source)));
void chained(int *, int *) __attribute__((copy(copied)));

void use(void) {
  copied((void *)0, (void *)0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  address((void *)0, (void *)0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  chained((void *)0, (void *)0); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

// The destination must still satisfy the copied attribute's requirements.
void bad_index(int *) __attribute__((copy(source))); // expected-error {{'nonnull' attribute parameter 1 is out of bounds}}
void bad_type(int, int) __attribute__((copy(source))); // expected-warning {{'nonnull' attribute only applies to pointer arguments}} expected-warning {{'nonnull' attribute applied to function with no pointer arguments}}
void *allocator(int) __attribute__((alloc_size(1)));
void *bad_alloc(void *) __attribute__((copy(allocator))); // expected-error {{'alloc_size' attribute argument may only refer to a function parameter of integer type}}

int print_source(const char *, ...) __attribute__((format(printf, 1, 2)));
int print_copy(const char *, ...) __attribute__((copy(print_source)));
void use_format(void) {
  print_copy("%d", "wrong"); // expected-warning {{format specifies type 'int' but the argument has type 'char *'}}
}

void sentinel_source(int, ...) __attribute__((sentinel));
void sentinel_copy(int, ...) __attribute__((copy(sentinel_source))); // expected-note {{function has been explicitly marked sentinel here}}
void use_sentinel(void) {
  sentinel_copy(0, 1); // expected-warning {{missing sentinel in function call}}
}

void cold_source(void) __attribute__((cold));
// Excluding inline attributes from copy must not relax explicit conflicts.
void explicit_conflict(void) __attribute__((noinline, always_inline)); // expected-error {{'always_inline' and 'noinline' attributes are not compatible}} expected-note {{conflicting attribute is here}}
void conflict(void) __attribute__((hot, copy(cold_source))); // expected-error {{'cold' and 'hot' attributes are not compatible}} expected-note {{conflicting attribute is here}}
static void weak_alias(void)
    __attribute__((weakref, copy(cold_source), alias("cold_source")));

int variable __attribute__((aligned(32)));
int variable_copy __attribute__((copy(variable)));
int variable_address __attribute__((copy(&variable)));
_Static_assert(__alignof__(variable_copy) == 32, "variable alignment");
_Static_assert(__alignof__(variable_address) == 32, "variable address");

void shadow(void) {
  int local __attribute__((aligned(32)));
  {
    int local __attribute__((copy(local)));
    _Static_assert(__alignof__(local) == 32, "copy from an enclosing scope");
  }
}

struct __attribute__((packed, aligned(16))) A { char c; int i; };
struct __attribute__((copy((struct A *)0))) B { char c; int i; };
_Static_assert(_Alignof(struct B) == 16, "type alignment");
_Static_assert(__builtin_offsetof(struct B, i) == 1, "packed layout");

typedef int aligned_int __attribute__((aligned(32)));
typedef int copied_int __attribute__((copy((aligned_int *)0)));
_Static_assert(_Alignof(copied_int) == 32, "typedef alignment");

// Expressions whose types have no attributes are valid, and are unevaluated.
int no_attributes __attribute__((copy((int *)0)));
int unevaluated __attribute__((copy(variable++)));

void (*pointer_source)(void) __attribute__((aligned(32)));
void (*pointer_copy)(void) __attribute__((copy(pointer_source)));
_Static_assert(__alignof__(pointer_copy) == 32, "function pointer variable");
void pointer_kind(void) __attribute__((copy(*pointer_source))); // expected-warning {{'copy' attribute ignored on a declaration of a different kind than its argument}}

int no_argument __attribute__((copy)); // expected-error {{'copy' attribute takes one argument}}
int two_arguments __attribute__((copy(variable, variable))); // expected-error {{'copy' attribute takes one argument}}
int integer __attribute__((copy(1))); // expected-error {{'copy' attribute requires an expression referring to a function, variable, or type}}
int string __attribute__((copy("variable"))); // expected-error {{'copy' attribute requires an expression referring to a function, variable, or type}}
int missing __attribute__((copy(undeclared))); // expected-error {{use of undeclared identifier 'undeclared'}}
int wrong_kind __attribute__((copy(source))); // expected-warning {{'copy' attribute ignored on a declaration of a different kind than its argument}}
void wrong_function(void) __attribute__((copy(variable))); // expected-warning {{'copy' attribute ignored on a declaration of a different kind than its argument}}

extern int self;
extern int self __attribute__((copy(self))); // expected-warning {{'copy' attribute ignored on a declaration referring to itself}}

#if __STDC_VERSION__ >= 202311L
[[gnu::copy(source)]] void standard_copy(int *, int *);
void use_standard(void) {
  standard_copy((void *)0, (void *)0); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
#endif
