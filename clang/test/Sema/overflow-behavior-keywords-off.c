// RUN: %clang_cc1 -fsyntax-only -verify %s
// Verify that __ob_wrap and __ob_trap are ignored when
// -Xclang -fexperimental-overflow-behavior-types is not passed.

typedef int __ob_wrap wrap_int; // expected-warning {{'__ob_wrap' specifier is ignored because overflow behavior types are not enabled;}}
typedef int __ob_trap trap_int; // expected-warning {{'__ob_trap' specifier is ignored because overflow behavior types are not enabled;}}

int __ob_wrap a; // expected-warning {{'__ob_wrap' specifier is ignored}}
int __ob_trap b; // expected-warning {{'__ob_trap' specifier is ignored}}

// The type should just be plain int when the flag is off.
int *p_a = &a;
int *p_b = &b;

// Test with pointer type qualifiers
int * __ob_wrap ptr_w; // expected-warning {{'__ob_wrap' specifier is ignored}}
int * __ob_trap ptr_t; // expected-warning {{'__ob_trap' specifier is ignored}}

void test_params(int __ob_wrap x, int __ob_trap y) { // expected-warning {{'__ob_wrap' specifier is ignored}} expected-warning {{'__ob_trap' specifier is ignored}}
  (void)x;
  (void)y;
}

struct S {
  int __ob_wrap member; // expected-warning {{'__ob_wrap' specifier is ignored}}
};
