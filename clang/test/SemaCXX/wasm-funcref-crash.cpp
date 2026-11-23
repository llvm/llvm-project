// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -triple wasm32 -target-feature +reference-types %s

// Test for issue #118233 - crash when using __funcref on non-function-pointer types in C++

// Valid usage - __funcref on function pointer type
int (* __funcref valid_ptr)();
typedef void (*__funcref funcref_t)();

// Invalid usage - __funcref on non-pointer types should give error, not crash
__funcref int global_var; // expected-error {{'__funcref' attribute can only be applied to a function pointer type}} \
                          // expected-error {{'__funcref' attribute only applies to functions pointers}}

typedef __funcref int bad_typedef; // expected-error {{'__funcref' attribute can only be applied to a function pointer type}} \
                                   // expected-error {{'__funcref' attribute only applies to functions pointers}}

void test_funcref_non_pointer() {
  __funcref int local_var; // expected-error {{'__funcref' attribute can only be applied to a function pointer type}} \
                           // expected-error {{'__funcref' attribute only applies to functions pointers}}
}

// Invalid - __funcref on non-function pointer (pointer to int)
typedef int *__funcref bad_ptr_typedef; // expected-error {{'__funcref' attribute can only be applied to a function pointer type}} \
                                        // expected-error {{'__funcref' attribute only applies to functions pointers}}

// Invalid - __funcref with missing type in parameter (C++ requires type)
void foo(__funcref x); // expected-error {{unknown type name 'x'}} \
                       // expected-error {{'__funcref' attribute can only be applied to a function pointer type}} \
                       // expected-error {{'__funcref' attribute only applies to functions pointers}}
