// RUN: %clang_cc1 -std=c11 %s -fsyntax-only -verify -pedantic
// RUN: %clang_cc1 -std=c99 %s -fsyntax-only -verify=expected,ext -pedantic -Wno-typedef-redefinition

typedef _Atomic(int) atomic_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic int atomic_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic _Atomic _Atomic(int) atomic_int; // expected-warning {{duplicate '_Atomic' declaration specifier}} \
                                                 // ext-warning 3 {{'_Atomic' is a C11 extension}}

typedef const int const_int;

typedef const atomic_int const_atomic_int;
typedef _Atomic const int const_atomic_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef const _Atomic int const_atomic_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef const _Atomic(int) const_atomic_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef const _Atomic(_Atomic int) const_atomic_int; // expected-error {{_Atomic cannot be applied to atomic type '_Atomic(int)'}} \
                                                     // ext-warning 2 {{'_Atomic' is a C11 extension}}
typedef _Atomic const_int const_atomic_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic(const_int) const_atomic_int; // expected-error {{_Atomic cannot be applied to qualified type 'const_int' (aka 'const int')}} \
                                             // ext-warning {{'_Atomic' is a C11 extension}}

typedef int *_Atomic atomic_int_ptr; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic(int *) atomic_int_ptr; // ext-warning {{'_Atomic' is a C11 extension}}
typedef int (*_Atomic atomic_int_ptr); // ext-warning {{'_Atomic' is a C11 extension}}

typedef int _Atomic *int_atomic_ptr; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic(int) *int_atomic_ptr; // ext-warning {{'_Atomic' is a C11 extension}}

typedef int int_fn();
typedef _Atomic int_fn atomic_int_fn; // expected-error {{_Atomic cannot be applied to function type 'int_fn' (aka 'int ()')}} \
                                      // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic int atomic_int_array[3]; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic atomic_int_array atomic_int_atomic_array; // expected-error {{_Atomic cannot be applied to array type 'atomic_int_array' (aka '_Atomic(int)[3]')}} \
                                                          // ext-warning {{'_Atomic' is a C11 extension}}

_Atomic struct S { int n; }; // expected-warning {{'_Atomic' ignored on this declaration}} \
                             // ext-warning {{'_Atomic' is a C11 extension}}

typedef _Atomic int __attribute__((address_space(1))) atomic_addr_space_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic(int) __attribute__((address_space(1))) atomic_addr_space_int; // ext-warning {{'_Atomic' is a C11 extension}}

typedef _Atomic int __attribute__((vector_size(16))) atomic_vector_int; // ext-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic(int __attribute__((vector_size(16)))) atomic_vector_int; // ext-warning {{'_Atomic' is a C11 extension}}

struct S
_Atomic atomic_s_no_missing_semicolon; // ext-warning {{'_Atomic' is a C11 extension}}

int *const _Atomic atomic_return_type(); // ext-warning {{'_Atomic' is a C11 extension}}
