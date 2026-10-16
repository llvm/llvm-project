// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -fsyntax-only -cl-std=CL2.0 -cl-ext=-cl_khr_int64_base_atomics
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -verify -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -verify -fsyntax-only -cl-std=CLC++
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -verify -fsyntax-only -cl-std=CL2.0 -cl-ext=-cl_khr_int64_base_atomics

#if defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= CL_VERSION_2_0
#define LANG_VER_OK
#endif

void atomic_types_test(void) {
// OpenCL v2.0 s6.13.11.6 defines supported atomic types.

// Non-optional types
  atomic_int i;
  atomic_uint ui;
  atomic_float f;
  atomic_flag fl;
#if !defined(LANG_VER_OK)
// expected-error@-5 {{use of undeclared identifier 'atomic_int'}}
// expected-error@-5 {{use of undeclared identifier 'atomic_uint'}}
// expected-error@-5 {{use of undeclared identifier 'atomic_float'}}
// expected-error@-5 {{use of undeclared identifier 'atomic_flag'}}
#endif

// Optional types
  atomic_long l;
  atomic_ulong ul;
  atomic_double d;
  atomic_size_t s;
  atomic_intptr_t ip;
  atomic_uintptr_t uip;
  atomic_ptrdiff_t pd;
// Optional type identifiers are not added in earlier version or if at least
// one of the extensions is not supported. Here we check with
// `cl_khr_int64_base_atomics` only.
#if !defined(LANG_VER_OK) || !defined(cl_khr_int64_base_atomics)
// expected-error@-11 {{use of undeclared identifier 'atomic_long'}}
// expected-error@-11 {{use of undeclared identifier 'atomic_ulong'}}
// expected-error@-11 {{use of undeclared identifier 'atomic_double'}}
#if defined(LANG_VER_OK)
#endif
#if !defined(LANG_VER_OK) || defined(__SPIR64__)
// expected-error@-14 {{use of undeclared identifier 'atomic_size_t'}}
// expected-error@-12 {{use of undeclared identifier 'atomic_ptrdiff_t'}}
#if !defined(LANG_VER_OK)
// expected-error@-16 {{use of undeclared identifier 'atomic_intptr_t'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_uintptr_t'}}
#else
// expected-error@-19 {{unknown type name 'atomic_intptr_t'; did you mean 'atomic_int'?}}
// expected-note@* {{'atomic_int' declared here}}
// expected-error@-20 {{unknown type name 'atomic_uintptr_t'; did you mean 'atomic_uint'?}}
// expected-note@* {{'atomic_uint' declared here}}
#endif
#endif
#endif

// OpenCL v2.0 s6.13.11.8, _Atomic type specifier and _Atomic type qualifier
// are not supported by OpenCL.
  _Atomic int i;
#ifdef __OPENCL_C_VERSION__
// expected-error@-2 {{use of undeclared identifier '_Atomic'}}
#else
 // expected-error@-4 {{unknown type name '_Atomic'}}
#endif
}

#if defined(LANG_VER_OK)
int atomic_uint; //expected-error{{redefinition of 'atomic_uint' as different kind of symbol}}
void foo(atomic_int * ptr) {}
void atomic_ops_test() {
  atomic_int i;
  foo(&i);
// OpenCL v2.0 s6.13.11.8, arithemtic operations are not permitted on atomic types.
  i++; // expected-error {{invalid argument type '__private atomic_int' (aka '__private _Atomic(int)') to unary expression}}
  i = 1; // expected-error {{atomic variable can be assigned to a variable only in global address space}}
  i += 1; // expected-error {{invalid operands to binary expression ('__private atomic_int' (aka '__private _Atomic(int)') and 'int')}}
  i = i + i; // expected-error {{invalid operands to binary expression ('__private atomic_int' (aka '__private _Atomic(int)') and '__private atomic_int')}}
}
#else
__constant int atomic_uint = 1;
#endif
