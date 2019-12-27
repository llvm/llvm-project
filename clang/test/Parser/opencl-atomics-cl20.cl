// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -fsyntax-only -cl-std=CL2.0 -DCL20
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -verify -fsyntax-only -cl-std=CL2.0 -DCL20
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -fsyntax-only -cl-std=CL2.0 -DCL20 -DEXT -Wpedantic-core-features

#ifdef EXT
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics:enable
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#if __OPENCL_C_VERSION__ >= CL_VERSION_1_2
// expected-warning@-2{{OpenCL extension 'cl_khr_fp64' is core feature or supported optional core feature - ignoring}}
#endif
#endif

void atomic_types_test() {
// OpenCL v2.0 s6.13.11.6 defines supported atomic types.
  atomic_int i;
  atomic_uint ui;
  atomic_long l;
  atomic_ulong ul;
  atomic_float f;
  atomic_double d;
  atomic_flag fl;
  atomic_intptr_t ip;
  atomic_uintptr_t uip;
  atomic_size_t s;
  atomic_ptrdiff_t pd;
// OpenCL v2.0 s6.13.11.8, _Atomic type specifier and _Atomic type qualifier
// are not supported by OpenCL.
  _Atomic int i; // expected-error {{use of undeclared identifier '_Atomic'}}
}
#ifndef CL20
// expected-error@-16 {{use of undeclared identifier 'atomic_int'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_uint'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_long'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_ulong'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_float'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_double'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_flag'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_intptr_t'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_uintptr_t'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_size_t'}}
// expected-error@-16 {{use of undeclared identifier 'atomic_ptrdiff_t'}}
#elif !EXT
// expected-error@-26 {{use of type 'atomic_long' (aka '_Atomic(long)') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error@-27 {{use of type 'atomic_long' (aka '_Atomic(long)') requires cl_khr_int64_extended_atomics extension to be enabled}}
// expected-error@-27 {{use of type 'atomic_ulong' (aka '_Atomic(unsigned long)') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error@-28 {{use of type 'atomic_ulong' (aka '_Atomic(unsigned long)') requires cl_khr_int64_extended_atomics extension to be enabled}}
// expected-error@-27 {{use of type 'atomic_double' (aka '_Atomic(double)') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error@-28 {{use of type 'atomic_double' (aka '_Atomic(double)') requires cl_khr_int64_extended_atomics extension to be enabled}}
#if __LP64__
// expected-error-re@-28 {{use of type 'atomic_intptr_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error-re@-29 {{use of type 'atomic_intptr_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_extended_atomics extension to be enabled}}
// expected-error-re@-29 {{use of type 'atomic_uintptr_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error-re@-30 {{use of type 'atomic_uintptr_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_extended_atomics extension to be enabled}}
// expected-error-re@-30 {{use of type 'atomic_size_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error-re@-31 {{use of type 'atomic_size_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_extended_atomics extension to be enabled}}
// expected-error-re@-31 {{use of type 'atomic_ptrdiff_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_base_atomics extension to be enabled}}
// expected-error-re@-32 {{use of type 'atomic_ptrdiff_t' (aka '_Atomic({{.+}})') requires cl_khr_int64_extended_atomics extension to be enabled}}
#endif
#endif

#ifdef CL20
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
#endif
