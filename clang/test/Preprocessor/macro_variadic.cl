// RUN: %clang_cc1 -verify %s -cl-std=CL1.2
// RUN: %clang_cc1 -verify %s -pedantic -DPEDANTIC -cl-std=CL1.2
// RUN: %clang_cc1 -verify %s -cl-std=CLC++
// RUN: %clang_cc1 -verify %s -pedantic -cl-std=CLC++


#define NO_VAR_FUNC(...)  5
#define VAR_FUNC(...) func(__VA_ARGS__);
#define VAR_PRINTF(str, ...) printf(str, __VA_ARGS__);
#ifdef PEDANTIC
// expected-warning@-4{{variadic macros are a Clang extension in OpenCL}}
// expected-warning@-4{{variadic macros are a Clang extension in OpenCL}}
// expected-warning@-4{{variadic macros are a Clang extension in OpenCL}}
#endif

int printf(__constant const char *st, ...);

void foo() {
  NO_VAR_FUNC(1, 2, 3);
  VAR_FUNC(1, 2, 3);
#if !__OPENCL_CPP_VERSION__
// expected-error@-2{{implicit declaration of function 'func' is invalid in OpenCL}}
#else
// expected-error@-4{{use of undeclared identifier 'func'}}
#endif
  VAR_PRINTF("%i", 1);
}
