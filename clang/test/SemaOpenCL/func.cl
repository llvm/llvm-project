// RUN: %clang_cc1 %s -cl-std=CL1.0 -verify -pedantic -fsyntax-only -triple spir-unknown-unknown
// RUN: %clang_cc1 %s -cl-std=CL1.0 -verify -pedantic -fsyntax-only -triple spir-unknown-unknown -DFUNCPTREXT
// RUN: %clang_cc1 %s -cl-std=CL1.0 -verify -pedantic -fsyntax-only -triple spir-unknown-unknown -DVARARGEXT

#ifdef FUNCPTREXT
#pragma OPENCL EXTENSION __cl_clang_function_pointers : enable
#endif
#ifdef VARARGEXT
#pragma OPENCL EXTENSION __cl_clang_variadic_functions : enable
#endif

// Variadic functions
void vararg_f(int, ...);
#ifndef VARARGEXT
// expected-error@-2 {{invalid prototype, variadic arguments are not allowed in OpenCL}}
#endif
void __vararg_f(int, ...);
typedef void (*vararg_fptr_t)(int, ...);
#ifndef VARARGEXT
// expected-error@-2 {{invalid prototype, variadic arguments are not allowed in OpenCL}}
#endif
#ifndef FUNCPTREXT
// expected-error@-5 {{pointers to functions are not allowed}}
#endif
int printf(__constant const char *st, ...);
#ifndef VARARGEXT
// expected-error@-2 {{invalid prototype, variadic arguments are not allowed in OpenCL}}
#endif

// Struct type with function pointer field
typedef struct s
{
   void (*f)(struct s *self, int *i);
#ifndef FUNCPTREXT
// expected-error@-2 {{pointers to functions are not allowed}}
#endif
} s_t;

//Function pointer
void foo(void*);
#ifdef FUNCPTREXT
//expected-note@-2{{passing argument to parameter here}}
#endif

// Expect no diagnostics for an empty parameter list.
void bar(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}

void bar() // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
{
  // declaring a function pointer is an error
  void (*fptr)(int);
#ifndef FUNCPTREXT
  // expected-error@-2 {{pointers to functions are not allowed}}
#endif

  // taking the address of a function is an error
  foo((void*)foo);
#ifndef FUNCPTREXT
  // expected-error@-2{{taking address of function is not allowed}}
#else
  // FIXME: Functions should probably be in the address space defined by the
  // implementation. It might make sense to put them into the Default address
  // space that is bind to a physical segment by the target rather than fixing
  // it to any of the concrete OpenCL address spaces during parsing.
  // expected-error@-8{{casting 'void (*)(__private void *__private)' to type '__private void *' changes address space}}
#endif

  foo(&foo);
#ifndef FUNCPTREXT
  // expected-error@-2{{taking address of function is not allowed}}
#else
  // expected-error@-4{{passing 'void (*)(__private void *__private)' to parameter of type '__private void *' changes address space of pointer}}
#endif

  // FIXME: If we stop rejecting the line below a bug (PR49315) gets
  // hit due to incorrectly handled pointer conversion.
#ifndef FUNCPTREXT
  // initializing an array with the address of functions is an error
  void* vptrarr[2] = {foo, &foo}; // expected-error{{taking address of function is not allowed}} expected-error{{taking address of function is not allowed}}
#endif

  // just calling a function is correct
  foo(0);
}
