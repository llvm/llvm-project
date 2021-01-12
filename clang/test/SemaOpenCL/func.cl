// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -triple spir-unknown-unknown
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -triple spir-unknown-unknown -DFUNCPTREXT
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -triple spir-unknown-unknown -DVARARG

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

// Expect no diagnostics for an empty parameter list.
void bar();

void bar()
{
  // declaring a function pointer is an error
  void (*fptr)(int);
#ifndef FUNCPTREXT
  // expected-error@-2 {{pointers to functions are not allowed}}
#endif

  // taking the address of a function is an error
  foo((void*)foo); // expected-error{{taking address of function is not allowed}}
  foo(&foo); // expected-error{{taking address of function is not allowed}}

  // initializing an array with the address of functions is an error
  void* vptrarr[2] = {foo, &foo}; // expected-error{{taking address of function is not allowed}} expected-error{{taking address of function is not allowed}}

  // just calling a function is correct
  foo(0);
}
