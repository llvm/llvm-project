// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

// Variadic functions
void vararg_f(int, ...);                    // expected-error {{invalid prototype, variadic arguments are not allowed in OpenCL}}
void __vararg_f(int, ...);
typedef void (*vararg_fptr_t)(int, ...);    // expected-error {{invalid prototype, variadic arguments are not allowed in OpenCL}}
int printf(__constant const char *st, ...); // expected-error {{invalid prototype, variadic arguments are not allowed in OpenCL}}

//Function pointer
void foo(void*);

void bar()
{
  // declaring a function pointer is an error
  void (*fptr)(int); // expected-error{{pointers to functions are not allowed}}

  // taking the address of a function is an error
  foo((void*)foo); // expected-error{{taking address of function is not allowed}}
  foo(&foo); // expected-error{{taking address of function is not allowed}}

  // initializing an array with the address of functions is an error
  void* vptrarr[2] = {foo, &foo}; // expected-error{{taking address of function is not allowed}} expected-error{{taking address of function is not allowed}}

  // just calling a function is correct
  foo(0);
}
