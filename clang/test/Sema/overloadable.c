// RUN: %clang_cc1 -fsyntax-only -verify %s -Wincompatible-pointer-types

int var __attribute__((overloadable)); // expected-error{{'overloadable' attribute only applies to functions}}
void params(void) __attribute__((overloadable(12))); // expected-error {{'overloadable' attribute takes no arguments}}

int *f(int) __attribute__((overloadable)); // expected-note{{previous overload of function is here}}
float *f(float);
int *f(int); // expected-error{{redeclaration of 'f' must have the 'overloadable' attribute}} \
             // expected-note{{previous declaration is here}}
double *f(double) __attribute__((overloadable)); // okay, new

// Ensure we don't complain about overloadable on implicitly declared functions.
int isdigit(int) __attribute__((overloadable));

void test_f(int iv, float fv, double dv) {
  int *ip = f(iv);
  float *fp = f(fv);
  double *dp = f(dv);
}

int *accept_funcptr(int (*)()) __attribute__((overloadable)); //         \
  // expected-note{{candidate function}}
float *accept_funcptr(int (*)(int, double)) __attribute__((overloadable)); //  \
  // expected-note{{candidate function}}

void test_funcptr(int (*f1)(int, double),
                  int (*f2)(int, float)) {
  float *fp = accept_funcptr(f1);
  accept_funcptr(f2); // expected-error{{no matching function for call to 'accept_funcptr'}}
}

struct X { int x; float y; };
struct Y { int x; float y; };
int* accept_struct(struct X x) __attribute__((__overloadable__));
float* accept_struct(struct Y y) __attribute__((overloadable));

void test_struct(struct X x, struct Y y) {
  int *ip = accept_struct(x);
  float *fp = accept_struct(y);
}

double *f(int) __attribute__((overloadable)); // expected-error{{conflicting types for 'f'}}

double promote(float) __attribute__((__overloadable__));
double promote(double) __attribute__((__overloadable__));
long double promote(long double) __attribute__((__overloadable__));

void promote(...) __attribute__((__overloadable__, __unavailable__)); // expected-note {{marked unavailable here}}

void test_promote(short* sp) {
  promote(1.0);
  promote(sp); // expected-error{{'promote' is unavailable}}
}

// PR6600
typedef double Double;
typedef Double DoubleVec __attribute__((vector_size(16)));
typedef int Int;
typedef Int IntVec __attribute__((vector_size(16)));
double magnitude(DoubleVec) __attribute__((__overloadable__));
double magnitude(IntVec) __attribute__((__overloadable__));
double test_p6600(DoubleVec d) {
  return magnitude(d) * magnitude(d);
}

// PR7738
extern int __attribute__((overloadable)) f0(); // expected-error{{'overloadable' function 'f0' must have a prototype}}
typedef int f1_type();
f1_type __attribute__((overloadable)) f1; // expected-error{{'overloadable' function 'f1' must have a prototype}}

void test() {
  f0();
  f1();
}

void before_local_1(int) __attribute__((overloadable));
void before_local_2(int); // expected-note {{here}}
void before_local_3(int) __attribute__((overloadable));
void local() {
  void before_local_1(char);
  void before_local_2(char); // expected-error {{conflicting types}}
  void before_local_3(char) __attribute__((overloadable));
  void after_local_1(char);
  void after_local_2(char) __attribute__((overloadable));
  void after_local_3(char) __attribute__((overloadable));
}
void after_local_1(int) __attribute__((overloadable));
void after_local_2(int);
void after_local_3(int) __attribute__((overloadable));

// Make sure we allow C-specific conversions in C.
void conversions() {
  void foo(char *c) __attribute__((overloadable));
  void foo(char *c) __attribute__((overloadable, enable_if(c, "nope.jpg")));

  void *ptr;
  foo(ptr);

  void multi_type(unsigned char *c) __attribute__((overloadable));
  void multi_type(signed char *c) __attribute__((overloadable));
  unsigned char *c;
  multi_type(c);
}

// Ensure that we allow C-specific type conversions in C
void fn_type_conversions() {
  void foo(void *c) __attribute__((overloadable));
  void foo(char *c) __attribute__((overloadable));
  void (*ptr1)(void *) = &foo;
  void (*ptr2)(char *) = &foo;
  void (*ambiguous)(int *) = &foo; // expected-error{{initializing 'void (*)(int *)' with an expression of incompatible type '<overloaded function type>'}} expected-note@-4{{candidate function}} expected-note@-3{{candidate function}}
  void *vp_ambiguous = &foo; // expected-error{{initializing 'void *' with an expression of incompatible type '<overloaded function type>'}} expected-note@-5{{candidate function}} expected-note@-4{{candidate function}}

  void (*specific1)(int *) = (void (*)(void *))&foo; // expected-warning{{incompatible function pointer types initializing 'void (*)(int *)' with an expression of type 'void (*)(void *)'}}
  void *specific2 = (void (*)(void *))&foo;

  void disabled(void *c) __attribute__((overloadable, enable_if(0, "")));
  void disabled(int *c) __attribute__((overloadable, enable_if(c, "")));
  void disabled(char *c) __attribute__((overloadable, enable_if(1, "The function name lies.")));
  // To be clear, these should all point to the last overload of 'disabled'
  void (*dptr1)(char *c) = &disabled;
  void (*dptr2)(void *c) = &disabled; // expected-warning{{incompatible function pointer types initializing 'void (*)(void *)' with an expression of type '<overloaded function type>'}} expected-note@-5{{candidate function made ineligible by enable_if}} expected-note@-4{{candidate function made ineligible by enable_if}} expected-note@-3{{candidate function has type mismatch at 1st parameter (expected 'void *' but has 'char *')}}
  void (*dptr3)(int *c) = &disabled; // expected-warning{{incompatible function pointer types initializing 'void (*)(int *)' with an expression of type '<overloaded function type>'}} expected-note@-6{{candidate function made ineligible by enable_if}} expected-note@-5{{candidate function made ineligible by enable_if}} expected-note@-4{{candidate function has type mismatch at 1st parameter (expected 'int *' but has 'char *')}}

  void *specific_disabled = &disabled;
}

void incompatible_pointer_type_conversions() {
  char charbuf[1];
  unsigned char ucharbuf[1];
  int intbuf[1];

  void foo(char *c) __attribute__((overloadable));
  void foo(short *c) __attribute__((overloadable));
  foo(charbuf);
  foo(ucharbuf); // expected-error{{call to 'foo' is ambiguous}} expected-note@-3{{candidate function}} expected-note@-2{{candidate function}}
  foo(intbuf); // expected-error{{call to 'foo' is ambiguous}} expected-note@-4{{candidate function}} expected-note@-3{{candidate function}}

  void bar(unsigned char *c) __attribute__((overloadable));
  void bar(signed char *c) __attribute__((overloadable));
  bar(charbuf); // expected-error{{call to 'bar' is ambiguous}} expected-note@-2{{candidate function}} expected-note@-1{{candidate function}}
  bar(ucharbuf);
  bar(intbuf); // expected-error{{call to 'bar' is ambiguous}} expected-note@-4{{candidate function}} expected-note@-3{{candidate function}}
}

void dropping_qualifiers_is_incompatible() {
  const char ccharbuf[1];
  volatile char vcharbuf[1];

  void foo(char *c) __attribute__((overloadable));
  void foo(const volatile unsigned char *c) __attribute__((overloadable));

  foo(ccharbuf); // expected-error{{call to 'foo' is ambiguous}} expected-note@-3{{candidate function}} expected-note@-2{{candidate function}}
  foo(vcharbuf); // expected-error{{call to 'foo' is ambiguous}} expected-note@-4{{candidate function}} expected-note@-3{{candidate function}}
}

void overloadable_with_global() {
  void wg_foo(void) __attribute__((overloadable)); // expected-note{{previous}}
  void wg_foo(int) __attribute__((overloadable));
}

int wg_foo; // expected-error{{redefinition of 'wg_foo' as different kind of symbol}}

#if !__has_extension(overloadable_unmarked)
#error "We should have unmarked overload support"
#endif

void to_foo0(int);
void to_foo0(double) __attribute__((overloadable)); // expected-note{{previous overload}}
void to_foo0(int);
void to_foo0(double); // expected-error{{must have the 'overloadable' attribute}}
void to_foo0(int);

void to_foo1(int) __attribute__((overloadable)); // expected-note 2{{previous overload}}
void to_foo1(double);
void to_foo1(int); // expected-error{{must have the 'overloadable' attribute}}
void to_foo1(double);
void to_foo1(int); // expected-error{{must have the 'overloadable' attribute}}

void to_foo2(int); // expected-note{{previous unmarked overload}}
void to_foo2(double) __attribute__((overloadable)); // expected-note 2{{previous overload}}
void to_foo2(int) __attribute__((overloadable)); // expected-error {{must not have the 'overloadable' attribute}}
void to_foo2(double); // expected-error{{must have the 'overloadable' attribute}}
void to_foo2(int);
void to_foo2(double); // expected-error{{must have the 'overloadable' attribute}}
void to_foo2(int);

void to_foo3(int);
void to_foo3(double) __attribute__((overloadable)); // expected-note{{previous overload}}
void to_foo3(int);
void to_foo3(double); // expected-error{{must have the 'overloadable' attribute}}

void to_foo4(int) __attribute__((overloadable)); // expected-note{{previous overload}}
void to_foo4(int); // expected-error{{must have the 'overloadable' attribute}}
void to_foo4(double) __attribute__((overloadable));

void to_foo5(int);
void to_foo5(int); // expected-note 3{{previous unmarked overload}}
void to_foo5(float) __attribute__((overloadable));
void to_foo5(double); // expected-error{{at most one overload for a given name may lack the 'overloadable' attribute}}
void to_foo5(float) __attribute__((overloadable));
void to_foo5(short); // expected-error{{at most one overload for a given name may lack the 'overloadable' attribute}}
void to_foo5(long); // expected-error{{at most one overload for a given name may lack the 'overloadable' attribute}}
void to_foo5(double) __attribute__((overloadable));

void to_foo6(int) __attribute__((enable_if(1, ""), overloadable)); // expected-note{{previous overload}}
void to_foo6(int) __attribute__((enable_if(1, ""))); // expected-error{{must have the 'overloadable' attribute}}
void to_foo6(int) __attribute__((enable_if(1, ""), overloadable));

void to_foo7(int) __attribute__((enable_if(1, ""))); // expected-note{{previous unmarked overload}}
void to_foo7(int) __attribute__((enable_if(1, ""), overloadable)); // expected-error{{must not have the 'overloadable' attribute}}
void to_foo7(int) __attribute__((enable_if(1, "")));

void to_foo8(char *__attribute__((pass_object_size(0))))
  __attribute__((enable_if(1, "")));
void to_foo8(char *__attribute__((pass_object_size(0))))
  __attribute__((overloadable));

void to_foo9(int); // expected-note{{previous unmarked overload}}
// FIXME: It would be nice if we did better with the "previous unmarked
// overload" diag.
void to_foo9(int) __attribute__((overloadable)); // expected-error{{must not have the 'overloadable' attribute}} expected-note{{previous declaration}} expected-note{{previous unmarked overload}}
void to_foo9(float); // expected-error{{conflicting types for 'to_foo9'}}
void to_foo9(float) __attribute__((overloadable));
void to_foo9(double); // expected-error{{at most one overload for a given name may lack the 'overloadable' attribute}}
void to_foo9(double) __attribute__((overloadable));

void to_foo10(int) __attribute__((overloadable));
void to_foo10(double); // expected-note{{previous unmarked overload}}
// no "note: previous redecl" if no previous decl has `overloadable`
// spelled out
void to_foo10(float); // expected-error{{at most one overload for a given name may lack the 'overloadable' attribute}}
void to_foo10(float); // expected-error{{must have the 'overloadable' attribute}}
void to_foo10(float); // expected-error{{must have the 'overloadable' attribute}}

// Bug: we used to treat `__typeof__(foo)` as though it was `__typeof__(&foo)`
// if `foo` was overloaded with only one function that could have its address
// taken.
void typeof_function_is_not_a_pointer() {
  void not_a_pointer(void *) __attribute__((overloadable));
  void not_a_pointer(char *__attribute__((pass_object_size(1))))
    __attribute__((overloadable));

  __typeof__(not_a_pointer) *fn;

  void take_fn(void (*)(void *));
  // if take_fn is passed a void (**)(void *), we'll get a warning.
  take_fn(fn);
}
