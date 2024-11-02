// RUN: %clang_cc1 %s -fsyntax-only -verify -verify=c2x -pedantic -Wno-strict-prototypes

// PR1892, PR11354
void f(double a[restrict][5]) { __typeof(a) x = 10; } // expected-warning {{(aka 'double (*restrict)[5]')}}

int foo (__const char *__path);
int foo(__const char *__restrict __file);

void func(const char*); // expected-note {{previous declaration is here}}
void func(char*); // expected-error{{conflicting types for 'func'}}

void g(int (*)(const void **, const void **));
void g(int (*compar)()) {
}

void h();  // expected-note {{previous declaration is here}}
void h (const char *fmt, ...) {} // expected-error{{conflicting types for 'h'}}

// PR1965
int t5(b);          // expected-error {{parameter list without types}}
int t6(int x, g);   // expected-error {{type specifier missing, defaults to 'int'}}

int t7(, );       // expected-error {{expected parameter declarator}} expected-error {{expected parameter declarator}}
int t8(, int a);  // expected-error {{expected parameter declarator}}
int t9(int a, );  // expected-error {{expected parameter declarator}}


// PR2042
void t10(){}
void t11(){t10(1);} // expected-warning{{too many arguments}}

// PR3208
void t12(int) {}  // c2x-warning{{omitting the parameter name in a function definition is a C2x extension}}

// PR2790
void t13() {
  return 0; // expected-error {{void function 't13' should not return a value}}
}
int t14() {
  return; // expected-error {{non-void function 't14' should return a value}}
}

// <rdar://problem/6097326>
y(y) { return y; } // expected-error{{parameter 'y' was not declared, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                   // expected-error{{type specifier missing, defaults to 'int'}}


// PR3137, <rdar://problem/6127293>
extern int g0_3137(void);
void f0_3137() {
  int g0_3137(void);
}
void f1_3137() {
  int (*fp)(void) = g0_3137;
}

void f1static() {
  static void f2static(int); // expected-error{{function declared in block scope cannot have 'static' storage class}}
  register void f2register(int); // expected-error{{illegal storage class on function}}
}

struct incomplete_test a(void) {} // expected-error{{incomplete result type 'struct incomplete_test' in function definition}} \
    // expected-note{{forward declaration of 'struct incomplete_test'}}


extern __inline
__attribute__((__gnu_inline__))
void gnu_inline1() {}

void
__attribute__((__gnu_inline__)) // expected-warning {{'gnu_inline' attribute requires function to be marked 'inline', attribute ignored}}
gnu_inline2() {}


// rdar://6802350
inline foo_t invalid_type() {  // expected-error {{unknown type name 'foo_t'}}
}

typedef void fn_t(void);
fn_t t17;

// PR4049
unknown_type t18(void*) {   // expected-error {{unknown type name 'unknown_type'}} \
                            // c2x-warning {{omitting the parameter name in a function definition is a C2x extension}}
}

unknown_type t19(int* P) {   // expected-error {{unknown type name 'unknown_type'}}
  P = P+1;  // no warning.
}

// missing ',' before '...'
void t20(int i...) { } // expected-error {{requires a comma}}

int n;
void t21(int n, int (*array)[n]);

int func_e(int x) {
  int func_n(int y) { // expected-error {{function definition is not allowed here}}
    if (y > 22) {
      return y+2;
    } else {
      return y-2;
    }
  }
  return x + 3;
}

void decays(int a[3][3]);   // expected-note {{passing argument to parameter 'a' here}}
void no_decay(int (*a)[3]); // expected-note {{passing argument to parameter 'a' here}}

void t22(int *ptr, int (*array)[3]) {
  decays(ptr);   // expected-warning {{incompatible pointer types passing 'int *' to parameter of type 'int (*)[3]'}}
  no_decay(ptr); // expected-warning {{incompatible pointer types passing 'int *' to parameter of type 'int (*)[3]'}}
  decays(array);
  no_decay(array);
}

void const Bar (void); // ok on decl
// PR 20146
void const Bar (void) // also okay on defn per DR 423
{
}
