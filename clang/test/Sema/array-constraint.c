// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

struct s;  // expected-note 2 {{forward declaration of 'struct s'}}
struct s* t (struct s z[]) {   // expected-error {{array has incomplete element type}}
  return z;
}

void ff() { 
  struct s v, *p; // expected-error {{variable has incomplete type 'struct s'}}

  p = &v;
}

void *k (void l[2]) {          // expected-error {{array has incomplete element type}}
  return l; 
}

struct vari {
  int a;
  int b[];
};

struct vari *func(struct vari a[]) { // expected-warning {{'struct vari' may not be used as an array element due to flexible array member}}
  return a;
}

int foo[](void);  // expected-error {{'foo' declared as array of functions}}
int foo2[1](void);  // expected-error {{'foo2' declared as array of functions}}

typedef int (*pfunc)(void);

pfunc xx(int f[](void)) { // expected-error {{'f' declared as array of functions}}
  return f;
}

void check_size() {
  float f;
  int size_not_int[f]; // expected-error {{size of array has non-integer type 'float'}}
  int negative_size[1-2]; // expected-error{{array with a negative size}}
  int zero_size[0]; // expected-warning{{zero size arrays are an extension}}
}

static int I;
typedef int TA[I]; // expected-error {{variable length array declaration not allowed at file scope}}

void strFunc(char *); // expected-note{{passing argument to parameter here}}
const char staticAry[] = "test";
void checkStaticAry() { 
  strFunc(staticAry); // expected-warning{{passing 'const char [5]' to parameter of type 'char *' discards qualifiers}}
}


