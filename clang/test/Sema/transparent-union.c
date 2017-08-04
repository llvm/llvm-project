// RUN: %clang_cc1 -fsyntax-only -verify %s
typedef union {
  int *ip;
  float *fp;
  long *__restrict rlp;
  void *vpa[1];
} TU __attribute__((transparent_union));

void f(TU); // expected-note{{passing argument to parameter here}}

void g(int *ip, float *fp, char *cp) {
  f(ip);
  f(fp);
  f(cp); // expected-error{{incompatible type}}
  f(0);

  TU tu_ip = ip; // expected-error{{incompatible type}}
  TU tu;
  tu.ip = ip;
}

/* Test ability to redeclare a function taking a transparent_union arg
   with various compatible and incompatible argument types. */

void fip(TU);
void fip(int *i) {}

void ffp(TU);
void ffp(float *f) {}

void flp(TU);
void flp(long *l) {}

void fvp(TU); // expected-note{{previous declaration is here}}
void fvp(void *p) {} // expected-error{{conflicting types}}

void fsp(TU); // expected-note{{previous declaration is here}}
void fsp(short *s) {} // expected-error{{conflicting types}}

void fi(TU); // expected-note{{previous declaration is here}}
void fi(int i) {} // expected-error{{conflicting types}}

void fvpp(TU); // expected-note{{previous declaration is here}}
void fvpp(void **v) {} // expected-error{{conflicting types}}

/* FIXME: we'd like to just use an "int" here and align it differently
   from the normal "int", but if we do so we lose the alignment
   information from the typedef within the compiler. */
typedef struct { int x, y; } __attribute__((aligned(8))) aligned_struct8;

typedef struct { int x, y; } __attribute__((aligned(4))) aligned_struct4;
typedef union {
  aligned_struct4 s4; // expected-note{{alignment of first field}}
  aligned_struct8 s8; // expected-warning{{alignment of field}}
} TU1 __attribute__((transparent_union));

typedef union __attribute__((transparent_union)) {
  aligned_struct4 s4; // expected-note{{alignment of first field}}
  aligned_struct8 s8; // expected-warning{{alignment of field}}
} TU1b ;

typedef union {
  char c; // expected-note{{size of first field is 8 bits}}
  int i; // expected-warning{{size of field}}
} TU2 __attribute__((transparent_union));

typedef union __attribute__((transparent_union)){
  char c; // expected-note{{size of first field is 8 bits}}
  int i; // expected-warning{{size of field}}
} TU2b;

typedef union {
  float f; // expected-warning{{floating}}
} TU3 __attribute__((transparent_union));

typedef union { } TU4 __attribute__((transparent_union)); // expected-warning{{field}} 

typedef int int4 __attribute__((ext_vector_type(4)));
typedef union {
  int4 vec; // expected-warning{{first field of a transparent union cannot have vector type 'int4' (vector of 4 'int' values); transparent_union attribute ignored}}
} TU5 __attribute__((transparent_union));

union pr15134 {
  unsigned int u;
  struct {
    unsigned int expo:2;
    unsigned int mant:30;
  } __attribute__((packed));
  // The packed attribute is acceptable because it defines a less strict
  // alignment than required by the first field of the transparent union.
} __attribute__((transparent_union));

union pr15134v2 {
  struct { // expected-note {{alignment of first field is 32 bits}}
    unsigned int u1;
    unsigned int u2;
  };
  struct {  // expected-warning {{alignment of field '' (64 bits) does not match the alignment of the first field in transparent union; transparent_union attribute ignored}}
    unsigned int u3;
  } __attribute__((aligned(8)));
} __attribute__((transparent_union));

union pr30520v { void b; } __attribute__((transparent_union)); // expected-error {{field has incomplete type 'void'}}

union pr30520a { int b[]; } __attribute__((transparent_union)); // expected-error {{field has incomplete type 'int []'}}

// expected-note@+1 2 {{forward declaration of 'struct stb'}}
union pr30520s { struct stb b; } __attribute__((transparent_union)); // expected-error {{field has incomplete type 'struct stb'}}

union pr30520s2 { int *v; struct stb b; } __attribute__((transparent_union)); // expected-error {{field has incomplete type 'struct stb'}}

typedef union __attribute__((__transparent_union__)) {
  int *i;
  struct st *s;
} TU6;

void bar(TU6);

void foo11(int *i) {
  bar(i);
}
void foo2(struct st *s) {
  bar(s);
}
