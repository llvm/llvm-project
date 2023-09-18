// RUN: %clang_cc1 -Wno-pointer-to-int-cast -fsyntax-only -verify %s

#define offsetof(TYPE, MEMBER) __builtin_offsetof (TYPE, MEMBER)

typedef struct P { int i; float f; } PT;
struct external_sun3_core
{
 unsigned c_regs;

  PT  X[100];

};

// Ensure the builtin works as a constant expression
int i = offsetof(PT, f);

void swap(void)
{
  int x;
  x = offsetof(struct external_sun3_core, c_regs);
  x = __builtin_offsetof(struct external_sun3_core, X[42].f);

  x = __builtin_offsetof(struct external_sun3_core, X[42].f2);  // expected-error {{no member named 'f2'}}
  x = __builtin_offsetof(int, X[42].f2);  // expected-error {{offsetof requires struct}}

  int a[__builtin_offsetof(struct external_sun3_core, X) == 4 ? 1 : -1];
  int b[__builtin_offsetof(struct external_sun3_core, X[42]) == 340 ? 1 : -1];
  int c[__builtin_offsetof(struct external_sun3_core, X[42].f2) == 344 ? 1 : -1];  // expected-error {{no member named 'f2'}}
}

extern int f(void);

struct s1 { int a; };
int v1 = offsetof (struct s1, a) == 0 ? 0 : f();

struct s2 { int a; };
int v2 = (int)(&((struct s2 *) 0)->a) == 0 ? 0 : f();

struct s3 { int a; };
int v3 = __builtin_offsetof(struct s3, a) == 0 ? 0 : f();

// PR3396
struct sockaddr_un {
 unsigned char sun_len;
 char sun_path[104];
};
int a(int len) {
int a[__builtin_offsetof(struct sockaddr_un, sun_path[len+1])];
}

// PR4079
union x {struct {int x;};};
int x[__builtin_offsetof(union x, x)];

struct incomplete; // expected-note 2 {{forward declaration of 'struct incomplete'}}
int test1[__builtin_offsetof(struct incomplete, foo)]; // expected-error {{offsetof of incomplete type 'struct incomplete'}}

int test2[__builtin_offsetof(struct incomplete[10], [4].foo)]; // expected-error {{array has incomplete element type 'struct incomplete'}}

// Bitfields
struct has_bitfields {
  int i : 7;
  int j : 12; // expected-note{{bit-field is declared here}}
};

int test3 = __builtin_offsetof(struct has_bitfields, j); // expected-error{{cannot compute offset of bit-field 'j'}}

typedef struct Array { int array[1]; } Array;
int test4 = __builtin_offsetof(Array, array);

int test5(void) {
  return __builtin_offsetof(Array, array[*(int*)0]); // expected-warning{{indirection of non-volatile null pointer}} expected-note{{__builtin_trap}}
}

