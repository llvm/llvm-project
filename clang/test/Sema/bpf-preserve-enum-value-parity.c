// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=expected,attr %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=expected,explicit -DEXPLICIT %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=expected,attr -std=c23 %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=expected,explicit -std=c23 -DEXPLICIT %s
// RUN: %clang_cc1 -triple bpf -fsyntax-only -verify=expected,attr %s
// RUN: %clang_cc1 -triple bpf -fsyntax-only -verify=expected,explicit -DEXPLICIT %s

// An enumerator of an enum with the preserve_access_index attribute is
// accepted and rejected where __builtin_preserve_enum_value() is. Where a
// constant is required, it is kept a constant with a single error naming the
// attribute (attr-error) instead of the error for the call (explicit-error).
// A builtin argument error gets an extra note naming the attribute (attr-note).
#ifdef EXPLICIT
enum E { A = 4 };
#define X __builtin_preserve_enum_value(*(enum E *)A, 1)
#else
enum E { A = 4 } __attribute__((preserve_access_index));
#define X A
#endif

typedef unsigned long size_t;
void *memcpy(void *, const void *, size_t);

// Rejected where a compile-time constant is required.
int g1 = X; // explicit-error {{initializer element is not a compile-time constant}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
int *g3 = (int[]){X}; // explicit-error {{initializer element is not a compile-time constant}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
int g4[X]; // explicit-error {{variable length array declaration not allowed at file scope}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
int g5[8] = {[X] = 1}; // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
_Alignas(X) int g6; // explicit-error {{'aligned' attribute requires integer constant}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
typedef int v4 __attribute__((vector_size(X * 4))); // explicit-error {{vector_size attribute requires an integer constant}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
struct s1 {
  int a[X]; // explicit-error {{fields must have a constant size: 'variable length array in structure' extension will never be supported}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
  int b : X; // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
};
enum F { F1 = X }; // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
_Static_assert(X == 4, ""); // explicit-error {{static assertion expression is not an integral constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}

void h(int);
int rejected(int v, void *p) {
  static int s = X; // explicit-error {{initializer element is not a compile-time constant}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
  int a[8] = {[X] = 1}; // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
  switch (v) {
  case X: // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
    return 1;
  case 5 ... X + 2: // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
    return 2;
  }
#pragma unroll X // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
  for (int i = 0; i < 8; i++)
    h(i);
#pragma clang loop unroll_count(X) // explicit-error {{expression is not an integer constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
  for (int i = 0; i < 8; i++)
    h(i);
  __builtin_prefetch(p, X - 4); // expected-error {{argument to '__builtin_prefetch' must be a constant integer}} attr-note {{enumerator 'A' is not a constant expression since 'enum E' has the 'preserve_access_index' attribute}}
  return v + a[0] + s;
}

#if __STDC_VERSION__ >= 202311L
struct item { int v; };
constexpr int cx2 = X; // explicit-error {{constexpr variable 'cx2' must be initialized by a constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
int constexpr_locals(void) {
  constexpr struct item cx3 = {X}; // explicit-error {{constexpr variable 'cx3' must be initialized by a constant expression}} attr-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
  return cx3.v;
}
#endif

// Accepted where the value is used at runtime, or not used at all.
struct key { int a; int b; };
void use(void *);
unsigned long accepted(int v, int *p, void *d, const void *s) {
  const int k = X;
  struct key kk = {X, 1};
  int arr[2] = {X, 1};
  use(&kk);
  use(arr);
  switch (X) {
  case 4:
    v++;
  }
  v += p[X] + (v == X) + (v && X ? X : 0) + (v << X) + k;
  v += __builtin_bswap32(X) + __builtin_expect(v, X);
  v += __builtin_constant_p(X) + sizeof(X) + _Generic(X, int: 1, default: 2);
  memcpy(d, s, X);
  __builtin___memcpy_chk(d, s, X, 64);
  return v + __builtin_align_up((unsigned long)v, X);
}
