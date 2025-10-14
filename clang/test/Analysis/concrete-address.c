// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -verify %s

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));

#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

typedef unsigned long long uintptr_t;

void f0(void) {
  int *p = (int*) 0x10000; // Should not crash here.
  *p = 3; // expected-warning{{Dereference of a fixed address}}
}

void f1(int *p) {
  if (p != (int *)-1)
    *p = 1;
  else
    *p = 0; // expected-warning{{Dereference of a fixed address}}
}

struct f2_struct {
  int x;
};

int f2(struct f2_struct* p) {

  if (p != (struct f2_struct *)1)
    p->x = 1;

  return p->x++; // expected-warning{{Access to field 'x' results in a dereference of a fixed address (loaded from variable 'p')}}
}

int f3_1(char* x) {
  int i = 2;

  if (x != (char *)1)
    return x[i - 1];

  return x[i+1]; // expected-warning{{Array access (from variable 'x') results in a dereference of a fixed address}}
}

int f3_2(char* x) {
  int i = 2;

  if (x != (char *)1)
    return x[i - 1];

  return x[i+1]++; // expected-warning{{Array access (from variable 'x') results in a dereference of a fixed address}}
}

int f4_1(int *p) {
  uintptr_t x = (uintptr_t) p;

  if (x != (uintptr_t)1)
    return 1;

  int *q = (int*) x;
  return *q; // expected-warning{{Dereference of a fixed address (loaded from variable 'q')}}
}

int f4_2(void) {
  short array[2];
  uintptr_t x = (uintptr_t)array;
  short *p = (short *)x;

  // The following branch should be infeasible.
  if (!(p == &array[0])) {
    p = (short *)1;
    *p = 1; // no-warning
  }

  if (p != (short *)1) {
    *p = 5; // no-warning
    p = (short *)1; // expected-warning {{Using a fixed address is not portable}}
  }
  else return 1;

  *p += 10; // expected-warning{{Dereference of a fixed}}
  return 0;
}

int f5(void) {
  char *s = "hello world";
  return s[0]; // no-warning
}

void f6(int *p, int *q) {
  if (p != (int *)1)
    if (p == (int *)1)
      *p = 1; // no-warning

  if (q == (int *)1)
    if (q != (int *)1)
      *q = 1; // no-warning
}

int* qux(int);

int f7_1(unsigned len) {
  assert (len != 0);
  int *p = (int *)1;
  unsigned i;

  for (i = 0; i < len; ++i)
   p = qux(i);

  return *p++; // no-warning
}

int f7_2(unsigned len) {
  assert (len > 0);  // note use of '>'
  int *p = (int *)1;
  unsigned i;

  for (i = 0; i < len; ++i)
   p = qux(i);

  return *p++; // no-warning
}

struct f8_s {
  int x;
  int y[2];
};

void f8(struct f8_s *s, int coin) {
  if (s != (struct f8_s *)7)
    return;

  if (coin)
    s->x = 5; // expected-warning{{Access to field 'x' results in a dereference of a fixed address (loaded from variable 's')}}
  else
    s->y[1] = 6; // expected-warning{{Array access (via field 'y') results in a dereference of a fixed address}}
}

void f9() {
  int (*p_function) (char, char) = (int (*)(char, char))0x04040; // FIXME: warn at this initialization
  p_function = (int (*)(char, char))0x04080; // expected-warning {{Using a fixed address is not portable}}
  // FIXME: there should be a warning from calling the function pointer with fixed address
  int x = (*p_function) ('x', 'y');
}
