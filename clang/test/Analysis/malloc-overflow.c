// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.security.MallocOverflow -verify %s

#define NULL ((void *) 0)
typedef __typeof__(sizeof(int)) size_t;
extern void * malloc(size_t);

void * f1(int n)
{
  return malloc(n * sizeof(int));  // expected-warning {{the computation of the size of the memory allocation may overflow}}
}

void * f2(int n)
{
  return malloc(sizeof(int) * n); // // expected-warning {{the computation of the size of the memory allocation may overflow}}
}

void * f3(void)
{
  return malloc(4 * sizeof(int));  // no-warning
}

struct s4
{
  int n;
};

void * f4(struct s4 *s)
{
  return malloc(s->n * sizeof(int)); // expected-warning {{the computation of the size of the memory allocation may overflow}}
}

void * f5(struct s4 *s)
{
  struct s4 s2 = *s;
  return malloc(s2.n * sizeof(int)); // expected-warning {{the computation of the size of the memory allocation may overflow}}
}

void * f6(int n)
{
  return malloc((n + 1) * sizeof(int)); // expected-warning {{the computation of the size of the memory allocation may overflow}}
}

extern void * malloc (size_t);

void * f7(int n)
{
  if (n > 10)
    return NULL;
  return malloc(n * sizeof(int));  // no-warning
}

void * f8(int n)
{
  if (n < 10)
    return malloc(n * sizeof(int));  // no-warning
  else
    return NULL;
}

void * f9(int n)
{
  int * x = malloc(n * sizeof(int));  // expected-warning {{the computation of the size of the memory allocation may overflow}}
  for (int i = 0; i < n; i++)
    x[i] = i;
  return x;
}

void * f10(int n)
{
  int * x = malloc(n * sizeof(int));  // expected-warning {{the computation of the size of the memory allocation may overflow}}
  int i = 0;
  while (i < n)
    x[i++] = 0;
  return x;
}

void * f11(int n)
{
  int * x = malloc(n * sizeof(int));  // expected-warning {{the computation of the size of the memory allocation may overflow}}
  int i = 0;
  do {
    x[i++] = 0;
  } while (i < n);
  return x;
}

void * f12(int n)
{
  n = (n > 10 ? 10 : n);
  int * x = malloc(n * sizeof(int));  // no-warning
  for (int i = 0; i < n; i++)
    x[i] = i;
  return x;
}

struct s13
{
  int n;
};

void * f13(struct s13 *s)
{
  if (s->n > 10)
    return NULL;
  return malloc(s->n * sizeof(int)); // no-warning
}

void * f14(int n)
{
  if (n < 0)
    return NULL;
  return malloc(n * sizeof(int));  // expected-warning {{the computation of the size of the memory allocation may overflow}}
}

void *check_before_malloc(int n, int x) {
  int *p = NULL;
  if (n > 10)
    return NULL;
  if (x == 42)
    p = malloc(n * sizeof(int)); // no-warning, the check precedes the allocation

  // Do some other stuff, e.g. initialize the memory.
  return p;
}

void *check_after_malloc(int n, int x) {
  int *p = NULL;
  if (x == 42)
    p = malloc(n * sizeof(int)); // expected-warning {{the computation of the size of the memory allocation may overflow}}

  // The check is after the allocation!
  if (n > 10) {
    // Do something conditionally.
  }
  return p;
}

#define GREATER_THAN(lhs, rhs) (lhs > rhs)
void *check_after_malloc_using_macros(int n, int x) {
  int *p = NULL;
  if (x == 42)
    p = malloc(n * sizeof(int)); // expected-warning {{the computation of the size of the memory allocation may overflow}}

  if (GREATER_THAN(n, 10))
    return NULL;

  // Do some other stuff, e.g. initialize the memory.
  return p;
}
#undef GREATER_THAN
