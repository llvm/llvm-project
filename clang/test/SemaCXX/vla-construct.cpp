// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -O0 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -pedantic-errors -DPE -O0 -verify %s

# ifndef PE
// expected-no-diagnostics
# endif

extern "C" int printf(const char*, ...);

static int N;
struct S {
  S() __attribute__ ((nothrow))  { printf("%d: S()\n", ++N); }
  ~S()  __attribute__ ((nothrow))  { printf("%d: ~S()\n", N--); }
  int n[17];
};

void print(int n, int a, int b, int c, int d) {
  printf("n=%d\n,sizeof(S)=%d\nsizeof(array_t[0][0])=%d\nsizeof(array_t[0])=%d\nsizeof(array_t)=%d\n",
         n, a, b, c, d);
  if (n == 2) throw(n);
}

void test(int n) {
  S array_t[n][n+1];
# ifdef PE
   // expected-error@-2 {{variable length arrays are a C99 feature}} expected-note@-2 {{read of non-const}} expected-note@-3 {{here}}
   // expected-error@-3 {{variable length arrays are a C99 feature}} expected-note@-3 {{read of non-const}} expected-note@-4 {{here}}
# endif
  int sizeof_S = sizeof(S);
  int sizeof_array_t_0_0 = sizeof(array_t[0][0]);
  int sizeof_array_t_0 = sizeof(array_t[0]);
  int sizeof_array_t = sizeof(array_t);
  print(n, sizeof_S, sizeof_array_t_0_0, sizeof_array_t_0, sizeof_array_t);
}

int main()
{
  try {
    test(2);
  } catch(int e) {
    printf("exception %d\n", e);
  }
  try {
    test(3);
  } catch(int e) {
    printf("exception %d", e);
  }
}
