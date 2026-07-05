// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -O0 -verify -Wno-vla %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -pedantic-errors -O0 -verify=pe %s

// expected-no-diagnostics

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
  S array_t[n][n+1]; // pe-error 2{{variable length arrays in C++ are a Clang extension}} pe-note 2{{parameter}} pe-note@-1 2{{here}}
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
