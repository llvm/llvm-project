// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s

// This recursive function is not called from sycl kernel,
// so it should not be diagnosed.
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n-1) + fib(n-2);
}

typedef struct S {
template <typename T>
  // expected-note@+1 3{{function implemented using recursion declared here}}
T factT(T i, T j)
{
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  return factT(j,i);
}

int fact(unsigned i)
{
  if (i==0) return 1;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  else return factT<unsigned>(i-1, i);
}
} S_type;


  // expected-note@+1 2{{function implemented using recursion declared here}}
int fact(unsigned i);
  // expected-note@+1 2{{function implemented using recursion declared here}}
int fact1(unsigned i)
{
  if (i==0) return 1;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  else return fact(i-1) * i;
}
int fact(unsigned i)
{
  if (i==0) return 1;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  else return fact1(i-1) * i;
}

bool isa_B(void) {
  S_type s;

  unsigned f = s.fact(3);
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  unsigned f1 = s.factT<unsigned>(3,4);
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  unsigned g = fact(3);
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  unsigned g1 = fact1(3);
  return 0;
}

void kernel1(void) {
  isa_B();
}

using myFuncDef = int(int,int);

void usage(myFuncDef functionPtr) {
  kernel1();
}

int addInt(int n, int m) {
    return n+m;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() {usage(&addInt);});
  return fib(5);
}
