// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s

// This recursive function is not called from sycl kernel,
// so it should not be diagnosed.
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n-1) + fib(n-2);
}

  // expected-note@+1 2{{function implemented using recursion declared here}}
void kernel2(void) {
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  kernel2();
}

using myFuncDef = int(int,int);

void usage2(myFuncDef functionPtr) {
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  kernel2();
}

int addInt(int n, int m) {
  return n+m;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *ip = new int;
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() {usage2(&addInt);});
  return fib(5);
}
