// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s

// This recursive function is not called from sycl kernel,
// so it should not be diagnosed.
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n-1) + fib(n-2);
}

void kernel3(void) {
  ;
}

using myFuncDef = int(int,int);

void usage3(myFuncDef functionPtr) {
  kernel3();
}

int addInt(int n, int m) {
    return n+m;
}

template <typename name, typename Func>
  // expected-note@+1 2{{function implemented using recursion declared here}}
__attribute__((sycl_kernel)) void kernel_single_task2(Func kernelFunc) {
  kernelFunc();
  // expected-error@+1 2{{SYCL kernel cannot allocate storage}}
  int *ip = new int;
  // expected-error@+1 2{{SYCL kernel cannot call a recursive function}}
  kernel_single_task2<name, Func>(kernelFunc);
}

int main() {
  kernel_single_task2<class fake_kernel>([]() { usage3(  &addInt ); });
  return fib(5);
}
