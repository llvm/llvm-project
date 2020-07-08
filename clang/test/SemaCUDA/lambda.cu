// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=com %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fcuda-is-device -verify=com,dev %s

#include "Inputs/cuda.h"

auto global_lambda = [] () { return 123; };

template<class F>
__global__ void kernel(F f) { f(); }
// dev-note@-1 7{{called by 'kernel<(lambda}}

__host__ __device__ void hd(int x);

class A {
  int b;
public:
  void test() {
    [=](){ hd(b); }();

    [&](){ hd(b); }();

    kernel<<<1,1>>>([](){ hd(0); });

    kernel<<<1,1>>>([=](){ hd(b); });
    // dev-error@-1 {{capture host side class data member by this pointer in device or host device lambda function}}

    kernel<<<1,1>>>([&](){ hd(b); });
    // dev-error@-1 {{capture host side class data member by this pointer in device or host device lambda function}}

    kernel<<<1,1>>>([&] __device__ (){ hd(b); });
    // dev-error@-1 {{capture host side class data member by this pointer in device or host device lambda function}}

    kernel<<<1,1>>>([&](){
      auto f = [&]{ hd(b); };
      // dev-error@-1 {{capture host side class data member by this pointer in device or host device lambda function}}
      f();
    });
  }
};

int main(void) {
  auto lambda_kernel = [&]__global__(){};
  // com-error@-1 {{kernel function 'operator()' must be a free function or static member function}}

  int b;
  [&](){ hd(b); }();

  [=, &b](){ hd(b); }();

  kernel<<<1,1>>>(global_lambda);

  kernel<<<1,1>>>([](){ hd(0); });

  kernel<<<1,1>>>([=](){ hd(b); });

  kernel<<<1,1>>>([b](){ hd(b); });

  kernel<<<1,1>>>([&](){ hd(b); });
  // dev-error@-1 {{capture host variable 'b' by reference in device or host device lambda function}}

  kernel<<<1,1>>>([=, &b](){ hd(b); });
  // dev-error@-1 {{capture host variable 'b' by reference in device or host device lambda function}}

  kernel<<<1,1>>>([&, b](){ hd(b); });

  kernel<<<1,1>>>([&](){
      auto f = [&]{ hd(b); };
      // dev-error@-1 {{capture host variable 'b' by reference in device or host device lambda function}}
      f();
  });

  return 0;
}
