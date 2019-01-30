// RUN: %clang -I %S/Inputs -std=c++11 --sycl -Xclang -fsycl-int-header=%t.h %s -c -o %t.spv
// RUN: FileCheck %s --input-file=%t.h

// Checks that functors are supported as SYCL kernels.

#include "sycl.hpp"

constexpr auto sycl_read_write = cl::sycl::access::mode::read_write;
constexpr auto sycl_global_buffer = cl::sycl::access::target::global_buffer;

// Case 1:
// - functor class is defined in an anonymous namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
//   * has no 'const' qualifier
namespace {
  class Functor1 {
  public:
    Functor1(int X_, cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
      X(X_), Acc(Acc_)
    {}

    void operator()() {
      Acc.use(X);
    }

  private:
    int X;
    cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> Acc;
  };
}

// Case 2:
// - functor class is defined in a namespace
// - the '()' operator:
//   * does not have parameters (to be used in 'single_task').
//   * has the 'const' qualifier
namespace ns {
  class Functor2 {
  public:
    Functor2(int X_, cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
      X(X_), Acc(Acc_)
    {}

    void operator()() const {
      Acc.use(X);
    }

  private:
    int X;
    cl::sycl::accessor<int, 1, sycl_read_write, sycl_global_buffer> Acc;
  };
}

// Case 3:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type cl::sycl::id<1> (to be used in 'parallel_for').
//   * has no 'const' qualifier
template <typename T> class TmplFunctor {
public:
  TmplFunctor(T X_, cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
    X(X_), Acc(Acc_)
  {}

  void operator()(cl::sycl::id<1> id) {
    Acc.use(id, X);
  }

private:
  T X;
  cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};

// Case 4:
// - functor class is templated and defined in the translation unit scope
// - the '()' operator:
//   * has a parameter of type cl::sycl::id<1> (to be used in 'parallel_for').
//   * has the 'const' qualifier
template <typename T> class TmplConstFunctor {
public:
  TmplConstFunctor(T X_, cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> &Acc_) :
    X(X_), Acc(Acc_)
  {}

  void operator()(cl::sycl::id<1> id) const {
    Acc.use(id, X);
  }

private:
  T X;
  cl::sycl::accessor<T, 1, sycl_read_write, sycl_global_buffer> Acc;
};

// Exercise non-templated functors in 'single_task'.
int foo(int X) {
  int A[] = { 10 };
  {
    cl::sycl::queue Q;
    cl::sycl::buffer<int, 1> Buf(A, 1);

    Q.submit([&](cl::sycl::handler& cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      Functor1 F(X, Acc);

      cgh.single_task(F);
    });
    Q.submit([&](cl::sycl::handler& cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
    Q.submit([&](cl::sycl::handler& cgh) {
      auto Acc = Buf.get_access<sycl_read_write, sycl_global_buffer>(cgh);
      ns::Functor2 F(X, Acc);

      cgh.single_task(F);
    });
  }
  return A[0];
}

#define ARR_LEN(x) sizeof(x)/sizeof(x[0])

// Exercise templated functors in 'parallel_for'.
template <typename T> T bar(T X) {
  T A[] = { (T)10, (T)10 };
  {
    cl::sycl::queue Q;
    cl::sycl::buffer<T, 1> Buf(A, ARR_LEN(A));

    Q.submit([&](cl::sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      TmplFunctor<T> F(X, Acc);

      cgh.parallel_for(cl::sycl::range<1>(ARR_LEN(A)), F);
    });
    // Spice with lambdas to make sure functors and lambdas work together.
    Q.submit([&](cl::sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      cgh.parallel_for<class LambdaKernel>(
        cl::sycl::range<1>(ARR_LEN(A)),
        [=](cl::sycl::id<1> id) { Acc.use(id, X); });
    });
    Q.submit([&](cl::sycl::handler& cgh) {
      auto Acc = Buf.template get_access<sycl_read_write, sycl_global_buffer>(cgh);
      TmplConstFunctor<T> F(X, Acc);

      cgh.parallel_for(cl::sycl::range<1>(ARR_LEN(A)), F);
    });
  }
  T res = (T)0;

  for (int i = 0; i < ARR_LEN(A); i++) {
    res += A[i];
  }
  return res;
}

int main() {
  const int Res1 = foo(10);
  const int Res2 = bar(10);
  const int Gold1 = 40;
  const int Gold2 = 80;

#ifndef __SYCL_DEVICE_ONLY__
  cl::sycl::detail::KernelInfo<Functor1>::getName();
  // CHECK: Functor1
  cl::sycl::detail::KernelInfo<ns::Functor2>::getName();
  // CHECK: ::ns::Functor2
  cl::sycl::detail::KernelInfo<TmplFunctor<int>>::getName();
  // CHECK: TmplFunctor<int>
  cl::sycl::detail::KernelInfo<TmplConstFunctor<int>>::getName();
  // CHECK: TmplConstFunctor<int>
#endif // __SYCL_DEVICE_ONLY__

  return 0;
}

