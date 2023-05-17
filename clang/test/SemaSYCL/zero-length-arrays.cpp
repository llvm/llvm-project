// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -fsyntax-only -verify -std=c++14 %s
//
// This test checks if compiler reports compilation error on an attempt to use
// a zero-length array inside device code.

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  // expected-note@+1 5{{called by 'kernel}}
  kernelFunc(); // #KernelObjCall
}

typedef float ZEROARR[0];

struct Wrapper {
  int A;
  int BadArray[0]; // expected-note 3{{field of illegal type 'int[0]' declared here}}
};

struct WrapperOfWrapper { // expected-error 2{{zero-length arrays are not permitted in SYCL device code}}
  Wrapper F;              // expected-note 2{{within field of type 'Wrapper' declared here}}
  ZEROARR *Ptr;           //expected-note 5{{field of illegal pointer type 'ZEROARR *' (aka 'float (*)[0]') declared here}}
};

template <unsigned Size> struct InnerTemplated {
  double Array[Size]; // expected-note 8{{field of illegal type 'double[0]' declared here}}
};

template <unsigned Size, typename Ty> struct Templated {
  unsigned A;
  Ty Arr[Size];
  InnerTemplated<Size> Array[Size + 1]; // expected-note 8{{within field of type 'InnerTemplated<0U>[1]' declared here}}
};

struct KernelSt {
  int A;
  int BadArray[0]; // expected-note {{field of illegal type 'int[0]' declared here}}
  void operator()() const {}
};

WrapperOfWrapper offendingFoo() {
  // expected-note@+1 {{called by 'offendingFoo'}}
  return WrapperOfWrapper{};
}

template <unsigned Size>
void templatedContext() {
  Templated<Size, float> Var;
  // expected-error@#KernelObjCall 2{{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@#KernelObjCall {{called by 'kernel<TempContext, (lambda at}}
  // expected-note@+1 {{in instantiation of function template specialization}}
  kernel<class TempContext>([=] {
    // expected-note@+1 {{within field of type 'Templated<0U, float>' declared here}}
    (void)Var; // expected-error 2{{zero-length arrays are not permitted in SYCL device code}}
  });
  // expected-error@#KernelObjCall {{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  // expected-note@+1 {{within field of type 'Templated<0U, float>' declared here}}
  kernel<class TempContext1>([Var] {
  });
}

void foo(const unsigned X) {
  int Arr[0];      // expected-note 2{{declared here}}
  ZEROARR TypeDef; // expected-note {{declared here}}
  ZEROARR *Ptr;    // expected-note {{declared here}}
                   // expected-error@#KernelObjCall 3{{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+1 {{in instantiation of function template specialization}}
  kernel<class Simple>([=]() {
    (void)Arr;     // expected-error {{zero-length arrays are not permitted in SYCL device code}}
    (void)TypeDef; // expected-error {{zero-length arrays are not permitted in SYCL device code}}
    // expected-note@+1 {{field of illegal pointer type 'ZEROARR *' (aka 'float (*)[0]') declared here}}
    (void)Ptr; // expected-error {{zero-length arrays are not permitted in SYCL device code}}
  });
  // expected-error@#KernelObjCall {{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  // expected-note@+1 {{field of illegal type 'int[0]' declared here}}
  kernel<class Simple1>([Arr] { // expected-error {{zero-length arrays are not permitted in SYCL device code}}
  });
  WrapperOfWrapper St;
  // expected-error@#KernelObjCall 2{{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+1 {{in instantiation of function template specialization}}
  kernel<class SimpleStruct>([=] {
    // expected-note@+1 {{within field of type 'WrapperOfWrapper' declared here}}
    (void)St.F.BadArray; // expected-error 4{{zero-length arrays are not permitted in SYCL device code}}
  });
  // expected-error@#KernelObjCall 2{{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  // expected-note@+1 {{within field of type 'WrapperOfWrapper' declared here}}
  kernel<class SimpleStruct1>([St] { // expected-error 2{{zero-length arrays are not permitted in SYCL device code}}
  });

  Templated<1, int> OK;
  Templated<1 - 1, double> Weirdo;
  Templated<0, float> Zero;
  // expected-error@#KernelObjCall 4{{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+1 {{in instantiation of function template specialization}}
  kernel<class UseTemplated>([=] {
    (void)OK;   // No errors expected
    (void)Zero; // expected-error 2{{zero-length arrays are not permitted in SYCL device code}}
    // expected-note@+1 {{within field of type 'Templated<1 - 1, double>' declared here}}
    int A = Weirdo.A; // expected-error 2{{zero-length arrays are not permitted in SYCL device code}}
  });

  // expected-note@#KernelObjCall {{called by 'kernel<UseTemplated1, (lambda at}}
  // expected-error@#KernelObjCall 2{{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  // expected-note@+1 {{within field of type 'Templated<0, float>' declared here}}
  kernel<class UseTemplated1>([Zero] { // expected-error 2{{zero-length arrays are not permitted in SYCL device code}}
  });

  templatedContext<10>();
  // expected-note@+1 2{{in instantiation of function template specialization}}
  templatedContext<0>();

  KernelSt K;
  // expected-error@#KernelObjCall {{zero-length arrays are not permitted in SYCL device code}}
  // expected-note@+1 {{in instantiation of function template specialization}}
  kernel<class UseFunctor>(K);

  // expected-note@#KernelObjCall {{called by 'kernel<ReturnFromFunc, (lambda at}}
  kernel<class ReturnFromFunc>([=] {
    // expected-note@+1 {{called by 'operator()'}}
    offendingFoo();
  });
}
