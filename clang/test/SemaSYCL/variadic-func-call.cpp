// RUN: %clang_cc1 -isystem %S/Inputs/ -fsycl-is-device -triple spirv64 -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s
// RUN: %clang_cc1 -isystem %S/Inputs/ -fsycl-is-device -triple spirv64 -fsyntax-only -verify %s

template<typename KN, typename...Args>
void sycl_kernel_launch(Args ...args) {}

template<typename KN, typename K>
[[clang::sycl_kernel_entry_point(KN)]]
void sycl_entry_point(K k) {
  k(); // expected-note 2{{called by}}
}

void variadic(int, ...) {}
namespace NS {
void variadic(int, ...) {}
}

struct S {
  S(int, ...) {}
  void operator()(int, ...) {}
};

void foo() {
  auto x = [](int, ...) {};
  x(5, 10); //expected-error{{SYCL device code does not support variadic functions}}
}

void overloaded(int, int) {}
void overloaded(int, ...) {}

int main() {
  sycl_entry_point<class FK>([]() {
    variadic(5);        //expected-error{{SYCL device code does not support variadic functions}}
    variadic(5, 2);     //expected-error{{SYCL device code does not support variadic functions}}
    NS::variadic(5, 3); //expected-error{{SYCL device code does not support variadic functions}}
    S s(5, 4);          //expected-error{{SYCL device code does not support variadic functions}}
    S s2(5);            //expected-error{{SYCL device code does not support variadic functions}}
    s(5, 5);            //expected-error{{SYCL device code does not support variadic functions}}
    s2(5);              //expected-error{{SYCL device code does not support variadic functions}}
    foo();              //expected-note{{called by 'operator()'}}
    overloaded(5, 6);   //expected-no-error
    overloaded(5, s);   //expected-error{{SYCL device code does not support variadic functions}}
    overloaded(5);      //expected-error{{SYCL device code does not support variadic functions}}
  });
}
