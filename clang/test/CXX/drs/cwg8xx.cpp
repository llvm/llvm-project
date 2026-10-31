// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,cxx98-17
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,cxx98-17,since-cxx11
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,cxx98-17,since-cxx11
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,cxx98-17,since-cxx11
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx20,since-cxx11
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx20,since-cxx11
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx20,since-cxx11

namespace cwg812 { // cwg812: 2.8
#if __cplusplus >= 201103L
namespace Q {
  inline namespace V1 {
    int i = 1; // #cwg812-v1-i
    int j = 1; // #cwg812-v1-j
  }
  inline namespace V2 {
    int j = 2; // #cwg812-v2-j
  }
  int i = 0;   // #cwg812-Q-i
}

static_assert(Q::i, "");
// since-cxx11-error@-1 {{reference to 'i' is ambiguous}}
//   since-cxx11-note@#cwg812-v1-i {{candidate found by name lookup is 'cwg812::Q::V1::i'}}
//   since-cxx11-note@#cwg812-Q-i  {{candidate found by name lookup is 'cwg812::Q::i'}}
static_assert(Q::j, "");
// since-cxx11-error@-1 {{reference to 'j' is ambiguous}}
//   since-cxx11-note@#cwg812-v1-j {{candidate found by name lookup is 'cwg812::Q::V1::j'}}
//   since-cxx11-note@#cwg812-v2-j {{candidate found by name lookup is 'cwg812::Q::V2::j'}}
#endif
} // namespace cwg812

namespace cwg820 { // cwg820: 2.7
export template <class T> struct B {};
// cxx98-17-warning@-1 {{exported templates are unsupported}}
// since-cxx20-error@-2 {{export declaration can only be used within a module interface}}
export template<typename T> void f() {}
// cxx98-17-warning@-1 {{exported templates are unsupported}}
// since-cxx20-error@-2 {{export declaration can only be used within a module interface}}
} // namespace cwg820

namespace cwg861 { // cwg861: 2.7
#if __cplusplus >= 201103L
namespace A {
  inline namespace B {
    namespace C {
      constexpr int i = 0;
    }
    using namespace C;
  }
  constexpr int i = 1;
}

static_assert(A::i == 1, "");
#endif
} // namespace cwg861

namespace cwg873 { // cwg873: 3.0
#if __cplusplus >= 201103L
template <typename T> void f(T &&);
template <> void f(int &) = delete;  // #cwg873-lvalue-ref
template <> void f(int &&) = delete; // #cwg873-rvalue-ref
void g(int i) {
  f(i); // calls f<int&>(int&)
  // since-cxx11-error@-1 {{call to deleted function 'f'}}
  //   since-cxx11-note@#cwg873-lvalue-ref {{candidate function [with T = int &] has been explicitly deleted}}
  f(0); // calls f<int>(int&&)
  // since-cxx11-error@-1 {{call to deleted function 'f'}}
  //   since-cxx11-note@#cwg873-rvalue-ref {{candidate function [with T = int] has been explicitly deleted}}
}
#endif
} // namespace cwg873

// cwg882: 3.5
#if __cplusplus >= 201103L
int main() = delete;
// since-cxx11-error@-1 {{'main' is not allowed to be deleted}}
#endif
