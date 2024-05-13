// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++1z -verify %s

class A {
public:
  static const char X;
};
const char A::X = 0;

template<typename U> void func() noexcept(U::X);

template<class... B, char x>
void foo(void(B...) noexcept(x)) {} // expected-note{{candidate template ignored}}

void bar()
{
  foo(func<A>);	// expected-error{{no matching function for call}}
}
