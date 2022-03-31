// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++1z -verify %s
#if __cplusplus >= 201703
// expected-no-diagnostics
#endif
class A {
public:
  static const char X;
};
const char A::X = 0;

template<typename U> void func() noexcept(U::X);

template<class... B, char x>
#if __cplusplus >= 201703
void foo(void(B...) noexcept(x)) {} 
#else
void foo(void(B...) noexcept(x)) {} // expected-note{{candidate template ignored}}
#endif

void bar()
{
#if __cplusplus >= 201703
  foo(func<A>);
#else
  foo(func<A>);	// expected-error{{no matching function for call}}
#endif	
}


