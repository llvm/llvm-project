// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify -std=c++98 %s
// RUN: %clang_cc1 -verify -std=c++11 %s

class A {
public:
  explicit A();
  
  explicit operator int();
#if __cplusplus <= 199711L // C++03 or earlier modes
  // expected-warning@-2 {{explicit conversion functions are a C++11 extension}}
#endif

  explicit void f0(); // expected-error {{'explicit' can only be applied to a constructor or conversion function}}
  
  operator bool();
};

explicit A::A() { } // expected-error {{'explicit' can only be specified inside the class definition}}
explicit A::operator bool() { return false; }
#if __cplusplus <= 199711L // C++03 or earlier modes
// expected-warning@-2 {{explicit conversion functions are a C++11 extension}}
#endif
// expected-error@-4 {{'explicit' can only be specified inside the class definition}}

class B {
  friend explicit A::A(); // expected-error {{'explicit' is invalid in friend declarations}}
};
