// RUN: %clang_cc1 -std=c++11 -fmodules %s -verify

#pragma clang module build M
  module M { module A {} module B {} module C {} }
#pragma clang module contents
  
  #pragma clang module begin M.A
    template<typename U> struct X {
      template<typename T> void f();
    };
  #pragma clang module end
  
  #pragma clang module begin M.B
    template<typename T, typename U = void> struct ST { static void f(); };
  #pragma clang module end
  
  #pragma clang module begin M.C
    template<typename U> struct X;
    void foo(X<int>);
  #pragma clang module end
#pragma clang module endbuild

#pragma clang module build N
  module N {}
#pragma clang module contents
  #pragma clang module begin N
    #pragma clang module import M.B // not re-exported

    template<typename U> struct X {
      template<typename T> void f();
      template<typename T> void g();
    };

    template<typename U> template<typename T>
    void X<U>::f() {
      ST<T>::f(); // definition and default argument found in M.B
      foo(*this); // found by ADL in M.C
    };

    #pragma clang module import M.C // not re-exported
  #pragma clang module end
#pragma clang module endbuild

#pragma clang module import N
void g() {
  X<int>().f<int>();

  ST<int>::f(); // expected-error {{must be imported from module 'M.B'}}
  foo(X<int>()); // expected-error {{must be imported from module 'M.C'}}
  // expected-note@* 2{{declaration here is not visible}}
}
