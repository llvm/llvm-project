// RUN: %clang_cc1 -std=c++23 -fsyntax-only %s -verify -fcxx-exceptions
// RUN: %clang_cc1 -std=c++23 -fsyntax-only %s -verify -fexperimental-new-constant-interpreter -fcxx-exceptions

namespace GH139818{
    struct A {
      constexpr ~A() { ref = false; }
      constexpr operator bool() {
        return b;
      }
      bool b;
      bool& ref;
    };

    constexpr bool f1() {
      bool ret = true;
      for (bool b = false; A x{b, ret}; b = true) {}
      return ret;
    }

    static_assert(!f1());

    struct Y {
      constexpr ~Y() noexcept(false) { throw "oops"; }  // expected-note {{subexpression not valid in a constant expression}}
                                                        
      constexpr operator bool() {
        return b;
      }
      bool b;
    };
    constexpr bool f2() {
      for (bool b = false; Y x = {b}; b = true) {} // expected-note {{in call to 'x.~Y()'}}
      return true;
    }
    static_assert(f2()); // expected-error {{static assertion expression is not an integral constant expression}}
                         // expected-note@-1 {{in call to 'f2()'}}
};
