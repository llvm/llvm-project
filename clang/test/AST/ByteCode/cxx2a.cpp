// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fcxx-exceptions -verify=ref,both %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fcxx-exceptions -verify=expected,both %s -fexperimental-new-constant-interpreter

template <unsigned N>
struct S {
  S() requires (N==1) = default;
  S() requires (N==2) {} // both-note {{declared here}}
  consteval S() requires (N==3) = default;
};

consteval int aConstevalFunction() { // both-error {{consteval function never produces a constant expression}}
  S<2> s4; // both-note {{non-constexpr constructor 'S' cannot be used in a constant expression}}
  return 0;
}
/// We're NOT calling the above function. The diagnostics should appear anyway.

namespace Covariant {
  struct A {
    virtual constexpr char f() const { return 'Z'; }
    char a = f();
  };

  struct D : A {};
  struct Covariant1 {
    D d;
    virtual const A *f() const;
  };

  struct Covariant3 : Covariant1 {
    constexpr virtual const D *f() const { return &this->d; }
  };

  constexpr Covariant3 cb;
  constexpr const Covariant1 *cb1 = &cb;
  static_assert(cb1->f()->a == 'Z');
}

namespace DtorOrder {
  struct Buf {
    char buf[64];
    int n = 0;
    constexpr void operator+=(char c) { buf[n++] = c; }
    constexpr bool operator==(const char *str) const {
      if (str[n] != 0)
        return false;

      for (int i = 0; i < n; ++i) {
        if (buf[n] != str[n])
          return false;
      }
      return true;

      return __builtin_memcmp(str, buf, n) == 0;
    }
    constexpr bool operator!=(const char *str) const { return !operator==(str); }
  };

  struct A {
    constexpr A(Buf &buf, char c) : buf(buf), c(c) { buf += c; }
    constexpr ~A() { buf += (c - 32);}
    constexpr operator bool() const { return true; }
    Buf &buf;
    char c;
  };

  constexpr void abnormal_termination(Buf &buf) {
    struct Indestructible {
      constexpr ~Indestructible(); // not defined
    };
    A a(buf, 'a');
    A(buf, 'b');
    int n = 0;

    for (A &&c = A(buf, 'c'); A d = A(buf, 'd'); A(buf, 'e')) {
      switch (A f(buf, 'f'); A g = A(buf, 'g')) { // both-warning {{boolean}}
      case false: {
        A x(buf, 'x');
      }

      case true: {
        A h(buf, 'h');
        switch (n++) {
        case 0:
          break;
        case 1:
          continue;
        case 2:
          return;
        }
        break;
      }

      default:
        Indestructible indest;
      }

      A j = (A(buf, 'i'), A(buf, 'j'));
    }
  }

  constexpr bool check_abnormal_termination() {
    Buf buf = {};
    abnormal_termination(buf);
    return buf ==
      "abBc"
        "dfgh" /*break*/ "HGFijIJeED"
        "dfgh" /*continue*/ "HGFeED"
        "dfgh" /*return*/ "HGFD"
      "CA";
  }
  static_assert(check_abnormal_termination());
}
