// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fcxx-exceptions -verify=ref,both %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fcxx-exceptions -verify=expected,both %s -fexperimental-new-constant-interpreter


namespace std {
  struct type_info;
  struct destroying_delete_t {
    explicit destroying_delete_t() = default;
  } inline constexpr destroying_delete{};
  struct nothrow_t {
    explicit nothrow_t() = default;
  } inline constexpr nothrow{};
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t {};
};

constexpr void *operator new(std::size_t, void *p) { return p; }
namespace std {
  template<typename T> constexpr T *construct(T *p) { return new (p) T; }
  template<typename T> constexpr void destroy(T *p) { p->~T(); }
}

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

namespace std {
  struct type_info;
}

namespace TypeId {
  struct A {
    const std::type_info &ti = typeid(*this);
  };
  struct A2 : A {};
  static_assert(&A().ti == &typeid(A));
  static_assert(&typeid((A2())) == &typeid(A2));
  extern A2 extern_a2;
  static_assert(&typeid(extern_a2) == &typeid(A2));

  constexpr A2 a2;
  constexpr const A &a1 = a2;
  static_assert(&typeid(a1) == &typeid(A));

  struct B {
    virtual void f();
    const std::type_info &ti1 = typeid(*this);
  };
  struct B2 : B {
    const std::type_info &ti2 = typeid(*this);
  };
  static_assert(&B2().ti1 == &typeid(B));
  static_assert(&B2().ti2 == &typeid(B2));
  extern B2 extern_b2;
  static_assert(&typeid(extern_b2) == &typeid(B2));

  constexpr B2 b2;
  constexpr const B &b1 = b2;
  static_assert(&typeid(b1) == &typeid(B2));

  constexpr bool side_effects() {
    // Not polymorphic nor a glvalue.
    bool OK = true;
    (void)typeid(OK = false, A2()); // both-warning {{has no effect}}
    if (!OK) return false;

    // Not polymorphic.
    A2 a2;
    (void)typeid(OK = false, a2); // both-warning {{has no effect}}
    if (!OK) return false;

    // Not a glvalue.
    (void)typeid(OK = false, B2()); // both-warning {{has no effect}}
    if (!OK) return false;

    // Polymorphic glvalue: operand evaluated.
    OK = false;
    B2 b2;
    (void)typeid(OK = true, b2); // both-warning {{will be evaluated}}
    return OK;
  }
  static_assert(side_effects());
}

consteval int f(int i);
constexpr bool test(auto i) {
    return f(0) == 0;
}
consteval int f(int i) {
    return 2 * i;
}
static_assert(test(42));

namespace PureVirtual {
  struct Abstract {
    constexpr virtual void f() = 0; // both-note {{declared here}}
    constexpr Abstract() { do_it(); } // both-note {{in call to}}
    constexpr void do_it() { f(); } // both-note {{pure virtual function 'PureVirtual::Abstract::f' called}}
  };
  struct PureVirtualCall : Abstract { void f(); }; // both-note {{in call to 'Abstract}}
  constexpr PureVirtualCall pure_virtual_call; // both-error {{constant expression}} both-note {{in call to 'PureVirtualCall}}
}

namespace Dtor {
  constexpr bool pseudo(bool read, bool recreate) {
    using T = bool;
    bool b = false; // both-note {{lifetime has already ended}}
    // This evaluates the store to 'b'...
    (b = true).~T();
    // ... and ends the lifetime of the object.
    return (read
            ? b // both-note {{read of object outside its lifetime}}
            : true) +
           (recreate
            ? (std::construct(&b), true)
            : true);
  }
  static_assert(pseudo(false, false)); // both-error {{constant expression}} both-note {{in call}}
  static_assert(pseudo(true, false)); // both-error {{constant expression}} both-note {{in call}}
  static_assert(pseudo(false, true));
}

namespace GH150705 {
  struct A { };
  struct B : A { };
  struct C : A {
    constexpr virtual int foo() const { return 0; }
  };

  constexpr auto p = &C::foo;
  constexpr auto q = static_cast<int (A::*)() const>(p);
  constexpr B b;
  constexpr const A& a = b;
  constexpr auto x = (a.*q)(); // both-error {{constant expression}}
}

namespace DependentRequiresExpr {
  template <class T,
            bool = []() -> bool { // both-error {{not a constant expression}}
              if (requires { T::type; })
                return true;
              return false;
            }()>
  struct p {
    using type = void;
  };

  template <class T> using P = p<T>::type; // both-note {{while checking a default template argument}}
}
