// RUN: %clang_cc1 -std=c++2a -fexceptions -verify %s
// RUN: %clang_cc1 -std=c++2a  -verify %s

namespace std {
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t;

  struct destroying_delete_t {
    struct __construct { explicit __construct() = default; };
    explicit destroying_delete_t(__construct) {}
  };

  inline constexpr destroying_delete_t destroying_delete(destroying_delete_t::__construct());
}

void operator delete(void*, std::destroying_delete_t); // ok, just a placement delete

struct A;
void operator delete(A*, std::destroying_delete_t); // expected-error {{first parameter of 'operator delete' must have type 'void *'}}

struct A {
  void operator delete(A*, std::destroying_delete_t);
  void operator delete(A*, std::destroying_delete_t, std::size_t);
  void operator delete(A*, std::destroying_delete_t, std::align_val_t);
  void operator delete(A*, std::destroying_delete_t, std::size_t, std::align_val_t);
  void operator delete(A*, std::destroying_delete_t, int); // expected-error {{destroying operator delete can have only an optional size and optional alignment parameter}}
  // FIXME: It's probably a language defect that we permit usual operator delete to be variadic.
  void operator delete(A*, std::destroying_delete_t, std::size_t, ...);

  void operator delete(struct X*, std::destroying_delete_t, std::size_t, ...); // expected-error {{first parameter of 'operator delete' must have type 'A *'}}

  void operator delete(void*, std::size_t);
};

void delete_A(A *a) { delete a; }

namespace convert_param {
  struct A {
    void operator delete(
        A*,
        std::destroying_delete_t);
  };
  struct B : private A { using A::operator delete; }; // expected-note 2{{declared private here}}
  struct C : B {};
  void delete_C(C *c) { delete c; } // expected-error {{cannot cast 'convert_param::C' to its private base class 'convert_param::A'}}

  // expected-error@-7 {{cannot cast 'convert_param::D' to its private base class 'convert_param::A'}}
  struct D : B { virtual ~D() {} }; // expected-note {{while checking implicit 'delete this' for virtual destructor}}
}

namespace delete_selection {
  struct B {
    void operator delete(void*) = delete;
    void operator delete(B *, std::destroying_delete_t) = delete; // expected-note {{deleted}}
  };
  void delete_B(B *b) { delete b; } // expected-error {{deleted}}

  struct C {
    C();
    void *operator new(std::size_t);
    void operator delete(void*) = delete;
    void operator delete(C *, std::destroying_delete_t) = delete; // expected-note 0-1 {{deleted here}}
  };
  // TODO: We only diagnose the use of a deleted operator delete when exceptions
  // are enabled. Otherwise we don't bother doing the lookup.
#ifdef __EXCEPTIONS
  // expected-error@+2 {{attempt to use a deleted function}}
#endif
  C *new_C() { return new C; }

  struct D {
    void operator delete(D *, std::destroying_delete_t) = delete; // expected-note {{deleted}}
    void operator delete(D *, std::destroying_delete_t, std::align_val_t) = delete;
  };
  void delete_D(D *d) { delete d; } // expected-error {{deleted}}

  struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) E {
    void operator delete(E *, std::destroying_delete_t) = delete;
    void operator delete(E *, std::destroying_delete_t, std::align_val_t) = delete; // expected-note {{deleted}}
  };
  void delete_E(E *e) { delete e; } // expected-error {{deleted}}

  struct F {
    void operator delete(F *, std::destroying_delete_t) = delete; // expected-note {{deleted}}
    void operator delete(F *, std::destroying_delete_t, std::size_t) = delete;
  };
  void delete_F(F *f) { delete f; } // expected-error {{deleted}}

  struct G {
    void operator delete(G *, std::destroying_delete_t, std::align_val_t) = delete;
    void operator delete(G *, std::destroying_delete_t, std::size_t) = delete; // expected-note {{deleted}}
  };
  void delete_G(G *g) { delete g; } // expected-error {{deleted}}

  struct H {
    void operator delete(H *, std::destroying_delete_t, std::align_val_t) = delete; // expected-note {{deleted}}
    void operator delete(H *, std::destroying_delete_t, std::size_t, std::align_val_t) = delete;
  };
  void delete_H(H *h) { delete h; } // expected-error {{deleted}}

  struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) I {
    void operator delete(I *, std::destroying_delete_t, std::size_t) = delete;
    void operator delete(I *, std::destroying_delete_t, std::size_t, std::align_val_t) = delete; // expected-note {{deleted}}
  };
  void delete_I(I *i) { delete i; } // expected-error {{deleted}}
}

namespace first_param_conversion {
  struct A {
    void operator delete(A *, std::destroying_delete_t);
  };
  void f(const volatile A *a) {
    delete a; // ok
  }

  struct B {
    void operator delete(B *, std::destroying_delete_t);
  };
  struct C : B {};
  struct D : B {};
  struct E : C, D {};
  void g(E *e) {
    delete e; // expected-error {{ambiguous conversion from derived class 'first_param_conversion::E' to base class 'first_param_conversion::B':}}
  }
}
