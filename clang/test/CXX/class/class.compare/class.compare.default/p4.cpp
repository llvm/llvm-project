// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -Wc++23-default-comp-relaxed-constexpr -verify=expected,extension %s

// This test is for [class.compare.default]p3 as modified and renumbered to p4
// by P2002R0.
// Also covers modifications made by P2448R2 and extension warnings

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
}

namespace N {
  struct A {
    friend constexpr std::strong_ordering operator<=>(const A&, const A&) = default;
  };

  constexpr bool (*test_a_not_found)(const A&, const A&) = &operator==; // expected-error {{undeclared}}

  constexpr bool operator==(const A&, const A&) noexcept;
  constexpr bool (*test_a)(const A&, const A&) noexcept = &operator==;
  static_assert((*test_a)(A(), A()));
}

struct B1 {
  virtual std::strong_ordering operator<=>(const B1&) const = default;
};
bool (B1::*test_b)(const B1&) const = &B1::operator==;

struct C1 : B1 {
  // OK, B1::operator== is virtual.
  bool operator==(const B1&) const override;
};

struct B2 {
  std::strong_ordering operator<=>(const B2&) const = default;
};

struct C2 : B2 {
  bool operator==(const B2&) const override; // expected-error {{only virtual member functions}}
};

struct D {
  std::strong_ordering operator<=>(const D&) const;
  virtual std::strong_ordering operator<=>(const struct E&) const = 0;
};
struct E : D {
  // expected-error@+2 {{only virtual member functions}}
  // expected-note@+1 {{while declaring the corresponding implicit 'operator==' for this defaulted 'operator<=>'}}
  std::strong_ordering operator<=>(const E&) const override = default;
};

struct F {
  [[deprecated("oh no")]] std::strong_ordering operator<=>(const F&) const = default; // expected-note 4{{deprecated}}
};
void use_f(F f) {
  void(f <=> f); // expected-warning {{oh no}}
  void(f < f); // expected-warning {{oh no}}
  void(f == f); // expected-warning {{oh no}}
  void(f != f); // expected-warning {{oh no}}
}

class G {
  // expected-note@+2 {{implicitly declared private here}}
  // expected-note-re@+1 {{{{^}}declared private here}}
  std::strong_ordering operator<=>(const G&) const = default;
public:
};
void use_g(G g) {
  void(g <=> g); // expected-error {{private}}
  void(g == g); // expected-error {{private}}
}

struct H {
  bool operator==(const H&) const; // extension-note {{non-constexpr comparison function declared here}}
  constexpr std::strong_ordering operator<=>(const H&) const { return std::strong_ordering::equal; }
};

struct I {
  H h; // extension-note {{non-constexpr comparison function would be used to compare member 'h'}}
  constexpr std::strong_ordering operator<=>(const I&) const = default; // extension-warning {{implicit 'operator=='  invokes a non-constexpr comparison function is a C++23 extension}}
};

struct J {
  std::strong_ordering operator<=>(const J&) const & = default; // expected-note {{candidate function (the implicit 'operator==' for this 'operator<=>)'}}
  friend std::strong_ordering operator<=>(const J&, const J&) = default; // expected-note {{candidate function (the implicit 'operator==' for this 'operator<=>)'}}
};
void use_j(J j) {
  void(j == j); // expected-error {{ambiguous}}
}

namespace DeleteAfterFirstDecl {
  bool operator==(const struct Q&, const struct Q&);
  struct Q {
    struct X {
      friend std::strong_ordering operator<=>(const X&, const X&);
    } x; // expected-note {{no viable 'operator=='}}
    // expected-error@+1 {{defaulting the corresponding implicit 'operator==' for this defaulted 'operator<=>' would delete it after its first declaration}}
    friend std::strong_ordering operator<=>(const Q&, const Q&) = default;
  };
}

// Note, substitution here results in the second parameter of 'operator=='
// referring to the first parameter of 'operator==', not to the first parameter
// of 'operator<=>'.
// FIXME: Find a case where this matters (attribute enable_if?).
struct K {
  friend std::strong_ordering operator<=>(const K &k, decltype(k)) = default;
};
bool test_k = K() == K();

namespace NoInjectionIfOperatorEqualsDeclared {
  struct A {
    void operator==(int); // expected-note 2{{not viable}}
    std::strong_ordering operator<=>(const A&) const = default;
  };
  bool test_a = A() == A(); // expected-error {{invalid operands}}

  struct B {
    friend void operator==(int, struct Q); // expected-note 2{{not viable}}
    std::strong_ordering operator<=>(const B&) const = default;
  };
  bool test_b = B() == B(); // expected-error {{invalid operands}}

  struct C {
    void operator==(int); // expected-note 2{{not viable}}
    friend std::strong_ordering operator<=>(const C&, const C&) = default;
  };
  bool test_c = C() == C(); // expected-error {{invalid operands}}

  struct D {
    void f() {
      void operator==(const D&, int);
    }
    struct X {
      friend void operator==(const D&, int);
    };
    friend std::strong_ordering operator<=>(const D&, const D&) = default;
  };
  bool test_d = D() == D();
}

namespace GH61238 {
template <typename A> struct my_struct {
    A value; // extension-note {{non-constexpr comparison function would be used to compare member 'value'}}

    constexpr friend bool operator==(const my_struct &, const my_struct &) noexcept = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
};

struct non_constexpr_type {
    friend bool operator==(non_constexpr_type, non_constexpr_type) noexcept { // extension-note {{non-constexpr comparison function declared here}}
        return false;
    }
};

my_struct<non_constexpr_type> obj; // extension-note {{in instantiation of template class 'GH61238::my_struct<GH61238::non_constexpr_type>' requested here}}
}
