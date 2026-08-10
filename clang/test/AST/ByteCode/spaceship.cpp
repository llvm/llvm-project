// RUN: %clang_cc1 -std=c++2a -verify=both,ref %s -fcxx-exceptions
// RUN: %clang_cc1 -std=c++2a -verify=both,expected %s -fcxx-exceptions -fexperimental-new-constant-interpreter

namespace std {
  struct strong_ordering { // both-note 6{{candidate}}
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};

  struct weak_ordering {
    int n;
    constexpr weak_ordering(int n) : n(n) {}
    constexpr weak_ordering(strong_ordering o) : n(o.n) {}
    constexpr operator int() const { return n; }
    static const weak_ordering less, equivalent, greater;
  };
  constexpr weak_ordering weak_ordering::less{-1},
      weak_ordering::equivalent{0}, weak_ordering::greater{1};

  struct partial_ordering {
    double d;
    constexpr partial_ordering(double d) : d(d) {}
    constexpr partial_ordering(strong_ordering o) : d(o.n) {}
    constexpr partial_ordering(weak_ordering o) : d(o.n) {}
    constexpr operator double() const { return d; }
    static const partial_ordering less, equivalent, greater, unordered;
  };
  constexpr partial_ordering partial_ordering::less{-1},
      partial_ordering::equivalent{0}, partial_ordering::greater{1},
      partial_ordering::unordered{__builtin_nan("")};

  static_assert(!(partial_ordering::unordered < 0));
  static_assert(!(partial_ordering::unordered == 0));
  static_assert(!(partial_ordering::unordered > 0));
}

namespace Deletedness {
  struct A {
    std::strong_ordering operator<=>(const A&) const;
  };
  struct B {
    bool operator==(const B&) const;
    bool operator<(const B&) const;
  };
  struct C {
    std::strong_ordering operator<=>(const C&) const = delete; // both-note 2{{deleted}}
  };
  struct D1 {
    bool operator==(const D1&) const;
    std::strong_ordering operator<=>(int) const; // both-note 2{{function not viable}} both-note 2{{function (with reversed parameter order) not viable}}
    bool operator<(int) const; // both-note 2{{function not viable}}
  };
  struct D2 {
    bool operator<(const D2&) const;
    std::strong_ordering operator<=>(int) const; // both-note 2{{function not viable}} both-note 2{{function (with reversed parameter order) not viable}}
    bool operator==(int) const; // both-note 2{{function not viable}}
  };
  struct E {
    bool operator==(const E&) const;
    bool operator<(const E&) const = delete; // both-note 2{{deleted}}
  };
  struct F {
    std::strong_ordering operator<=>(const F&) const; // both-note 2{{candidate}}
    std::strong_ordering operator<=>(F) const; // both-note 2{{candidate}}
  };
  struct G1 {
    bool operator==(const G1&) const;
    void operator<(const G1&) const;
  };
  struct G2 {
    void operator==(const G2&) const;
    bool operator<(const G2&) const;
  };
  struct H {
    void operator<=>(const H&) const;
  };

  // both-note@#base {{deleted comparison function for base class 'C'}}
  // both-note@#base {{no viable three-way comparison function for base class 'D1'}}
  // both-note@#base {{three-way comparison cannot be synthesized because there is no viable function for '<' comparison}}
  // both-note@#base {{no viable 'operator==' for base class 'D2'}}
  // both-note@#base {{three-way comparison cannot be synthesized because there is no viable function for '==' comparison}}
  // both-note@#base {{deleted comparison function for base class 'E'}}
  // both-note@#base {{implied comparison for base class 'F' is ambiguous}}
  template<typename T> struct Cmp : T { // #base
    std::strong_ordering operator<=>(const Cmp&) const = default; // #cmp both-note 5{{here}}
  };

  void use(...);
  void f() {
    use(
      Cmp<A>() <=> Cmp<A>(),
      Cmp<B>() <=> Cmp<B>(),
      Cmp<C>() <=> Cmp<C>(), // both-error {{deleted}}
      Cmp<D1>() <=> Cmp<D1>(), // both-error {{deleted}}
      Cmp<D2>() <=> Cmp<D2>(), // both-error {{deleted}}
      Cmp<E>() <=> Cmp<E>(), // both-error {{deleted}}
      Cmp<F>() <=> Cmp<F>(), // both-error {{deleted}}
      // FIXME: The following three errors are not very good.
      // both-error@#cmp {{value of type 'void' is not contextually convertible to 'bool'}}
      Cmp<G1>() <=> Cmp<G1>(), // both-note-re {{in defaulted three-way comparison operator for '{{.*}}Cmp<{{.*}}G1>' first required here}}j
      // both-error@#cmp {{value of type 'void' is not contextually convertible to 'bool'}}
      Cmp<G2>() <=> Cmp<G2>(), // both-note-re {{in defaulted three-way comparison operator for '{{.*}}Cmp<{{.*}}G2>' first required here}}j
      // both-error@#cmp {{no matching conversion for static_cast from 'void' to 'std::strong_ordering'}}
      Cmp<H>() <=> Cmp<H>(), // both-note-re {{in defaulted three-way comparison operator for '{{.*}}Cmp<{{.*}}H>' first required here}}j
      0
    );
  }

  // both-note@#arr {{deleted comparison function for member 'arr'}}
  // both-note@#arr {{no viable three-way comparison function for member 'arr'}}
  // both-note@#arr {{three-way comparison cannot be synthesized because there is no viable function for '<' comparison}}
  // both-note@#arr {{no viable 'operator==' for member 'arr'}}
  // both-note@#arr {{three-way comparison cannot be synthesized because there is no viable function for '==' comparison}}
  // both-note@#arr {{deleted comparison function for member 'arr'}}
  // both-note@#arr {{implied comparison for member 'arr' is ambiguous}}
  template<typename T> struct CmpArray {
    T arr[3]; // #arr
    std::strong_ordering operator<=>(const CmpArray&) const = default; // #cmparray both-note 5{{here}}
  };
  void g() {
    use(
      CmpArray<A>() <=> CmpArray<A>(),
      CmpArray<B>() <=> CmpArray<B>(),
      CmpArray<C>() <=> CmpArray<C>(), // both-error {{deleted}}
      CmpArray<D1>() <=> CmpArray<D1>(), // both-error {{deleted}}
      CmpArray<D2>() <=> CmpArray<D2>(), // both-error {{deleted}}
      CmpArray<E>() <=> CmpArray<E>(), // both-error {{deleted}}
      CmpArray<F>() <=> CmpArray<F>(), // both-error {{deleted}}
      // FIXME: The following three errors are not very good.
      // both-error@#cmparray {{value of type 'void' is not contextually convertible to 'bool'}}
      CmpArray<G1>() <=> CmpArray<G1>(), // both-note-re {{in defaulted three-way comparison operator for '{{.*}}CmpArray<{{.*}}G1>' first required here}}j
      // both-error@#cmparray {{value of type 'void' is not contextually convertible to 'bool'}}
      CmpArray<G2>() <=> CmpArray<G2>(), // both-note-re {{in defaulted three-way comparison operator for '{{.*}}CmpArray<{{.*}}G2>' first required here}}j
      // both-error@#cmparray {{no matching conversion for static_cast from 'void' to 'std::strong_ordering'}}
      CmpArray<H>() <=> CmpArray<H>(), // both-note-re {{in defaulted three-way comparison operator for '{{.*}}CmpArray<{{.*}}H>' first required here}}j
      0
    );
  }
}

namespace Access {
  class A {
    std::strong_ordering operator<=>(const A &) const; // both-note {{here}}
  public:
    bool operator==(const A &) const;
    bool operator<(const A &) const;
  };
  struct B {
    A a; // both-note {{would invoke a private 'operator<=>'}}
    friend std::strong_ordering operator<=>(const B &, const B &) = default; // both-warning {{deleted}} both-note{{replace 'default'}}
  };

  class C {
    std::strong_ordering operator<=>(const C &); // not viable (not const)
    bool operator==(const C &) const; // both-note {{here}}
    bool operator<(const C &) const;
  };
  struct D {
    C c; // both-note {{would invoke a private 'operator=='}}
    friend std::strong_ordering operator<=>(const D &, const D &) = default; // both-warning {{deleted}} both-note{{replace 'default'}}
  };
}

namespace Synthesis {
  enum Result { False, True, Mu };

  constexpr bool toBool(Result R) {
    if (R == Mu) throw "should not ask this question";
    return R == True;
  }

  struct Val {
    Result equal, less;
    constexpr bool operator==(const Val&) const { return toBool(equal); }
    constexpr bool operator<(const Val&) const { return toBool(less); }
  };

  template<typename T> struct Cmp {
    Val val;
    friend T operator<=>(const Cmp&, const Cmp&) = default; // both-note {{deleted}}
  };

  template<typename T> constexpr auto cmp(Result equal, Result less = Mu, Result reverse_less = Mu) {
    return Cmp<T>{equal, less} <=> Cmp<T>{Mu, reverse_less};
  }

  static_assert(cmp<std::strong_ordering>(True) == 0);
  static_assert(cmp<std::strong_ordering>(False, True) < 0);
  static_assert(cmp<std::strong_ordering>(False, False) > 0);

  static_assert(cmp<std::weak_ordering>(True) == 0);
  static_assert(cmp<std::weak_ordering>(False, True) < 0);
  static_assert(cmp<std::weak_ordering>(False, False) > 0);

  static_assert(cmp<std::partial_ordering>(True) == 0);
  static_assert(cmp<std::partial_ordering>(False, True) < 0);
  static_assert(cmp<std::partial_ordering>(False, False, True) > 0);
  static_assert(!(cmp<std::partial_ordering>(False, False, False) > 0));
  static_assert(!(cmp<std::partial_ordering>(False, False, False) == 0));
  static_assert(!(cmp<std::partial_ordering>(False, False, False) < 0));

  // No synthesis is performed for a custom return type, even if it can be
  // converted from a standard ordering.
  struct custom_ordering {
    custom_ordering(std::strong_ordering o);
  };
  void f(Cmp<custom_ordering> c) {
    c <=> c; // both-error {{deleted}}
  }
}

namespace Preference {
  struct A {
    A(const A&) = delete; // both-note {{deleted}}
    // "usable" candidate that can't actually be called
    friend void operator<=>(A, A); // both-note {{passing}}
    // Callable candidates for synthesis not considered.
    friend bool operator==(A, A);
    friend bool operator<(A, A);
  };

  struct B {
    B();
    A a;
    std::strong_ordering operator<=>(const B&) const = default; // both-error {{call to deleted constructor of 'A'}}
  };
  bool x = B() < B(); // both-note {{in defaulted three-way comparison operator for 'B' first required here}}
}
