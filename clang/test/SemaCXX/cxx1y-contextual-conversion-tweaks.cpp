// RUN: %clang_cc1 -std=c++11 -verify=expected,cxx11 -fsyntax-only -pedantic-errors %s
// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx14 -fsyntax-only -pedantic-errors %s -DCXX1Y

// Explicit member declarations behave as in C++11.

namespace n3323_example {

  template <class T> class zero_init {
  public:
    zero_init() : val(static_cast<T>(0)) {}
    zero_init(T val) : val(val) {}

    operator T &() { return val; }     //@13
    operator T() const { return val; } //@14

  private:
    T val;
  };

  void Delete() {
    zero_init<int *> p;
    p = new int(7);
    delete p; //@23
    delete (p + 0);
    delete + p;
  }

  void Switch() {
    zero_init<int> i;
    i = 7;
    switch (i) {} // @31
    switch (i + 0) {}
    switch (+i) {}
  }
}

#ifdef CXX1Y
#else
//expected-error@23 {{ambiguous conversion of delete expression of type 'zero_init<int *>' to a pointer}}
//expected-note@13 {{conversion to pointer type 'int *'}}
//expected-note@14 {{conversion to pointer type 'int *'}}
//expected-error@31 {{multiple conversions from switch condition type 'zero_init<int>' to an integral or enumeration type}}
//expected-note@13 {{conversion to integral type 'int'}}
//expected-note@14 {{conversion to integral type 'int'}}
#endif

namespace extended_examples {

  struct A0 {
    operator int();      // matching and viable
  };

  struct A1 {
    operator int() &&;   // matching and not viable
  };

  struct A2 {
    operator float();    // not matching
  };

  struct A3 {
    template<typename T> operator T();  // not matching (ambiguous anyway)
  };

  struct A4 {
    template<typename T> operator int();  // not matching (ambiguous anyway)
  };

  struct B1 {
    operator int() &&;  // @70
    operator int();     // @71  -- duplicate declaration with different qualifier is not allowed
  };

  struct B2 {
    operator int() &&;  // matching but not viable
    operator float();   // not matching
  };

  void foo(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, B2 b2) {
    switch (a0) {}
    switch (a1) {}     // @81 -- fails for different reasons
    switch (a2) {}     // @82
    switch (a3) {}     // @83
    switch (a4) {}     // @84
    switch (b2) {}     // @85 -- fails for different reasons
  }
}

//expected-error@71 {{cannot overload a member function without a ref-qualifier with a member function with ref-qualifier '&&'}}
//expected-note@70 {{previous declaration is here}}
//expected-error@82 {{statement requires expression of integer type ('A2' invalid)}}
//expected-error@83 {{statement requires expression of integer type ('A3' invalid)}}
//expected-error@84 {{statement requires expression of integer type ('A4' invalid)}}

#ifdef CXX1Y
//expected-error@81 {{statement requires expression of integer type ('A1' invalid)}}
//expected-error@85 {{statement requires expression of integer type ('B2' invalid)}}
#else
//expected-error@81 {{'this' argument to member function 'operator int' is an lvalue, but function has rvalue ref-qualifier}} expected-note@54 {{'operator int' declared here}}
//expected-error@85 {{'this' argument to member function 'operator int' is an lvalue, but function has rvalue ref-qualifier}} expected-note@75 {{'operator int' declared here}}
#endif

namespace extended_examples_cxx1y {

  struct A1 {   // leads to viable match in C++1y, and no viable match in C++11
    operator int() &&;                  // matching but not viable
    template <typename T> operator T(); // In C++1y: matching and viable, since disambiguated by L.100
  };

  struct A2 {   // leads to ambiguity in C++1y, and no viable match in C++11
    operator int() &&;                    // matching but not viable
    template <typename T> operator int(); // In C++1y: matching but ambiguous (disambiguated by L.105).
  };

  struct B1 {    // leads to one viable match in both cases
    operator int();                  // matching and viable
    template <typename T> operator T(); // In C++1y: matching and viable, since disambiguated by L.110
  };

  struct B2 {    // leads to one viable match in both cases
    operator int();                  // matching and viable
    template <typename T> operator int(); // In C++1y: matching but ambiguous, since disambiguated by L.115
  };

  struct C {    // leads to no match in both cases
    operator float();                  // not matching
    template <typename T> operator T(); // In C++1y: not matching, nor viable.
  };

  struct D {   // leads to viable match in C++1y, and no viable match in C++11
    operator int() &&;                  // matching but not viable
    operator float();                   // not matching
    template <typename T> operator T(); // In C++1y: matching and viable, since disambiguated by L.125
  };


  void foo(A1 a1, A2 a2, B1 b1, B2 b2, C c, D d) {
    switch (a1) {} // @138 --  should presumably call templated conversion operator to convert to int.
    switch (a2) {} // @139
    switch (b1) {}
    switch (b2) {}
    switch (c) {}  // @142
    switch (d) {}  // @143
  }
}

//expected-error@142 {{statement requires expression of integer type ('C' invalid)}}

#ifdef CXX1Y
//expected-error@139 {{statement requires expression of integer type ('A2' invalid)}}
#else
//expected-error@138 {{'this' argument to member function 'operator int' is an lvalue, but function has rvalue ref-qualifier}} expected-note@106 {{'operator int' declared here}}
//expected-error@139 {{'this' argument to member function 'operator int' is an lvalue, but function has rvalue ref-qualifier}} expected-note@111 {{'operator int' declared here}}
//expected-error@143 {{'this' argument to member function 'operator int' is an lvalue, but function has rvalue ref-qualifier}} expected-note@131 {{'operator int' declared here}}
#endif

namespace extended_examples_array_bounds {

  typedef decltype(sizeof(int)) size_t;

  struct X {
    constexpr operator size_t() const { return 1; } // cxx11-note 3{{conversion}}
    constexpr operator unsigned short() const { return 0; } // cxx11-note 3{{conversion}}
  };

  void f() {
    X x;
    int *p = new int[x]; // cxx11-error {{ambiguous}}

    int arr[x]; // cxx11-error {{ambiguous}}
    int (*q)[1] = new int[1][x]; // cxx11-error {{ambiguous}}
  }

  struct Y {
    constexpr operator float() const { return 0.0f; } // cxx14-note 3{{candidate}}
    constexpr operator int() const { return 1; } // cxx14-note 3{{candidate}}
  };

  void g() {
    Y y;
    int *p = new int[y]; // cxx14-error {{ambiguous}}

    int arr[y]; // cxx14-error {{ambiguous}}
    int (*q)[1] = new int[1][y]; // cxx14-error {{ambiguous}}
  }

  template<int N> struct Z {
    constexpr operator int() const { return N; }
  };
  void h() {
    int arrA[Z<1>()];
    int arrB[Z<0>()]; // expected-error {{zero size array}}
    int arrC[Z<-1>()]; // expected-error {{'arrC' declared as an array with a negative size}}
  }
}
