// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++20  -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20  -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s

#define AQ __ptrauth(1,1,50)
#define AQ2 __ptrauth(1,1,51)
#define IQ __ptrauth(1,0,50)

struct __attribute__((trivial_abi)) AddrDisc { // expected-warning {{'trivial_abi' cannot be applied to 'AddrDisc'}} expected-note {{'trivial_abi' is disallowed on 'AddrDisc' because it has an address-discriminated '__ptrauth' field}}
  int * AQ m0;
};

struct __attribute__((trivial_abi)) NoAddrDisc {
  int * IQ m0;
};

namespace test_union {

  union U0 {
    int * AQ f0; // expected-note 4 {{'U0' is implicitly deleted because variant field 'f0' has an address-discriminated '__ptrauth' qualifier}}

    // ptrauth fields that don't have an address-discriminated qualifier don't
    // delete the special functions.
    int * IQ f1;
  };

  union U1 {
    int * AQ f0; // expected-note 8 {{'U1' is implicitly deleted because variant field 'f0' has an address-discriminated '__ptrauth' qualifier}}
    U1() = default;
    ~U1() = default;
    U1(const U1 &) = default; // expected-warning {{explicitly defaulted copy constructor is implicitly deleted}} expected-note 2 {{explicitly defaulted function was implicitly deleted here}} expected-note{{replace 'default'}}
    U1(U1 &&) = default; // expected-warning {{explicitly defaulted move constructor is implicitly deleted}} expected-note{{replace 'default'}}
    U1 & operator=(const U1 &) = default; // expected-warning {{explicitly defaulted copy assignment operator is implicitly deleted}} expected-note 2 {{explicitly defaulted function was implicitly deleted here}} expected-note{{replace 'default'}}
    U1 & operator=(U1 &&) = default; // expected-warning {{explicitly defaulted move assignment operator is implicitly deleted}} expected-note{{replace 'default'}}
  };

  // It's fine if the user has explicitly defined the special functions.
  union U2 {
    int * AQ f0;
    U2() = default;
    ~U2() = default;
    U2(const U2 &);
    U2(U2 &&);
    U2 & operator=(const U2 &);
    U2 & operator=(U2 &&);
  };

  // Address-discriminated ptrauth fields in anonymous union fields delete the
  // defaulted copy/move constructors/assignment operators of the containing
  // class.
  struct S0 {
    union {
      int * AQ f0; // expected-note 4 {{' is implicitly deleted because variant field 'f0' has an address-discriminated '__ptrauth' qualifier}}
      char f1;
    };
  };

  struct S1 {
    union {
      union {
        int * AQ f0; // expected-note 4 {{implicitly deleted because variant field 'f0' has an address-discriminated '__ptrauth' qualifier}}
        char f1;
      } u; // expected-note 4 {{'S1' is implicitly deleted because field 'u' has a deleted}}
      int f2;
    };
  };

  U0 *x0;
  U1 *x1;
  U2 *x2;
  S0 *x3;
  S1 *x4;

  // No diagnostics since constructors/destructors of the unions aren't deleted by default.
  void testDefaultConstructor() {
    U0 u0;
    U1 u1;
    U2 u2;
    S0 s0;
    S1 s1;
  }

  // No diagnostics since destructors of the unions aren't deleted by default.
  void testDestructor(U0 *u0, U1 *u1, U2 *u2, S0 *s0, S1 *s1) {
    delete u0;
    delete u1;
    delete u2;
    delete s0;
    delete s1;
  }

  void testCopyConstructor(U0 *u0, U1 *u1, U2 *u2, S0 *s0, S1 *s1) {
    U0 t0(*u0); // expected-error {{call to implicitly-deleted copy constructor}}
    U1 t1(*u1); // expected-error {{call to implicitly-deleted copy constructor}}
    U2 t2(*u2);
    S0 t3(*s0); // expected-error {{call to implicitly-deleted copy constructor}}
    S1 t4(*s1); // expected-error {{call to implicitly-deleted copy constructor}}
  }

  void testCopyAssignment(U0 *u0, U1 *u1, U2 *u2, S0 *s0, S1 *s1) {
    *x0 = *u0; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x1 = *u1; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x2 = *u2;
    *x3 = *s0; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x4 = *s1; // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
  }

  void testMoveConstructor(U0 *u0, U1 *u1, U2 *u2, S0 *s0, S1 *s1) {
    U0 t0(static_cast<U0 &&>(*u0)); // expected-error {{call to implicitly-deleted copy constructor}}
    U1 t1(static_cast<U1 &&>(*u1)); // expected-error {{call to implicitly-deleted copy constructor}}
    U2 t2(static_cast<U2 &&>(*u2));
    S0 t3(static_cast<S0 &&>(*s0)); // expected-error {{call to implicitly-deleted copy constructor}}
    S1 t4(static_cast<S1 &&>(*s1)); // expected-error {{call to implicitly-deleted copy constructor}}
  }

  void testMoveAssignment(U0 *u0, U1 *u1, U2 *u2, S0 *s0, S1 *s1) {
    *x0 = static_cast<U0 &&>(*u0); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x1 = static_cast<U1 &&>(*u1); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x2 = static_cast<U2 &&>(*u2);
    *x3 = static_cast<S0 &&>(*s0); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
    *x4 = static_cast<S1 &&>(*s1); // expected-error {{cannot be assigned because its copy assignment operator is implicitly deleted}}
  }
}

bool test_composite_type0(bool c, int * AQ * a0, int * AQ * a1) {
  auto t = c ? a0 : a1;
  return a0 == a1;
}

bool test_composite_type1(bool c, int * AQ * a0, int * AQ2 * a1) {
  auto t = c ? a0 : a1; // expected-error {{incompatible operand types ('int *__ptrauth(1,1,50) *' and 'int *__ptrauth(1,1,51) *')}}
  return a0 == a1;      // expected-error {{comparison of distinct pointer types ('int *__ptrauth(1,1,50) *' and 'int *__ptrauth(1,1,51) *')}}
}

void test_bad_call_diag(void *AQ *ptr); // expected-note{{candidate function not viable: 1st argument ('void *__ptrauth(1,1,51) *') has __ptrauth(1,1,51) qualifier, but parameter has __ptrauth(1,1,50) qualifier}} expected-note {{candidate function not viable: 1st argument ('void **') has no '__ptrauth' qualifier, but parameter has __ptrauth(1,1,50) qualifier}}
void test_bad_call_diag2(void **ptr); // expected-note {{candidate function not viable: 1st argument ('void *__ptrauth(1,1,50) *') has __ptrauth(1,1,50) qualifier, but parameter has no '__ptrauth' qualifier}}

int test_call_diag() {
  void *AQ ptr1, *AQ2 ptr2, *ptr3;
  test_bad_call_diag(&ptr2); // expected-error {{no matching function for call to 'test_bad_call_diag'}}
  test_bad_call_diag(&ptr3); // expected-error {{no matching function for call to 'test_bad_call_diag'}}
  test_bad_call_diag2(&ptr1); // expected-error {{no matching function for call to 'test_bad_call_diag2'}}
}

namespace test_constexpr {
  constexpr int i = 100;
  constexpr const int * AQ p = &i;
  constexpr const int * const AQ *pp = &p;
  constexpr int i1 = **((const int * const AQ *)pp);
  constexpr int i2 = **((const int * const AQ2 *)pp);
  // expected-error@-1 {{constexpr variable 'i2' must be initialized by a constant expression}}
  // expected-note@-2 {{cast that performs the conversions of a reinterpret_cast is not allowed in a constant expression}}
}

namespace test_lambda {
  void test() {
    int * AQ v0;
    int * AQ *v1;

    [v0, v1]() {
      static_assert(__is_same(decltype(v0), int * AQ));
      static_assert(__is_same(decltype(v1), int * AQ *));
    }();

    [v2 = v0, v3 = v1]() {
      static_assert(__is_same(decltype(v2), int *));
      static_assert(__is_same(decltype(v3), int * AQ *));
    }();
  }
}

namespace test_concept {
  template <typename T> struct is_qualified {
    static constexpr bool value = false;
  };

  template <typename T> struct is_qualified<T * AQ> {
    static constexpr bool value = true;
  };

  template <typename T>
  concept Ptrauthable = is_qualified<T>::value;
  // expected-note@-1 2 {{because 'is_qualified<int *>::value' evaluated to false}}
  // expected-note@-2 2 {{because 'is_qualified<int *__ptrauth(1,1,51)>::value' evaluated to false}}

  template <typename T>
    requires(Ptrauthable<T>)
  struct S {};
  // expected-note@-2 {{because 'int *' does not satisfy 'Ptrauthable'}}
  // expected-note@-3 {{because 'int *__ptrauth(1,1,51)' does not satisfy 'Ptrauthable'}}

  S<int * AQ> s0;
  S<int *> s1;
  // expected-error@-1 {{constraints not satisfied for class template 'S' [with T = int *]}}
  S<int * AQ2> s1;
  // expected-error@-1 {{constraints not satisfied for class template 'S' [with T = int *__ptrauth(1,1,51)]}}

  template <typename T>
    requires(Ptrauthable<T>)
  void func(T *);
  // expected-note@-1 {{candidate template ignored: constraints not satisfied [with T = int *]}}
  // expected-note@-3 {{because 'int *' does not satisfy 'Ptrauthable'}}
  // expected-note@-3 {{candidate template ignored: constraints not satisfied [with T = int *__ptrauth(1,1,51)]}}
  // expected-note@-5 {{because 'int *__ptrauth(1,1,51)' does not satisfy 'Ptrauthable'}}

  void test() {
    int * AQ p0;
    int *p1;
    int * AQ2 p2;
    func(&p0);
    func(&p1); // expected-error {{no matching function for call to 'func'}}
    func(&p2); // expected-error {{no matching function for call to 'func'}}
  }
}
