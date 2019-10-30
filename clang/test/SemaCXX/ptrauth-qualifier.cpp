// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++11  -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)

struct __attribute__((trivial_abi)) AddrDisc { // expected-warning {{'trivial_abi' cannot be applied to 'AddrDisc'}}
  int * AQ m0;
};

struct __attribute__((trivial_abi)) NoAddrDisc {
  int * IQ m0;
};

namespace test_union {

  union U0 {
    int * AQ f0; // expected-note 4 {{'U0' is implicitly deleted because variant field 'f0' has an address-discriminated ptrauth qualifier}}

    // ptrauth fields that don't have an address-discriminated qualifier don't
    // delete the special functions.
    int * IQ f1;
  };

  union U1 {
    int * AQ f0; // expected-note 8 {{'U1' is implicitly deleted because variant field 'f0' has an address-discriminated ptrauth qualifier}}
    U1() = default;
    ~U1() = default;
    U1(const U1 &) = default; // expected-warning {{explicitly defaulted copy constructor is implicitly deleted}} expected-note 2 {{explicitly defaulted function was implicitly deleted here}}
    U1(U1 &&) = default; // expected-warning {{explicitly defaulted move constructor is implicitly deleted}}
    U1 & operator=(const U1 &) = default; // expected-warning {{explicitly defaulted copy assignment operator is implicitly deleted}} expected-note 2 {{explicitly defaulted function was implicitly deleted here}}
    U1 & operator=(U1 &&) = default; // expected-warning {{explicitly defaulted move assignment operator is implicitly deleted}}
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
      int * AQ f0; // expected-note 4 {{'' is implicitly deleted because variant field 'f0' has an address-discriminated ptrauth qualifier}}
      char f1;
    };
  };

  struct S1 {
    union {
      union { // expected-note 2 {{'S1' is implicitly deleted because variant field '' has a non-trivial}} expected-note 2 {{'S1' is implicitly deleted because field '' has a deleted}}
        int * AQ f0;
        char f1;
      };
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
