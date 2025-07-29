// RUN: %clang_cc1 -verify -std=c++26 %s -Wno-defaulted-function-deleted -triple x86_64-linux-gnu

struct NonTrivial {
  int i;
  constexpr NonTrivial(int i) :i(i) { }
  constexpr ~NonTrivial() { }
};

union U0 {
  NonTrivial nt;
  int i;
};
U0 u0;

// check for constant evaluation failure
constexpr NonTrivial make() {
    U0 u0;
    return u0.nt;
}
constexpr NonTrivial nt = make(); // expected-error {{must be initialized by a constant expression}}}
                                  // expected-note@-3 {{union with no active member}}
                                  // expected-note@-4 {{in call to 'NonTrivial(u0.nt)'}}
                                  // expected-note@-3 {{in call to 'make()'}}

// overload resolution to select a constructor to default-initialize an object of type X either fails
union U1 {
  U1(int);
  NonTrivial nt; // #1
};
U1 u1(1); // expected-error {{deleted function}} expected-note@#1 {{non-trivial destructor}}

// or selects a constructor that is either deleted or not trivial, or
union U2 {
  U2() : nt(2) { }
  NonTrivial nt; // #2
};
U2 u2; // expected-error {{deleted function}} expected-note@#2 {{non-trivial destructor}}

union U3 {
  U3() = delete;
  U3(int);
  NonTrivial nt; // #3
};
U3 u3(1); // expected-error {{deleted function}} expected-note@#3 {{non-trivial destructor}}

// or X has a variant member V of class type M (or possibly multi-dimensional array thereof) where V has a default member initializer and M has a destructor that is non-trivial,
union U4 {
  NonTrivial nt = 1; // #4
};
U4 u4; // expected-error {{deleted function}} expected-note@#4 {{non-trivial destructor}}

union U5 {
  NonTrivial nt[2] = {1, 2}; // #5
};
U5 u5; // expected-error {{deleted function}} expected-note@#5 {{non-trivial destructor}}

union U6 {
  NonTrivial nt;
  U6* next = nullptr;
};
U6 u6;

union U7 {
  U7() = default;
  NonTrivial nt;
  U6* next = nullptr;
};
U7 u7;

union U8 {
    struct {
        NonTrivial x;
    };
} u8;

union U9 {
    struct {
        NonTrivial x = 1; // #6
    };
} u9; // expected-error {{deleted function}}

struct DeletedDtor {
  ~DeletedDtor() = delete; // expected-note 2 {{deleted here}}
};
union B1 {
  B1();
  DeletedDtor a; // expected-note {{because field 'a' has a deleted destructor}}
};
B1 b1; // expected-error {{deleted function}}
union B2 {
  B2();
  union {          // expected-note {{deleted destructor}}
    DeletedDtor a; // expected-note {{because field 'a' has a deleted destructor}}
  };
};
B2 b2; // expected-error {{deleted function}}
