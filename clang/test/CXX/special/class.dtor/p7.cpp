// RUN: %clang_cc1 -verify -std=c++26 %s -Wno-defaulted-function-deleted -triple x86_64-linux-gnu

struct NonTrivial {
    NonTrivial(int) { }
    ~NonTrivial() { }
};

union U0 {
    NonTrivial nt;
    int i;
  };
  U0 u0;

// overload resolution to select a constructor to default-initialize an object of type X either fails
union U1 {
    U1(int);
    NonTrivial nt;
};
U1 u1(1); // expected-error {{deleted destructor}}

// or selects a constructor that is either deleted or not trivial, or
union U2 {
    U2() : nt(2) { }
    NonTrivial nt;
};
U2 u2; // expected-error {{deleted destructor}}

union U3 {
    U3() = delete;
    U3(int);
    NonTrivial nt;
};
U3 u3(1); // expected-error {{deleted destructor}}

// or X has a variant member V of class type M (or possibly multi-dimensional array thereof) where V has a default member initializer and M has a destructor that is non-trivial,
union U4 {
    NonTrivial nt = 1;
};
U4 u4; // expected-error {{deleted destructor}}

union U5 {
  NonTrivial nt;
  U5* next = nullptr;
};
U5 u5;



