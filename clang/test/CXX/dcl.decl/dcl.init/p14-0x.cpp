// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct NoDefault {
  NoDefault() = delete; // expected-note {{here}}
  NoDefault(int);
};
struct Explicit { // expected-note 2 {{candidate}} expected-note {{here}}
  explicit Explicit(int); // expected-note {{not a candidate}}
};
struct NoCopy {
  NoCopy();
  NoCopy(const NoCopy &) = delete; // expected-note {{here}}
};
struct NoMove {
  NoMove();
  NoMove(NoMove &&) = delete; // expected-note {{here}}
};
class Private {
  Private(int); // expected-note {{here}}
public:
  Private();
};
class Friend {
  friend class S;
  Friend(int);
};


class S {
  NoDefault nd1;
  NoDefault nd2 = 42;
  Explicit e1; // expected-note {{here}}
  Explicit e2 = 42; // expected-error {{no viable conversion}}
  NoCopy nc = NoCopy(); // expected-error {{call to deleted}}
  NoMove nm = NoMove(); // expected-error {{call to deleted}}
  Private p = 42; // expected-error {{private constructor}}
  Friend f = 42;

  S() {} // expected-error {{call to deleted constructor of 'NoDefault'}} \
            expected-error {{must explicitly initialize the member 'e1' which does not have a default constructor}}
  S(int) : nd1(42), e1(42) {}
};

// FIXME: test the other forms which use copy-initialization
