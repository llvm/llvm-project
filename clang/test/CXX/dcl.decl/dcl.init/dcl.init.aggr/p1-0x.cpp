// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// An aggregate is an array or a class...
struct Aggr {
private:
  static const int n;
  void f();
protected:
  struct Inner { int m; };
public:
  bool &br; // expected-note {{default constructor of 'Aggr' is implicitly deleted because field 'br' of reference type 'bool &' would not be initialized}}
};
bool b;
Aggr ag = { b };

// with no user-provided constructors, ...
struct NonAggr1a { // expected-note 2 {{candidate constructor}}
  NonAggr1a(int, int); // expected-note {{candidate constructor}}
  int k;
};
// In C++0x, 'user-provided' is only defined for special member functions, so
// this type is considered to be an aggregate. This is considered to be
// a language defect.
NonAggr1a na1a = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr1a'}}

struct NonAggr1b {
  NonAggr1b(const NonAggr1b &); // expected-note {{candidate constructor}}
  int k;
};
NonAggr1b na1b = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr1b'}}

// no brace-or-equal-initializers for non-static data members, ...
struct NonAggr2 { // expected-note 3 {{candidate constructor}}
  int m = { 123 };
};
NonAggr2 na2 = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr2'}}

// no private...
struct NonAggr3 { // expected-note 3 {{candidate constructor}}
private:
  int n;
};
NonAggr3 na3 = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr3'}}

// or protected non-static data members, ...
struct NonAggr4 { // expected-note 3 {{candidate constructor}}
protected:
  int n;
};
NonAggr4 na4 = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr4'}}

// no base classes, ...
struct NonAggr5 : Aggr { // expected-note 3 {{candidate constructor}}
};
NonAggr5 na5 = { b }; // expected-error {{no matching constructor for initialization of 'NonAggr5'}}
template<typename...BaseList>
struct MaybeAggr5a : BaseList... {}; // expected-note {{default constructor of 'MaybeAggr5a<Aggr>' is implicitly deleted because base class 'Aggr' has a deleted default constructor}}
MaybeAggr5a<> ma5a0 = {}; // ok
MaybeAggr5a<Aggr> ma5a1 = {}; // expected-error {{call to implicitly-deleted default constructor of 'MaybeAggr5a<Aggr>'}}

// and no virtual functions.
struct NonAggr6 { // expected-note 3 {{candidate constructor}}
  virtual void f();
  int n;
};
NonAggr6 na6 = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr6'}}

struct DefaultedAggr {
  int n;

  DefaultedAggr() = default;
  DefaultedAggr(const DefaultedAggr &) = default;
  DefaultedAggr(DefaultedAggr &&) = default;
  DefaultedAggr &operator=(const DefaultedAggr &) = default;
  DefaultedAggr &operator=(DefaultedAggr &&) = default;
  ~DefaultedAggr() = default;
};
DefaultedAggr da = { 42 } ;
