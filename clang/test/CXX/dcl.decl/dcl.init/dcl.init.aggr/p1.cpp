// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

// An aggregate is an array or a class...
struct Aggr {
private:
  static const int n;
  void f();
protected:
  struct Inner { int m; };
public:
  bool &br;
};
bool b;
Aggr ag = { b };

// with no user-provided constructors, ...
struct NonAggr1a { // expected-note 2 {{candidate constructor}}
  NonAggr1a(int, int); // expected-note {{candidate constructor}}
  int k;
};
NonAggr1a na1a = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr1a'}}

struct NonAggr1b {
  NonAggr1b(const NonAggr1b &); // expected-note {{candidate constructor}}
  int k;
};
NonAggr1b na1b = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr1b'}}

// no brace-or-equal-initializers for non-static data members, ...
// Note, this bullet was removed in C++1y.
struct NonAggr2 {
  int m = { 123 };
};
NonAggr2 na2 = { 42 };
#if __cplusplus < 201402L
// expected-error@-2 {{no matching constructor for initialization of 'NonAggr2'}}
// expected-note@-6 3 {{candidate constructor}}
#endif

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

// [pre-C++1z] no base classes, ...
struct NonAggr5 : Aggr {
};
NonAggr5 na5 = { b };
#if __cplusplus <= 201402L
// expected-error@-2 {{no matching constructor for initialization of 'NonAggr5'}}
// expected-note@-5 3 {{candidate constructor}}
#endif
template<typename...BaseList>
struct MaybeAggr5a : BaseList... {};
MaybeAggr5a<> ma5a0 = {}; // ok
MaybeAggr5a<Aggr> ma5a1 = {}; // ok in C++17
MaybeAggr5a<NonAggr2> m5a2 = {}; // ok, aggregate init in C++17, default ctor in C++11 and C++14
MaybeAggr5a<NonAggr2> m5a3 = {0}; // ok in C++17, overrides default member initializer in base class
#if __cplusplus <= 201402L
// expected-error@-4 {{call to implicitly-deleted default constructor of 'MaybeAggr5a<Aggr>'}}
// expected-note@-7 {{default constructor of 'MaybeAggr5a<Aggr>' is implicitly deleted because base class 'Aggr' has a deleted default constructor}}
// expected-note@13 {{default constructor of 'Aggr' is implicitly deleted because field 'br' of reference type 'bool &' would not be initialized}}
// expected-error@-5 {{no matching constructor}} expected-note@-9 3{{candidate}}
#else
// expected-error@-9 {{reference member of type 'bool &' uninitialized}}
// expected-note@13 {{uninitialized reference member is here}}
#endif

// [C++1z] no virtual, protected, or private base classes, ...
struct NonAggr5b : virtual Aggr {}; // expected-note 3{{candidate}}
NonAggr5b na5b = { b }; // expected-error {{no matching constructor}}
struct NonAggr5c : NonAggr5b {}; // expected-note 3{{candidate}}
NonAggr5c na5c = { b }; // expected-error {{no matching constructor}}
struct NonAggr5d : protected Aggr {}; // expected-note 3{{candidate}}
NonAggr5d na5d = { b }; // expected-error {{no matching constructor}}
struct NonAggr5e : private Aggr {}; // expected-note 3{{candidate}}
NonAggr5e na5e = { b }; // expected-error {{no matching constructor}}
class NonAggr5f : Aggr {}; // expected-note 3{{candidate}}
NonAggr5f na5f = { b }; // expected-error {{no matching constructor}}

// [C++1z] (the base class need not itself be an aggregate)
struct MaybeAggr5g : NonAggr1a {};
MaybeAggr5g ma5g1 = { 1 };
MaybeAggr5g ma5g2 = { {1, 2} };
MaybeAggr5g ma5g3 = {};
#if __cplusplus <= 201402L
// expected-error@-4 {{no matching constructor}} // expected-note@-5 3{{candidate}}
// expected-error@-4 {{no matching constructor}} // expected-note@-6 3{{candidate}}
// expected-error@-4 {{implicitly-deleted default constructor}} expected-note@-7 {{no default constructor}}
#else
// expected-error@-8 {{no viable conversion from 'int' to 'NonAggr1a'}} expected-note@19 2{{candidate}}
// (ok)
// expected-error@-8 {{no matching constructor}} expected-note@19 2{{candidate}} expected-note@20 {{candidate}}
#endif

// and no virtual functions.
struct NonAggr6 { // expected-note 3 {{candidate constructor}}
  virtual void f();
  int n;
};
NonAggr6 na6 = { 42 }; // expected-error {{no matching constructor for initialization of 'NonAggr6'}}

struct NonAggr7 : NonAggr6 { // expected-note 3 {{candidate constructor}}
  int n;
};
NonAggr7 na7 = {{}, 42}; // expected-error {{no matching constructor for initialization of 'NonAggr7'}}

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

struct ExplicitDefaultedAggr {
  int n;
  explicit ExplicitDefaultedAggr() = default; // expected-note {{candidate}}
  ExplicitDefaultedAggr(const ExplicitDefaultedAggr &) = default; // expected-note {{candidate}}
  ExplicitDefaultedAggr(ExplicitDefaultedAggr &&) = default; // expected-note {{candidate}}
};
ExplicitDefaultedAggr eda = { 42 }; // expected-error {{no matching constructor}}
ExplicitDefaultedAggr eda2{};

struct DefaultedBase {
  int n;
  DefaultedBase() = default;
  DefaultedBase(DefaultedBase const&) = default;
  DefaultedBase(DefaultedBase &&) = default;
};

struct InheritingConstructors : DefaultedBase { // expected-note 3 {{candidate}}
  using DefaultedBase::DefaultedBase;
};
InheritingConstructors ic = { 42 }; // expected-error {{no matching constructor}}

struct NonInheritingConstructors : DefaultedBase {}; // expected-note 0+ {{candidate}}
NonInheritingConstructors nic = { 42 };
#if __cplusplus <= 201402L
// expected-error@-2 {{no matching constructor}}
#endif

struct NonAggrBase {
  NonAggrBase(int) {}
};
struct HasNonAggrBase : NonAggrBase {}; // expected-note 0+ {{candidate}}
HasNonAggrBase hnab = {42};
#if __cplusplus <= 201402L
// expected-error@-2 {{no matching constructor}}
#endif
