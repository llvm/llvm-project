// RUN: %clang_cc1 -std=c++1z -verify %s

struct Noncopyable {
  Noncopyable();
  Noncopyable(const Noncopyable &) = delete; // expected-note 1+{{deleted}}
  virtual ~Noncopyable();
};
struct Derived : Noncopyable {};
struct NoncopyableAggr {
  Noncopyable nc;
};
struct Indestructible {
  Indestructible();
  ~Indestructible() = delete; // expected-note 1+{{deleted}}
};
struct Incomplete; // expected-note 1+{{declar}}

Noncopyable make(int kind = 0) {
  switch (kind) {
  case 0: return {};
  case 1: return Noncopyable();
  case 2: return Noncopyable{};
  case 3: return make();
  }
  __builtin_unreachable();
}

Indestructible make_indestructible();
Incomplete make_incomplete(); // expected-note 1+{{here}}

void take(Noncopyable nc) {}

Noncopyable nrvo() {
  Noncopyable nrvo;
  return nrvo; // expected-error {{deleted constructor}}
}

Noncopyable nc1 = make();
Noncopyable nc2 = Noncopyable();
Noncopyable nc3 = Derived(); // expected-error {{deleted constructor}}

NoncopyableAggr nca1 = NoncopyableAggr{};
NoncopyableAggr nca2 = NoncopyableAggr{{}};
NoncopyableAggr nca3 = NoncopyableAggr{NoncopyableAggr{Noncopyable()}};

void test_expressions(bool b) {
  auto lambda = [a = make()] {};

  take({});
  take(Noncopyable());
  take(Noncopyable{});
  take(make());

  Noncopyable &&dc1 = dynamic_cast<Noncopyable&&>(Noncopyable());
  Noncopyable &&dc2 = dynamic_cast<Noncopyable&&>(nc1);
  Noncopyable &&dc3 = dynamic_cast<Noncopyable&&>(Derived());

  Noncopyable sc1 = static_cast<Noncopyable>(Noncopyable());
  Noncopyable sc2 = static_cast<Noncopyable>(nc1); // expected-error {{deleted}}
  Noncopyable sc3 = static_cast<Noncopyable&&>(Noncopyable()); // expected-error {{deleted}}
  Noncopyable sc4 = static_cast<Noncopyable>(static_cast<Noncopyable&&>(Noncopyable())); // expected-error {{deleted}}

  Noncopyable cc1 = (Noncopyable)Noncopyable();
  Noncopyable cc2 = (Noncopyable)Derived(); // expected-error {{deleted}}

  Noncopyable fc1 = Noncopyable(Noncopyable());
  Noncopyable fc2 = Noncopyable(Derived()); // expected-error {{deleted}}

  // We must check for a complete type for every materialized temporary. (Note
  // that in the special case of the top level of a decltype, no temporary is
  // materialized.)
  make_incomplete(); // expected-error {{incomplete}}
  make_incomplete().a; // expected-error {{incomplete}}
  make_incomplete().*(int Incomplete::*)nullptr; // expected-error {{incomplete}}
  dynamic_cast<Incomplete&&>(make_incomplete()); // expected-error {{incomplete}}
  const_cast<Incomplete&&>(make_incomplete()); // expected-error {{incomplete}}

  sizeof(Indestructible{}); // expected-error {{deleted}}
  sizeof(make_indestructible()); // expected-error {{deleted}}
  sizeof(make_incomplete()); // expected-error {{incomplete}}
  typeid(Indestructible{}); // expected-error {{deleted}}
  typeid(make_indestructible()); // expected-error {{deleted}}
  typeid(make_incomplete()); // expected-error {{incomplete}}

  // FIXME: The first two cases here are now also valid in C++17 onwards.
  using I = decltype(Indestructible()); // expected-error {{deleted}}
  using I = decltype(Indestructible{}); // expected-error {{deleted}}
  using I = decltype(make_indestructible());
  using J = decltype(make_incomplete());

  Noncopyable cond1 = b ? Noncopyable() : make();
  Noncopyable cond2 = b ? Noncopyable() : Derived(); // expected-error {{incompatible}}
  Noncopyable cond3 = b ? Derived() : Noncopyable(); // expected-error {{incompatible}}
  Noncopyable cond4 = b ? Noncopyable() : nc1; // expected-error {{deleted}}
  Noncopyable cond5 = b ? nc1 : Noncopyable(); // expected-error {{deleted}}
  // Could convert both to an xvalue of type Noncopyable here, but we're not
  // permitted to consider that.
  Noncopyable &&cond6 = b ? Noncopyable() : Derived(); // expected-error {{incompatible}}
  Noncopyable &&cond7 = b ? Derived() : Noncopyable(); // expected-error {{incompatible}}
  // Could convert both to a const lvalue of type Noncopyable here, but we're
  // not permitted to consider that, either.
  const Noncopyable cnc;
  const Noncopyable &cond8 = b ? cnc : Derived(); // expected-error {{incompatible}}
  const Noncopyable &cond9 = b ? Derived() : cnc; // expected-error {{incompatible}}

  extern const volatile Noncopyable make_cv();
  Noncopyable cv_difference1 = make_cv();
  const volatile Noncopyable cv_difference2 = make();
}

template<typename T> struct ConversionFunction { operator T(); };
Noncopyable cf1 = ConversionFunction<Noncopyable>();
Noncopyable cf2 = ConversionFunction<Noncopyable&&>(); // expected-error {{deleted}}
Noncopyable cf3 = ConversionFunction<const volatile Noncopyable>();
const volatile Noncopyable cf4 = ConversionFunction<Noncopyable>();
Noncopyable cf5 = ConversionFunction<Derived>(); // expected-error {{deleted}}

struct AsMember {
  Noncopyable member;
  AsMember() : member(make()) {}
};
// FIXME: DR (no number yet): we still get a copy for base or delegating construction.
struct AsBase : Noncopyable {
  AsBase() : Noncopyable(make()) {} // expected-error {{deleted}}
};
struct AsDelegating final {
  AsDelegating(const AsDelegating &) = delete; // expected-note {{deleted}}
  static AsDelegating make(int);

  // The base constructor version of this is problematic; the complete object
  // version would be OK. Perhaps we could allow copy omission here for final
  // classes?
  AsDelegating(int n) : AsDelegating(make(n)) {} // expected-error {{deleted}}
};
