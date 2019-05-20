// RUN: %check_clang_tidy %s performance-move-constructor-init,modernize-pass-by-value %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: modernize-pass-by-value.ValuesOnly, value: 1}]}' \
// RUN: -- -isystem %S/Inputs/Headers

#include <s.h>

// CHECK-FIXES: #include <utility>

template <class T> struct remove_reference      {typedef T type;};
template <class T> struct remove_reference<T&>  {typedef T type;};
template <class T> struct remove_reference<T&&> {typedef T type;};

template <typename T>
typename remove_reference<T>::type&& move(T&& arg) {
  return static_cast<typename remove_reference<T>::type&&>(arg);
}

struct C {
  C() = default;
  C(const C&) = default;
};

struct B {
  B() {}
  B(const B&) {}
  B(B &&) {}
};

struct D : B {
  D() : B() {}
  D(const D &RHS) : B(RHS) {}
  // CHECK-NOTES: :[[@LINE+3]]:16: warning: move constructor initializes base class by calling a copy constructor [performance-move-constructor-init]
  // CHECK-NOTES: 26:3: note: copy constructor being called
  // CHECK-NOTES: 27:3: note: candidate move constructor here
  D(D &&RHS) : B(RHS) {}
};

struct E : B {
  E() : B() {}
  E(const E &RHS) : B(RHS) {}
  E(E &&RHS) : B(move(RHS)) {} // ok
};

struct F {
  C M;

  F(F &&) : M(C()) {} // ok
};

struct G {
  G() = default;
  G(const G&) = default;
  G(G&&) = delete;
};

struct H : G {
  H() = default;
  H(const H&) = default;
  H(H &&RHS) : G(RHS) {} // ok
};

struct I {
  I(const I &) = default; // suppresses move constructor creation
};

struct J : I {
  J(J &&RHS) : I(RHS) {} // ok
};

struct K {}; // Has implicit copy and move constructors, is trivially copyable
struct L : K {
  L(L &&RHS) : K(RHS) {} // ok
};

struct M {
  B Mem;
  // CHECK-NOTES: :[[@LINE+1]]:16: warning: move constructor initializes class member by calling a copy constructor [performance-move-constructor-init]
  M(M &&RHS) : Mem(RHS.Mem) {}
  // CHECK-NOTES: 26:3: note: copy constructor being called
  // CHECK-NOTES: 27:3: note: candidate move constructor here
};

struct N {
  B Mem;
  N(N &&RHS) : Mem(move(RHS.Mem)) {}
};

struct O {
  O(O&& other) : b(other.b) {} // ok
  const B b;
};

struct P {
  P(O&& other) : b(other.b) {} // ok
  B b;
};

struct Movable {
  Movable(Movable &&) = default;
  Movable(const Movable &) = default;
  Movable &operator=(const Movable &) = default;
  ~Movable() {}
};

struct TriviallyCopyable {
  TriviallyCopyable() = default;
  TriviallyCopyable(TriviallyCopyable &&) = default;
  TriviallyCopyable(const TriviallyCopyable &) = default;
};

struct Positive {
  Positive(Movable M) : M_(M) {}
  // CHECK-NOTES: [[@LINE-1]]:12: warning: pass by value and use std::move [modernize-pass-by-value]
  // CHECK-FIXES: Positive(Movable M) : M_(std::move(M)) {}
  Movable M_;
};

struct NegativeMultipleInitializerReferences {
  NegativeMultipleInitializerReferences(Movable M) : M_(M), n_(M) {}
  Movable M_;
  Movable n_;
};

struct NegativeReferencedInConstructorBody {
  NegativeReferencedInConstructorBody(Movable M) : M_(M) { M_ = M; }
  Movable M_;
};

struct NegativeParamTriviallyCopyable {
  NegativeParamTriviallyCopyable(TriviallyCopyable T) : T_(T) {}
  NegativeParamTriviallyCopyable(int I) : I_(I) {}

  TriviallyCopyable T_;
  int I_;
};

struct NegativeNotPassedByValue {
  // This const ref constructor isn't warned about because the ValuesOnly option is set.
  NegativeNotPassedByValue(const Movable &M) : M_(M) {}
  NegativeNotPassedByValue(const Movable M) : M_(M) {}
  NegativeNotPassedByValue(Movable &M) : M_(M) {}
  NegativeNotPassedByValue(Movable *M) : M_(*M) {}
  NegativeNotPassedByValue(const Movable *M) : M_(*M) {}
  Movable M_;
};

struct Immovable {
  Immovable(const Immovable &) = default;
  Immovable(Immovable &&) = delete;
};

struct NegativeImmovableParameter {
  NegativeImmovableParameter(Immovable I) : I_(I) {}
  Immovable I_;
};
