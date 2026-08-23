//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// <hazard_pointer>

// The Mandates of try_protect/reset_protection/retire: T must be hazard-protectable, i.e. a class type
// with exactly one hazard_pointer_obj_base<T, D> base, public and non-virtual, and no other
// hazard_pointer_obj_base<T2, D2> base. Also checks the D requirements static_assert.

#include <hazard_pointer>
#include <atomic>

// Positive cases: no diagnostics.
struct Good : std::hazard_pointer_obj_base<Good> {};
static_assert(std::__hazard_protectable<Good>);
struct GoodDeleter {
  void operator()(struct GoodD*) const noexcept {}
};
struct GoodD : std::hazard_pointer_obj_base<GoodD, GoodDeleter> {};
static_assert(std::__hazard_protectable<GoodD>);
struct Other {};
struct GoodMulti : Other, std::hazard_pointer_obj_base<GoodMulti> {}; // an unrelated base is fine
static_assert(std::__hazard_protectable<GoodMulti>);

// Negative cases for the concept.
struct NoBase {};
static_assert(!std::__hazard_protectable<NoBase>);
struct PrivateBase : private std::hazard_pointer_obj_base<PrivateBase> {};
static_assert(!std::__hazard_protectable<PrivateBase>);
struct ProtectedBase : protected std::hazard_pointer_obj_base<ProtectedBase> {};
static_assert(!std::__hazard_protectable<ProtectedBase>);
struct VirtualBase : virtual std::hazard_pointer_obj_base<VirtualBase> {};
static_assert(!std::__hazard_protectable<VirtualBase>);
struct TwoBases : std::hazard_pointer_obj_base<TwoBases>, std::hazard_pointer_obj_base<Good> {};
static_assert(!std::__hazard_protectable<TwoBases>);
struct Base : std::hazard_pointer_obj_base<Base> {};
struct Derived : Base {}; // Derived's obj_base base is hazard_pointer_obj_base<Base>, not <Derived>
static_assert(!std::__hazard_protectable<Derived>);
static_assert(!std::__hazard_protectable<const Good>); // strict reading of [saferecl.hp.general]/2
static_assert(!std::__hazard_protectable<int>);

void diagnostics() {
  std::hazard_pointer h;
  std::atomic<NoBase*> src{nullptr};
  // The static_assert is the diagnostic under test; the follow-on errors from the body are expected too:
  // try_protect() calls reset_protection(), whose own Mandate fires, and __node_of() has no viable
  // derived-to-base conversion when T has no hazard_pointer_obj_base base at all.
  (void)h.protect(src); // expected-error@*:* 2 {{static assertion failed}}
  // expected-error@*:* {{no matching function for call to '__node_of'}}
  std::atomic<Derived*> src2{nullptr};
  Derived* p = nullptr;
  // Derived converts to hazard_pointer_obj_base<Base, D>, so __node_of() itself is fine here.
  (void)h.try_protect(p, src2);                         // expected-error@*:* 2 {{static assertion failed}}
  h.reset_protection(static_cast<const int*>(nullptr)); // expected-error@*:* {{static assertion failed}}
  // expected-error@*:* {{no matching function for call to '__node_of'}}

  // protect() rejects a T that isn't itself hazard-protectable, even when T is merely const-qualified:
  // const Good has no hazard_pointer_obj_base<const Good, D> base.
  std::atomic<const Good*> csrc{nullptr};
  (void)h.protect(csrc); // expected-error@*:* {{static assertion failed}}

  // retire() is inherited unambiguously from Base; the instantiated specialization is
  // hazard_pointer_obj_base<Base, D>, and Base -- unlike Derived -- IS hazard-protectable, so this is
  // well-formed: no diagnostic.
  Derived d;
  d.retire();
}

// retire()'s Mandate fires when the hazard_pointer_obj_base<T, D> specialization itself names a T that
// is not hazard-protectable (as opposed to the most-derived object being retired); the downcast in the
// reclamation function then fails as well.
struct Mismatched : std::hazard_pointer_obj_base<NoBase> {};
void bad_retire() {
  Mismatched m;
  m.retire(); // expected-error@*:* {{static assertion failed}}
  // expected-error@*:* {{not related by inheritance}}
}

// D requirements.
struct NotDefaultConstructible {
  NotDefaultConstructible(int);
  void operator()(struct BadD*) const noexcept {}
};
struct BadD : std::hazard_pointer_obj_base<BadD, NotDefaultConstructible> {
}; // expected-error@*:* {{static assertion failed}}
struct NotMoveAssignable {
  NotMoveAssignable()                               = default;
  NotMoveAssignable& operator=(NotMoveAssignable&&) = delete;
  void operator()(struct BadMoveD*) const noexcept {}
};
struct BadMoveD : std::hazard_pointer_obj_base<BadMoveD, NotMoveAssignable> {
}; // expected-error@*:* {{static assertion failed}}
