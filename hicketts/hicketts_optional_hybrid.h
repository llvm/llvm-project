#ifndef HICKETTS_OPTIONAL_HYBRID_H_
#define HICKETTS_OPTIONAL_HYBRID_H_

/// A custom optional-like type wired for the *hybrid* attribute scheme, written
/// to exercise ALL THREE ways the unchecked-optional-access model can learn a
/// method's role, side by side:
///
///   (A) AUTOMAGIC    -- a method with a std::optional name (value, has_value,
///                       reset, ...) is recognised STRUCTURALLY once the class
///                       carries [[clang::analyze_as_class("std::optional")]].
///                       No per-method annotation needed.
///   (B) NAME-MAPPED  -- a differently-named method is mapped to a std op by
///                       [[clang::analyze_as_method("value")]] (name-only; no
///                       signature strings -- this base does not match those).
///   (C) BEHAVIOURAL  -- a method carries a role verb (engaged / disengaged /
///                       assume_engaged / test_engaged). The natural fit for
///                       custom-named methods with no std analog, and the ONLY
///                       path for the same-name/different-outcome constructors
///                       and assignments (signatures can't disambiguate them).
///
/// Precedence when more than one could apply to a single decl: explicit role (C)
/// wins over name-map (B) wins over automagic (A) -- implemented as match-switch
/// ordering in the model (role cases registered first). Where a method already
/// works via (A)/(B), a comment notes that the behavioural verb would express
/// the same thing explicitly.
///
/// Real + inline: analyze_as_class / analyze_as_method, gsl::Owner, lifetimebound.
/// Only the role verbs are macro-guarded (-DHO_ROLES) -- they are not
/// implemented yet, so (C) cases are silent until then; (A) and (B) work today.
namespace mylib {

// The proposed role verbs -- the only guarded layer (not implemented yet).
// Shared spelling with hicketts_vector_hybrid.h.
#ifdef HO_ROLES
#define HO_ENGAGED        [[clang::engaged]]
#define HO_DISENGAGED     [[clang::disengaged]]
#define HO_ASSUME_ENGAGED [[clang::assume_engaged]]
#define HO_TEST_ENGAGED   [[clang::test_engaged]]
#else
#define HO_ENGAGED
#define HO_DISENGAGED
#define HO_ASSUME_ENGAGED
#define HO_TEST_ENGAGED
#endif

struct nothing_t {
  constexpr explicit nothing_t() {}
};

constexpr nothing_t nothing;

template <typename T>
class [[gsl::Owner]] [[clang::analyze_as_class("std::optional")]] HickettsOptional {
  T *storage_ = nullptr;

public:
  // === Constructors: (C) BEHAVIOURAL only ==================================
  // Same 1-arg shape, different outcome; there is no std name to pick up and
  // signatures don't disambiguate, so the role on each decl is the only path.
  HO_DISENGAGED constexpr HickettsOptional() noexcept {}
  HO_DISENGAGED constexpr HickettsOptional(nothing_t) noexcept {}
  HO_ENGAGED constexpr HickettsOptional(T) noexcept {}

  // Copy / move: no role -- default record-copy propagates the predicate.
  HickettsOptional(const HickettsOptional &) = default;
  HickettsOptional(HickettsOptional &&) = default;

  // === Accessors (require engaged) -- one per recognition path =============

  // (A) AUTOMAGIC: std name, no annotation -- recognised via analyze_as_class.
  //     A behavioural [[clang::assume_engaged]] would express the same thing.
  const T &value() const & [[clang::lifetimebound]] { return *storage_; }
  T &value() & [[clang::lifetimebound]] { return *storage_; }

  // (B) NAME-MAPPED: custom name -> std value() by name.
  //     A behavioural [[clang::assume_engaged]] would also work here.
  [[clang::analyze_as_method("value")]]
  const T &unwrap() const & [[clang::lifetimebound]] { return *storage_; }
  [[clang::analyze_as_method("value")]]
  T &unwrap() & [[clang::lifetimebound]] { return *storage_; }

  // (C) BEHAVIOURAL: custom name, no std analog -- the role is the natural fit.
  HO_ASSUME_ENGAGED const T &deref() const & [[clang::lifetimebound]] { return *storage_; }
  HO_ASSUME_ENGAGED T &deref() & [[clang::lifetimebound]] { return *storage_; }

  const T *operator->() const [[clang::lifetimebound]] { return storage_; }
  T *operator->() [[clang::lifetimebound]] { return storage_; }

  // === Queries (narrow the predicate) ======================================

  // (A) AUTOMAGIC: std names. A behavioural [[clang::test_engaged]] would also work.
  constexpr bool has_value() const noexcept { return storage_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return storage_ != nullptr; }

  // (B) NAME-MAPPED: custom name -> std has_value.
  //     A behavioural [[clang::test_engaged]] would also work here.
  [[clang::analyze_as_method("has_value")]]
  constexpr bool isPresent() const noexcept { return storage_ != nullptr; }

  // (C) BEHAVIOURAL: custom-named query, role-only -- the live test_engaged
  //     trial. Not a std name and not name-mapped, so ONLY the role can teach the
  //     model that isEngaged() narrows the predicate. Today a guarded value()
  //     still warns (false positive); with -DHO_ROLES it becomes safe.
  HO_TEST_ENGAGED constexpr bool isEngaged() const noexcept { return storage_ != nullptr; }

  // value_or: always safe, no precondition.
  template <typename U>
  constexpr T unwrapOr(U &&fallback) const & {
    return storage_ ? *storage_ : static_cast<T>(fallback);
  }

  // === State transitions ===================================================

  // (B) NAME-MAPPED: emplace by name -> engaged.
  //     A behavioural [[clang::engaged]] would also work here.
  template <typename... Args>
  [[clang::analyze_as_method("emplace")]]
  T &construct(Args &&...args) { return *storage_; }

  // (B) NAME-MAPPED: reset by name -> disengaged.
  //     A behavioural [[clang::disengaged]] would also work here.
  [[clang::analyze_as_method("reset")]]
  void clear() noexcept { storage_ = nullptr; }

  // swap: no role/mapping -> predicate becomes unknown.
  void exchange(HickettsOptional &other) noexcept {
    T *tmp = storage_;
    storage_ = other.storage_;
    other.storage_ = tmp;
  }

  // === Assignment: (C) BEHAVIOURAL only (same disambiguation story as ctors) =
  template <typename U>
  HO_ENGAGED HickettsOptional &operator=(const U &u) { return *this; }

  HO_DISENGAGED HickettsOptional &operator=(nothing_t) {
    storage_ = nullptr;
    return *this;
  }
};

} // namespace mylib

#endif // HICKETTS_OPTIONAL_HYBRID_H_
