#ifndef HICKETTS_OPTIONAL_HYBRID_H_
#define HICKETTS_OPTIONAL_HYBRID_H_

/// A custom optional-like type wired for the *hybrid* attribute scheme.
///
/// Three cooperating layers sit on the SAME type, each carrying a different
/// KIND of knowledge (see architecture.md, plan_general.md, why_class_comparison.md):
///
///   L1  IDENTITY   [[clang::analyze_as_class("std::optional")]]
///         "I behave like std::optional." A single class-level declaration that
///         anchors the type so OTHER clang-tidy checks reuse their built-in,
///         per-std-class knowledge by identity (why_class_comparison.md). No
///         per-method name-maps: on optional nothing else consumes them, and the
///         dataflow model is driven by the L2 roles below instead.
///
///   L2  BEHAVIOURAL ROLES   [[clang::engaged/disengaged/test_engaged/assume_engaged]]
///         The single per-object boolean predicate the flow-sensitive
///         bugprone-unchecked-optional-access model tracks. The predicate has no
///         name because it is INERT to the model -- the model tracks one opaque
///         bit, so "empty" and "disengaged" are the same action (set-false) and
///         a name would only matter if a type had >1 predicate. What is NOT inert
///         is polarity; optional's API here is uniformly positive (has_value,
///         value), so only the positive verbs appear:
///           engaged        -> establish the bit true   (value ctor, emplace, =value)
///           disengaged     -> establish the bit false  (nullopt ctor, reset, =nullopt)
///           test_engaged   -> branch-sensitive read    (has_value, operator bool)
///           assume_engaged -> precondition: warn if not established (value/*/->)
///         (A negative-polarity test/assume -- e.g. an isEmpty()-style query, or
///         vector's empty()/front() -- is where polarity, not the name, would
///         reappear. Out of scope for this fixture.)
///         PROPOSED, not yet implemented, so this is the one layer left behind a
///         macro: -DHO_ROLES turns it on for a roles-off/roles-on baseline.
///
///   L3  LIFETIME   [[gsl::Owner]] / [[clang::lifetimebound]]
///         Real, shipping-today attributes. An optional OWNS its T; unwrap()/
///         deref()/operator-> hand out handles INTO it, so a handle taken from
///         a *temporary* optional dangles (-Wdangling). This is the RELATIONAL
///         hazard the predicate layer deliberately does NOT model.
///
/// L1 and L3 are inline (they exist today, ignored where unsupported). Only L2
/// is guarded, because those attributes do not exist yet:
///   (default)    L1 + L3     -> -Wdangling live; dataflow model idle until roles
///   -DHO_ROLES   + L2        -> needs the proposed role attributes
namespace mylib {

// The only guarded layer: the proposed per-object-state roles, which do not
// exist yet. Elided by default so the header builds without -Wunknown-attributes;
// -DHO_ROLES emits them once they are implemented. No predicate argument -- the
// type has a single, model-opaque predicate.
#ifdef HO_ROLES
#define HO_ENGAGED        [[clang::engaged]]
#define HO_DISENGAGED     [[clang::disengaged]]
#define HO_TEST_ENGAGED   [[clang::test_engaged]]
#define HO_ASSUME_ENGAGED [[clang::assume_engaged]]
#else
#define HO_ENGAGED
#define HO_DISENGAGED
#define HO_TEST_ENGAGED
#define HO_ASSUME_ENGAGED
#endif

struct nothing_t {
  constexpr explicit nothing_t() {}
};

constexpr nothing_t nothing;

template <typename T>
class [[gsl::Owner]] [[clang::analyze_as_class("std::optional")]] HickettsOptional {
  T *storage_ = nullptr;

public:
  // Default ctor -> disengaged.
  HO_DISENGAGED
  constexpr HickettsOptional() noexcept {}

  // Nullopt-style ctor. SAME 1-arg shape as the value ctor below, DIFFERENT
  // outcome. The role on THIS decl disambiguates -- overload resolution picks it
  // for HickettsOptional(nothing); no signature string, no std header.
  HO_DISENGAGED
  constexpr HickettsOptional(nothing_t) noexcept {}

  // Value ctor -> engaged. Same 1-arg shape; the role on the OTHER decl is what
  // tells them apart (the whole point of the hybrid).
  HO_ENGAGED
  constexpr HickettsOptional(T) noexcept {}

  // Copy / move: no role -- the framework's default record-copy propagates the
  // predicate from the source.
  HickettsOptional(const HickettsOptional &) = default;
  HickettsOptional(HickettsOptional &&) = default;

  // value()/unwrap()/deref(): precondition engaged (L2) AND return a handle
  // INTO *this (L3). One method, two independent hazards:
  //   L2 assume_engaged -> unchecked-optional-access diagnostic
  //   L3 lifetimebound  -> -Wdangling when *this is a temporary
  HO_ASSUME_ENGAGED const T &unwrap() const & [[clang::lifetimebound]] { return *storage_; }
  HO_ASSUME_ENGAGED T &unwrap() & [[clang::lifetimebound]] { return *storage_; }

  HO_ASSUME_ENGAGED const T &deref() const & [[clang::lifetimebound]] { return *storage_; }
  HO_ASSUME_ENGAGED T &deref() & [[clang::lifetimebound]] { return *storage_; }

  const T *operator->() const [[clang::lifetimebound]] { return storage_; }
  T *operator->() [[clang::lifetimebound]] { return storage_; }

  // Queries: branch-sensitive read, positive polarity.
  HO_TEST_ENGAGED constexpr bool has_value() const noexcept { return storage_ != nullptr; }
  HO_TEST_ENGAGED constexpr explicit operator bool() const noexcept { return storage_ != nullptr; }
  HO_TEST_ENGAGED constexpr bool isPresent() const noexcept { return storage_ != nullptr; }

  // value_or: always safe, no precondition.
  template <typename U>
  constexpr T unwrapOr(U &&fallback) const & {
    return storage_ ? *storage_ : static_cast<T>(fallback);
  }

  // emplace -> engaged.
  template <typename... Args>
  HO_ENGAGED T &construct(Args &&...args) { return *storage_; }

  // reset -> disengaged.
  HO_DISENGAGED void clear() noexcept { storage_ = nullptr; }

  // swap: leaves both operands' predicates unknown -- no role (havoc), handled
  // by the model's default.
  void exchange(HickettsOptional &other) noexcept {
    T *tmp = storage_;
    storage_ = other.storage_;
    other.storage_ = tmp;
  }

  // Assignment from a value -> engaged.
  template <typename U>
  HO_ENGAGED HickettsOptional &operator=(const U &u) { return *this; }

  // Nullopt-style assignment -> disengaged. Same disambiguation story as the
  // ctors: the role on this decl routes it.
  HO_DISENGAGED HickettsOptional &operator=(nothing_t) {
    storage_ = nullptr;
    return *this;
  }
};

} // namespace mylib

#endif // HICKETTS_OPTIONAL_HYBRID_H_
