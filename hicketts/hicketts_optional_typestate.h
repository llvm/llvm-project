#ifndef HICKETTS_OPTIONAL_TYPESTATE_H_
#define HICKETTS_OPTIONAL_TYPESTATE_H_

/// Optional-only fixture for the hybrid attribute scheme, built entirely from
/// attributes Clang ALREADY SHIPS. Nothing here is a proposal: every attribute
/// below parses on stock clang today.
///
/// Three ways the model can learn a method's role:
///
///   (A) AUTOMAGIC    -- a std-named method (value, has_value, reset, ...) is
///                       recognised STRUCTURALLY once the class carries
///                       [[clang::analyze_as_class("std::optional")]].
///                       No per-method annotation needed.
///   (B) NAME-MAPPED  -- a differently-named method is mapped to a known op by
///                       [[clang::analyze_as_method("value")]].
///   (C) TYPESTATE    -- the decl declares its own behaviour using Clang's
///                       consumed-annotation family (Attr.td:4320-4412):
///
///         [[clang::return_typestate(unconsumed)]]  ctor/function -> engaged
///         [[clang::return_typestate(consumed)]]    ctor/function -> empty
///         [[clang::set_typestate(unconsumed)]]     method sets engaged
///         [[clang::set_typestate(consumed)]]       method sets empty
///         [[clang::callable_when("unconsumed")]]   method requires engaged
///         [[clang::test_typestate(unconsumed)]]    method queries engaged
///         [[clang::test_typestate(consumed)]]      method queries empty
///
///       unconsumed == engaged == has_value()
///       consumed   == disengaged == empty
///
/// THREE THINGS TO NOTICE, all of them RFC material:
///
/// 1. NO MACRO GUARD, because there is nothing unimplemented to guard. These
///    attributes are real, shipped, and documented (ConsumableDocs /
///    SetTypestateDocs / TestTypestateDocs / CallableWhenDocs in AttrDocs.td).
///    The optional model simply ignores them until the backend reads them.
///
/// 2. CONSTRUCTORS USE return_typestate, NOT set_typestate. Confirmed at
///    clang/lib/Analysis/Consumed.cpp:773 -- VisitCXXConstructExpr reads
///    ReturnTypestateAttr off the constructor. set_typestate on a ctor parses
///    (CXXConstructorDecl is a CXXMethodDecl) but the existing analysis never
///    looks at it.
///
/// 3. ARGUMENT SPELLING IS NOT UNIFORM. callable_when takes QUOTED strings
///    (VariadicEnumArgument, is_string=true); consumable / set_typestate /
///    test_typestate / return_typestate take BARE identifiers
///    (EnumArgument, is_string=false). Getting this wrong is a parse error.
///
/// Build/run instructions are in test_hicketts_optional_typestate.cpp.

namespace mylib {

struct nothing_t {
  constexpr explicit nothing_t() {}
};

constexpr nothing_t nothing;

/// TWO class-level attributes, doing two different jobs:
///   consumable(consumed)  -- enrols the class in the typestate vocabulary and
///                            sets the default state of a constructed object.
///                            REQUIRED: ConsumableDocs says every class using
///                            any typestate annotation must be marked, or the
///                            existing analysis warns.
///   analyze_as_class(...) -- enrols the class in the optional model.
///
/// OPEN QUESTION for the RFC: should analyze_as_class imply consumable, or must
/// library authors write both? Two attributes to say one thing is a wart a
/// reviewer will notice.
template <typename T>
class [[gsl::Owner]]
      [[clang::consumable(consumed)]]
      [[clang::analyze_as_class("std::optional")]] HickettsOptional {
  T *storage_ = nullptr;

public:
  // === Constructors: (C) TYPESTATE, via return_typestate ===================
  // Same 1-arg shape, opposite outcomes -- the case name-mapping cannot express,
  // because both constructors share a name and it isn't really a name at all.

  // Deliberately UNANNOTATED: Consumed.cpp:777 defaults a default-constructor
  // to CS_Consumed. Tests whether the optional model inherits that convention
  // for free or needs the attribute spelled out.
  constexpr HickettsOptional() noexcept {}

  [[clang::return_typestate(consumed)]]
  constexpr HickettsOptional(nothing_t) noexcept {}

  [[clang::return_typestate(unconsumed)]]
  constexpr HickettsOptional(T) noexcept {}

  // Copy / move left unannotated on purpose -- see the divergence probe in the
  // test file. The existing analysis hardwires move-source -> consumed
  // (Consumed.cpp:779), which is WRONG for optional: a moved-from std::optional
  // is still engaged. The optional model must not inherit that rule.
  HickettsOptional(const HickettsOptional &) = default;
  HickettsOptional(HickettsOptional &&) = default;

  // === Accessors (require engaged) =========================================

  // (A) AUTOMAGIC: std name, recognised via analyze_as_class.
  const T &value() const & [[clang::lifetimebound]] { return *storage_; }
  T &value() & [[clang::lifetimebound]] { return *storage_; }

  // (B) NAME-MAPPED: custom name -> std value().
  [[clang::analyze_as_method("value")]]
  const T &unwrap() const & [[clang::lifetimebound]] { return *storage_; }
  [[clang::analyze_as_method("value")]]
  T &unwrap() & [[clang::lifetimebound]] { return *storage_; }

  // (C) TYPESTATE: custom name, no std analog. NOTE the quoted argument.
  [[clang::callable_when("unconsumed")]]
  const T &deref() const & [[clang::lifetimebound]] { return *storage_; }
  [[clang::callable_when("unconsumed")]]
  T &deref() & [[clang::lifetimebound]] { return *storage_; }

  const T *operator->() const [[clang::lifetimebound]] { return storage_; }
  T *operator->() [[clang::lifetimebound]] { return storage_; }

  // === Queries (narrow the predicate) ======================================

  // (A) AUTOMAGIC: std names.
  constexpr bool has_value() const noexcept { return storage_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return storage_ != nullptr; }

  // (B) NAME-MAPPED.
  [[clang::analyze_as_method("has_value")]]
  constexpr bool isPresent() const noexcept { return storage_ != nullptr; }

  // (C) TYPESTATE, positive polarity.
  [[clang::test_typestate(unconsumed)]]
  constexpr bool isEngaged() const noexcept { return storage_ != nullptr; }

  // (C) TYPESTATE, NEGATIVE polarity -- the same attribute, the other enum
  // value. One attribute covers both directions.
  [[clang::test_typestate(consumed)]]
  constexpr bool isEmpty() const noexcept { return storage_ == nullptr; }

  // No precondition: always safe.
  template <typename U>
  constexpr T unwrapOr(U &&fallback) const & {
    return storage_ ? *storage_ : static_cast<T>(fallback);
  }

  // === State transitions ===================================================

  // (B) NAME-MAPPED: emplace by name -> engaged.
  template <typename... Args>
  [[clang::analyze_as_method("emplace")]]
  T &construct(Args &&...args) { return *storage_; }

  // (C) TYPESTATE: custom name -> engaged.
  [[clang::set_typestate(unconsumed)]]
  T &install(T v) { return *storage_; }

  // (C) TYPESTATE: custom name -> empty.
  [[clang::set_typestate(consumed)]]
  void clear() noexcept { storage_ = nullptr; }

  // Unannotated -> predicate becomes unknown.
  void exchange(HickettsOptional &other) noexcept {
    T *tmp = storage_;
    storage_ = other.storage_;
    other.storage_ = tmp;
  }

  // === Assignment: (C) TYPESTATE, via set_typestate ========================
  // Same disambiguation story as the constructors.
  template <typename U>
  [[clang::set_typestate(unconsumed)]]
  HickettsOptional &operator=(const U &u) { return *this; }

  [[clang::set_typestate(consumed)]]
  HickettsOptional &operator=(nothing_t) {
    storage_ = nullptr;
    return *this;
  }
};

/// FREE FACTORY FUNCTIONS -- return_typestate on a non-member.
///
/// gaps.md:46-47 lists the make_optional factory hook as one of two residual
/// hardcoded names with no attribute-based answer. return_typestate is that
/// answer, and it costs nothing extra: the attribute already applies to plain
/// functions (Subjects = [Function, ParmVar]), not just methods.
template <typename T>
[[clang::return_typestate(unconsumed)]]
HickettsOptional<T> makeEngaged(T v) {
  return HickettsOptional<T>(v);
}

template <typename T>
[[clang::return_typestate(consumed)]]
HickettsOptional<T> makeEmpty() {
  return HickettsOptional<T>(nothing);
}

} // namespace mylib

#endif // HICKETTS_OPTIONAL_TYPESTATE_H_
