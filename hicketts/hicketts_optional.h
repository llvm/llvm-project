#ifndef HICKETTS_OPTIONAL_H_
#define HICKETTS_OPTIONAL_H_

/// A custom optional-like type with differently named functions.
/// Mirrors std::optional semantics but uses its own vocabulary
/// In order to test implementation of attributes for clang-tidy
namespace mylib {

struct nothing_t {
  constexpr explicit nothing_t() {}
};

constexpr nothing_t nothing;

template <typename T>
class [[clang::analyze_as_class("std::optional")]] HickettsOptional {
  T *storage_ = nullptr;

public:
  // No matcher needed: default (0-arg) construction matches none of the
  // constructor cases, so has_value is left unconstrained and access is
  // conservatively treated as maybe-empty (warns).
  // [[clang::analyze_as_method("optional()")]]
  constexpr HickettsOptional() noexcept {}

  // KEEP (POC target): nothing_t is not std::nullopt_t, so
  // isOptionalNulloptConstructor (UncheckedOptionalAccessModel.cpp:288) misses
  // and this falls through to the value/conversion case (:300) -> wrongly
  // engaged. The new signature-matched constructor case will route this
  // annotation to the nullopt transfer (empty).
  [[clang::analyze_as_method("optional(std::nullopt_t)")]]
  constexpr HickettsOptional(nothing_t) noexcept {}

  // Already handled by isOptionalValueOrConversionConstructor (:300, registered
  // :1038): single-arg construction from a value -> engaged.
  // [[clang::analyze_as_method("optional(T&&)")]]
  constexpr HickettsOptional(T) noexcept {}

  // Copy ctor: no dedicated case; excluded from value/conversion (:302-303) and
  // handled by the framework's default record-copy, which propagates has_value
  // from the source.
  // [[clang::analyze_as_method("optional(const optional&)")]]
  HickettsOptional(const HickettsOptional &) = default;

  // Move ctor: same as copy — excluded from value/conversion (:302-303),
  // handled by the framework's default record-copy.
  // [[clang::analyze_as_method("optional(const optional&&)")]]
  HickettsOptional(HickettsOptional &&) = default;

  // Equivalent to std::optional::value()
  [[clang::analyze_as_method("value")]] const T &unwrap() const & { return *storage_; }
  [[clang::analyze_as_method("value")]] T &unwrap() & { return *storage_; }
  [[clang::analyze_as_method("value")]] const T &&unwrap() const && { return static_cast<const T &&>(*storage_); }
  [[clang::analyze_as_method("value")]] T &&unwrap() && { return static_cast<T &&>(*storage_); }

  const T &value() const & { return *storage_; }
  T &value() & { return *storage_; }
  const T &&value() const && { return static_cast<const T &&>(*storage_); }
  T &&value() && { return static_cast<T &&>(*storage_); }

  // Equivalent to std::optional::operator*()
  [[clang::analyze_as_method("value")]] const T &deref() const & { return *storage_; }
  [[clang::analyze_as_method("value")]] T &deref() & { return *storage_; }

  // Equivalent to std::optional::operator->()
  const T* operator ->() const { return storage_; }
  T* operator ->() { return storage_; }
  const T *arrow() const { return storage_; }
  T *arrow() { return storage_; }

  // Equivalent to std::optional::operator bool / hasValue()
  constexpr bool has_value() const noexcept { return storage_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return storage_ != nullptr; }
  [[clang::analyze_as_method("has_value")]] constexpr bool isPresent() const noexcept { return storage_ != nullptr; }

  // Equivalent to std::optional::value_or()
  template <typename U>
  constexpr T unwrapOr(U &&fallback) const & {
    return storage_ ? *storage_ : static_cast<T>(fallback);
  }

  // Equivalent to std::optional::emplace()
  template <typename... Args>
  [[clang::analyze_as_method("emplace(Args&&...)")]]
  T& construct(Args&&... args) { return *storage_; }

  // Demo of malformed-signature rejection — disabled. The parameter-balance
  // validation in Sema (isValidAnalyzeAsMethodAttr) that rejected this string
  // was removed, since matching is now a flat string compare that never parses
  // parameters. With validation gone this annotation would be accepted silently
  // (and simply never match), so the case no longer demonstrates anything.
  // [[clang::analyze_as_method("emplace(oops))")]]
  // T& load() { return *storage_; }

  // Equivalent to std::optional::reset()
  [[clang::analyze_as_method("reset")]] void clear() noexcept { storage_ = nullptr; }

  // Equivalent to std::optional::swap()
  [[clang::analyze_as_method("swap")]] void exchange(HickettsOptional &other) noexcept {
    T *tmp = storage_;
    storage_ = other.storage_;
    other.storage_ = tmp;
  }

  // Assignment
  template <typename U>
  HickettsOptional &operator=(const U &u) { return *this; }

  [[clang::analyze_as_method("operator=(nullopt_t)")]]
  HickettsOptional &operator=(mylib::nothing_t){ storage_ = nullptr; return *this;}

};

} // namespace mylib

#endif // HICKETTS_OPTIONAL_H_
