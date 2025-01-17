// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-optional-value-conversion %t

namespace std {
template <typename T> struct optional {
  constexpr optional() noexcept;
  constexpr optional(T &&) noexcept;
  constexpr optional(const T &) noexcept;
  template <typename U> constexpr optional(U &&) noexcept;
  const T &operator*() const;
  T *operator->();
  const T *operator->() const;
  T &operator*();
  const T &value() const;
  T &value();
  const T &get() const;
  T &get();
  T value_or(T) const;
};

template <class T> T &&move(T &x) { return static_cast<T &&>(x); }

template <typename T> class default_delete {};

template <typename type, typename Deleter = std::default_delete<type>>
class unique_ptr {};

template <typename type>
class shared_ptr {};

template <class T, class... Args> unique_ptr<T> make_unique(Args &&...args);
template <class T, class... Args> shared_ptr<T> make_shared(Args &&...args);

} // namespace std

struct A {
    explicit A (int);
};
std::optional<int> opt;

void invalid() {
  std::make_unique<std::optional<int>>(opt.value());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  using A = std::optional<int>;
  std::make_unique<A>(opt.value());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
  std::make_shared<std::optional<int>>(opt.value());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: conversion from 'std::optional<int>' into 'int' and back into 'std::optional<int>', remove potentially error-prone optional dereference [bugprone-optional-value-conversion]
}

void valid() {
  std::make_unique<A>(opt.value());
  std::make_shared<A>(opt.value());
}
