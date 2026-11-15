#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_EXPECTED_H_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_EXPECTED_H_

/// Mock of `std::expected`.
// TODO: move to checkers/Inputs/Headers/std/expected?

// For `std::in_place_t` and `std::in_place`.
#include <type_traits>

namespace std {

template <class E> class unexpected {
public:
  unexpected(const unexpected &) = default;
  unexpected(unexpected &&) = default;

  template <class Err = E> explicit unexpected(Err &&e);

  const E &error() const & noexcept;
  E &error() & noexcept;
  const E &&error() const && noexcept;
  E &&error() && noexcept;
};

template <class E> unexpected(E) -> unexpected<E>;

struct unexpect_t {
  explicit unexpect_t() = default;
};

inline constexpr unexpect_t unexpect{};

template <class T, class E> class expected;

template <class> constexpr bool __is_expected = false;
template <class T, class E>
constexpr bool __is_expected<expected<T, E>> = true;

template <class> constexpr bool __is_unexpected = false;
template <class E> constexpr bool __is_unexpected<unexpected<E>> = true;

// `__remove_cvref` is a clang builtin, so the helper has a different name.
// clang-format off
template <class T> struct __remove_cvref_impl              { using type = T; };
template <class T> struct __remove_cvref_impl<const T>     { using type = T; };
template <class T> struct __remove_cvref_impl<T &>         { using type = T; };
template <class T> struct __remove_cvref_impl<const T &>   { using type = T; };
template <class T> struct __remove_cvref_impl<T &&>        { using type = T; };
template <class T> struct __remove_cvref_impl<const T &&>  { using type = T; };
// clang-format on

template <class T>
using __remove_cvref_t = typename __remove_cvref_impl<T>::type;

// `__remove_cv` is a clang builtin as well.
template <class T> struct __remove_cv_impl { using type = T; };
template <class T> struct __remove_cv_impl<const T> { using type = T; };

template <class T> using __remove_cv_t = typename __remove_cv_impl<T>::type;

// Whether `T` can be constructed from `expected<U, G>`, in which case the
// converting constructors from `expected<U, G>` are disabled.
template <class T, class U, class G>
constexpr bool __cons_from_expected =
    __is_constructible(T, expected<U, G> &) ||
    __is_constructible(T, expected<U, G>) ||
    __is_constructible(T, const expected<U, G> &) ||
    __is_constructible(T, const expected<U, G>);

template <class T, class E> class expected {
public:
  using value_type = T;
  using error_type = E;
  using unexpected_type = unexpected<E>;

  expected();
  expected(const expected &);
  expected(expected &&) noexcept;

  template <class U, class G>
    requires(__is_constructible(T, const U &) &&
             __is_constructible(E, const G &) &&
             !__cons_from_expected<T, U, G>)
  expected(const expected<U, G> &x);
  template <class U, class G>
    requires(__is_constructible(T, U) && __is_constructible(E, G) &&
             !__cons_from_expected<T, U, G>)
  expected(expected<U, G> &&x);

  template <class U = T>
    requires(!__is_same(__remove_cvref_t<U>, expected) &&
             !__is_same(__remove_cvref_t<U>, in_place_t) &&
             !__is_same(__remove_cvref_t<U>, unexpect_t) &&
             !__is_unexpected<__remove_cvref_t<U>> && __is_constructible(T, U))
  expected(U &&v);

  template <class G> expected(const unexpected<G> &e);
  template <class G> expected(unexpected<G> &&e);

  template <class... Args> explicit expected(in_place_t, Args &&...args);

  template <class... Args>
  explicit expected(unexpect_t, Args &&...args);

  expected &operator=(const expected &);
  expected &operator=(expected &&) noexcept;

  template <class U = T>
    requires(!__is_same(__remove_cvref_t<U>, expected) &&
             !__is_unexpected<__remove_cvref_t<U>> &&
             __is_constructible(T, U) && __is_assignable(T &, U))
  expected &operator=(U &&v);

  template <class G> expected &operator=(const unexpected<G> &e);
  template <class G> expected &operator=(unexpected<G> &&e);

  const T *operator->() const noexcept;
  T *operator->() noexcept;

  const T &operator*() const & noexcept;
  T &operator*() & noexcept;
  const T &&operator*() const && noexcept;
  T &&operator*() && noexcept;

  explicit operator bool() const noexcept;
  bool has_value() const noexcept;

  const T &value() const &;
  T &value() &;
  const T &&value() const &&;
  T &&value() &&;

  const E &error() const & noexcept;
  E &error() & noexcept;
  const E &&error() const && noexcept;
  E &&error() && noexcept;

  template <class U> T value_or(U &&v) const &;
  template <class U> T value_or(U &&v) &&;

  template <class G> E error_or(G &&e) const &;
  template <class G> E error_or(G &&e) &&;

  // Defined, so that they can be called with lambdas, whose types have no
  // linkage.
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()(declval<T &>()))>, E>
  transform(F &&) & { return {}; }
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()(declval<const T &>()))>, E>
  transform(F &&) const & { return {}; }
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()(declval<T>()))>, E>
  transform(F &&) && { return {}; }
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()(declval<const T>()))>, E>
  transform(F &&) const && { return {}; }

  template <class F>
  expected<T, __remove_cv_t<decltype(declval<F>()(declval<E &>()))>>
  transform_error(F &&) & { return {}; }
  template <class F>
  expected<T, __remove_cv_t<decltype(declval<F>()(declval<const E &>()))>>
  transform_error(F &&) const & { return {}; }
  template <class F>
  expected<T, __remove_cv_t<decltype(declval<F>()(declval<E>()))>>
  transform_error(F &&) && { return {}; }
  template <class F>
  expected<T, __remove_cv_t<decltype(declval<F>()(declval<const E>()))>>
  transform_error(F &&) const && { return {}; }

  template <class... Args> T &emplace(Args &&...args) noexcept;

  void swap(expected &rhs) noexcept;

  template <class T2, class E2>
  friend bool operator==(const expected &x, const expected<T2, E2> &y);

  template <class T2>
    requires(!__is_expected<T2>)
  friend bool operator==(const expected &x, const T2 &v);

  template <class E2>
  friend bool operator==(const expected &x, const unexpected<E2> &e);
};

template <class E> class expected<void, E> {
public:
  using value_type = void;
  using error_type = E;
  using unexpected_type = unexpected<E>;

  expected() noexcept;
  expected(const expected &);
  expected(expected &&) noexcept;

  template <class U, class G> expected(const expected<U, G> &x);
  template <class U, class G> expected(expected<U, G> &&x);

  template <class G> expected(const unexpected<G> &e);
  template <class G> expected(unexpected<G> &&e);

  explicit expected(in_place_t) noexcept;

  template <class... Args>
  explicit expected(unexpect_t, Args &&...args);

  expected &operator=(const expected &);
  expected &operator=(expected &&) noexcept;

  template <class G> expected &operator=(const unexpected<G> &e);
  template <class G> expected &operator=(unexpected<G> &&e);

  explicit operator bool() const noexcept;
  bool has_value() const noexcept;

  void operator*() const noexcept;

  void value() const &;
  void value() &&;

  const E &error() const & noexcept;
  E &error() & noexcept;
  const E &&error() const && noexcept;
  E &&error() && noexcept;

  template <class G> E error_or(G &&e) const &;
  template <class G> E error_or(G &&e) &&;

  // Defined, so that they can be called with lambdas, whose types have no
  // linkage.
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()())>, E>
  transform(F &&) & { return {}; }
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()())>, E>
  transform(F &&) const & { return {}; }
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()())>, E>
  transform(F &&) && { return {}; }
  template <class F>
  expected<__remove_cv_t<decltype(declval<F>()())>, E>
  transform(F &&) const && { return {}; }

  template <class F>
  expected<void, __remove_cv_t<decltype(declval<F>()(declval<E &>()))>>
  transform_error(F &&) & { return {}; }
  template <class F>
  expected<void,
           __remove_cv_t<decltype(declval<F>()(declval<const E &>()))>>
  transform_error(F &&) const & { return {}; }
  template <class F>
  expected<void, __remove_cv_t<decltype(declval<F>()(declval<E>()))>>
  transform_error(F &&) && { return {}; }
  template <class F>
  expected<void, __remove_cv_t<decltype(declval<F>()(declval<const E>()))>>
  transform_error(F &&) const && { return {}; }

  void emplace() noexcept;

  void swap(expected &rhs) noexcept;

  template <class T2, class E2>
  friend bool operator==(const expected &x, const expected<T2, E2> &y);

  template <class E2>
  friend bool operator==(const expected &x, const unexpected<E2> &e);
};

} // namespace std

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_STD_TYPES_EXPECTED_H_
