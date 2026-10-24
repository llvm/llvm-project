#ifndef LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_TL_EXPECTED_HPP_
#define LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_TL_EXPECTED_HPP_

/// Mock of `tl::expected` from https://github.com/TartanLlama/expected.
///
/// Unlike `std::expected`, it has no `void` specialization and its comparison
/// operators are free functions that are declared for both operand orders,
/// including `operator!=`.

#include <type_traits>

namespace tl {

template <class T, class E> class expected;

struct in_place_t {
  explicit in_place_t() = default;
};
static constexpr in_place_t in_place{};

template <class E> class unexpected {
public:
  unexpected() = delete;
  explicit unexpected(const E &e);
  explicit unexpected(E &&e);

  const E &value() const &;
  E &value() &;
  E &&value() &&;
  const E &&value() const &&;
};

template <class E> unexpected(E) -> unexpected<E>;

template <class E>
unexpected<typename std::decay<E>::type> make_unexpected(E &&e);

struct unexpect_t {
  unexpect_t() = default;
};
static constexpr unexpect_t unexpect{};

namespace detail {

template <class> constexpr bool is_unexpected = false;
template <class E> constexpr bool is_unexpected<unexpected<E>> = true;

// The result of calling `F` with the value of an `expected<T, E>`.
template <class F, class T> struct value_result {
  using type = typename std::remove_cv<decltype(std::declval<F>()(
      std::declval<T &>()))>::type;
};
template <class F> struct value_result<F, void> {
  using type = typename std::remove_cv<decltype(std::declval<F>()())>::type;
};
template <class F> struct value_result<F, const void> {
  using type = typename std::remove_cv<decltype(std::declval<F>()())>::type;
};

// The result of calling `F` with the error of an `expected<T, E>`.
template <class F, class E> struct error_result {
  using type = typename std::remove_cv<decltype(std::declval<F>()(
      std::declval<E &>()))>::type;
};

} // namespace detail

template <class T, class E> class expected {
public:
  typedef T value_type;
  typedef E error_type;
  typedef unexpected<E> unexpected_type;

  constexpr expected() = default;
  constexpr expected(const expected &rhs) = default;
  constexpr expected(expected &&rhs) = default;
  expected &operator=(const expected &rhs) = default;
  expected &operator=(expected &&rhs) = default;

  template <class... Args> expected(in_place_t, Args &&...args);

  template <class G> expected(const unexpected<G> &e);
  template <class G> expected(unexpected<G> &&e);

  template <class... Args> explicit expected(unexpect_t, Args &&...args);

  template <class U, class G,
            std::enable_if_t<__is_constructible(T, const U &) &&
                             __is_constructible(E, const G &)> * = nullptr>
  expected(const expected<U, G> &rhs);
  template <class U, class G,
            std::enable_if_t<__is_constructible(T, U &&) &&
                             __is_constructible(E, G &&)> * = nullptr>
  expected(expected<U, G> &&rhs);

  template <
      class U = T,
      std::enable_if_t<
          __is_constructible(T, U &&) &&
          !std::is_same<typename std::decay<U>::type, in_place_t>::value &&
          !std::is_same<typename std::decay<U>::type, unexpect_t>::value &&
          !std::is_same<typename std::decay<U>::type, expected>::value &&
          !detail::is_unexpected<typename std::decay<U>::type>> * = nullptr>
  expected(U &&v);

  // `G` delays forming `T &`, which is ill-formed for `void`.
  template <
      class U = T, class G = T,
      std::enable_if_t<
          __is_constructible(T, U &&) && __is_assignable(G &, U &&) &&
          !std::is_same<typename std::decay<U>::type, expected>::value &&
          !detail::is_unexpected<typename std::decay<U>::type>> * = nullptr>
  expected &operator=(U &&v);

  template <class G> expected &operator=(const unexpected<G> &rhs);
  template <class G> expected &operator=(unexpected<G> &&rhs);

  template <class... Args> void emplace(Args &&...args);

  void swap(expected &rhs);

  const T *operator->() const;
  T *operator->();

  template <class U = T,
            std::enable_if_t<!std::is_void<U>::value> * = nullptr>
  const U &operator*() const &;
  template <class U = T,
            std::enable_if_t<!std::is_void<U>::value> * = nullptr>
  U &operator*() &;
  template <class U = T,
            std::enable_if_t<!std::is_void<U>::value> * = nullptr>
  U &&operator*() &&;

  bool has_value() const noexcept;
  explicit operator bool() const noexcept;

  template <class U = T,
            std::enable_if_t<!std::is_void<U>::value> * = nullptr>
  const U &value() const &;
  template <class U = T,
            std::enable_if_t<!std::is_void<U>::value> * = nullptr>
  U &value() &;
  template <class U = T,
            std::enable_if_t<!std::is_void<U>::value> * = nullptr>
  U &&value() &&;

  const E &error() const &;
  E &error() &;
  E &&error() &&;

  template <class U> T value_or(U &&v) const &;
  template <class U> T value_or(U &&v) &&;

  // Defined, so that they can be called with lambdas, whose types have no
  // linkage.
  template <class F>
  expected<typename detail::value_result<F, T>::type, E> map(F &&) & {
    return {};
  }
  template <class F>
  expected<typename detail::value_result<F, const T>::type, E>
  map(F &&) const & {
    return {};
  }
  template <class F>
  expected<typename detail::value_result<F, T>::type, E> map(F &&) && {
    return {};
  }

  template <class F>
  expected<typename detail::value_result<F, T>::type, E> transform(F &&) & {
    return {};
  }
  template <class F>
  expected<typename detail::value_result<F, const T>::type, E>
  transform(F &&) const & {
    return {};
  }
  template <class F>
  expected<typename detail::value_result<F, T>::type, E> transform(F &&) && {
    return {};
  }

  template <class F>
  expected<T, typename detail::error_result<F, E>::type> map_error(F &&) & {
    return {};
  }
  template <class F>
  expected<T, typename detail::error_result<F, const E>::type>
  map_error(F &&) const & {
    return {};
  }
  template <class F>
  expected<T, typename detail::error_result<F, E>::type> map_error(F &&) && {
    return {};
  }

  template <class F>
  expected<T, typename detail::error_result<F, E>::type>
  transform_error(F &&) & {
    return {};
  }
  template <class F>
  expected<T, typename detail::error_result<F, const E>::type>
  transform_error(F &&) const & {
    return {};
  }
  template <class F>
  expected<T, typename detail::error_result<F, E>::type>
  transform_error(F &&) && {
    return {};
  }
};

template <class T, class E, class U, class F>
bool operator==(const expected<T, E> &lhs, const expected<U, F> &rhs);
template <class T, class E, class U, class F>
bool operator!=(const expected<T, E> &lhs, const expected<U, F> &rhs);
template <class E, class F>
bool operator==(const expected<void, E> &lhs, const expected<void, F> &rhs);
template <class E, class F>
bool operator!=(const expected<void, E> &lhs, const expected<void, F> &rhs);

template <class T, class E, class U>
bool operator==(const expected<T, E> &x, const U &v);
template <class T, class E, class U>
bool operator==(const U &v, const expected<T, E> &x);
template <class T, class E, class U>
bool operator!=(const expected<T, E> &x, const U &v);
template <class T, class E, class U>
bool operator!=(const U &v, const expected<T, E> &x);

template <class T, class E>
bool operator==(const expected<T, E> &x, const unexpected<E> &e);
template <class T, class E>
bool operator==(const unexpected<E> &e, const expected<T, E> &x);
template <class T, class E>
bool operator!=(const expected<T, E> &x, const unexpected<E> &e);
template <class T, class E>
bool operator!=(const unexpected<E> &e, const expected<T, E> &x);

template <class T, class E>
void swap(expected<T, E> &lhs, expected<T, E> &rhs);

} // namespace tl

#endif // LLVM_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_INPUTS_TL_EXPECTED_HPP_
