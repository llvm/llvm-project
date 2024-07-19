// RUN: %check_clang_tidy %s cppcoreguidelines-missing-std-forward %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     {cppcoreguidelines-missing-std-forward.IgnoreStaticCasts: true }}" \
// RUN: -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {

template <typename T> struct remove_reference      { using type = T; };
template <typename T> struct remove_reference<T&>  { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };

template <typename T> using remove_reference_t = typename remove_reference<T>::type;

template <typename T> constexpr T &&forward(remove_reference_t<T> &t) noexcept;
template <typename T> constexpr T &&forward(remove_reference_t<T> &&t) noexcept;

} // namespace std
// NOLINTEND

namespace in_static_cast {

template<typename T>
void static_cast_to_lvalue_ref(T&& t) {
  static_cast<T&>(t);
}

} // namespace in_static_cast

