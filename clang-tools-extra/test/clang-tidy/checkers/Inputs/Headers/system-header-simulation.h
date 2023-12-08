#pragma clang system_header

namespace std {

template<class T, T v>
struct integral_constant {
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    constexpr operator value_type() const noexcept { return value; }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template<class T>
struct is_error_code_enum : false_type {};

template <class T>
void swap(T &a, T &b);

enum class io_errc {
  stream = 1,
};

template <class... Types>
class tuple;

template <typename T = void>
class less;

template <>
class less<void> {
public:
  template <typename T, typename U>
  bool operator()(T &&Lhs, U &&Rhs) const {
    return static_cast<T &&>(Lhs) < static_cast<U &&>(Rhs);
  }
  template <typename A, typename B = int>
  struct X {};
};

template <class Key>
struct hash;

template <class T>
class numeric_limits;

struct Outer {
  struct Inner {};
};

namespace detail {
struct X {};
} // namespace detail

} // namespace std

// Template specializations that are in a system-header file.
// The purpose is to test cert-dcl58-cpp (no warnings here).
namespace std {
template <>
void swap<short>(short &, short &){};

template <>
struct is_error_code_enum<short> : true_type {};

template <>
bool less<void>::operator()<short &&, short &&>(short &&, short &&) const {
  return false;
}

template <>
struct less<void>::X<short> {};
} // namespace std
