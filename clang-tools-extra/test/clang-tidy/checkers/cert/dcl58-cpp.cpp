// RUN: %check_clang_tidy -std=c++17-or-later %s cert-dcl58-cpp %t -- -- -I %clang_tidy_headers

#include "system-header-simulation.h"

namespace A {
  namespace B {
    int b;
  }
}

namespace A {
  namespace B {
    int c;
  }
}

namespace posix {
// CHECK-MESSAGES: :[[@LINE+2]]:11: warning: modification of 'posix' namespace can result in undefined behavior [cert-dcl58-cpp]
// CHECK-MESSAGES: :[[@LINE-2]]:11: note: 'posix' namespace opened here
namespace foo {
int foobar;
}
}

namespace std {
// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-2]]:11: note: 'std' namespace opened here
int stdInt;
// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-5]]:11: note: 'std' namespace opened here
int stdInt1;
}

namespace foobar {
  namespace std {
    int bar;
  }
}

namespace posix {
// CHECK-MESSAGES: :[[@LINE+2]]:11: warning: modification of 'posix' namespace
// CHECK-MESSAGES: :[[@LINE-2]]:11: note: 'posix' namespace opened here
namespace std {
}
} // namespace posix

namespace posix::a {
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: modification of 'posix' namespace
// CHECK-MESSAGES: :[[@LINE-2]]:11: note: 'posix' namespace opened here
}

namespace std {
// no-warning: empty
} // namespace std

namespace std {
// Warn for non-NamedDecls as well.
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
static_assert(1 == 1, "non-NamedDecl");
} // namespace std

enum class MyError {
  ErrorA,
  ErrorB
};

namespace std {
// no-warning: Class template specialized by a program-defined type.
template <>
struct is_error_code_enum<MyError> : std::true_type {};

// no-warning: Function template specialized by a program-defined type.
template<>
void swap<MyError>(MyError &a, MyError &b);
}

using ConstBoolPtr = const bool *;

namespace std {
// class template, builtin type
// CHECK-MESSAGES: :[[@LINE+3]]:8: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
template <>
struct is_error_code_enum<bool> : std::true_type {};
// function template, builtin type
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-8]]:11: note: 'std' namespace opened here
template <>
void swap<bool>(bool &, bool &);
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-12]]:11: note: 'std' namespace opened here
template <>
void swap<ConstBoolPtr>(ConstBoolPtr &, ConstBoolPtr &);
} // namespace std

namespace std {
// class template, std type
// CHECK-MESSAGES: :[[@LINE+3]]:8: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
template <>
struct is_error_code_enum<std::io_errc> : std::true_type {};
// function template, std type
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-8]]:11: note: 'std' namespace opened here
template <>
void swap<std::io_errc>(std::io_errc &, std::io_errc &);
} // namespace std

// parameter pack, has program-defined type
namespace std {
// no-warning: there is one program-defined type.
template <>
class tuple<int, MyError, std::io_errc> {};
} // namespace std

// parameter pack, only builtin or std type
namespace std {
// Forbid variadic specializations over only `std::` or builtin types.
// CHECK-MESSAGES: :[[@LINE+3]]:7: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
template <>
class tuple<int, const std::io_errc, float> {};
} // namespace std

namespace std {
// Test nested standard declarations.
// CHECK-MESSAGES: :[[@LINE+3]]:8: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
template <>
struct is_error_code_enum<std::Outer::Inner> : std::true_type {};
} // namespace std

namespace std {
// Test nested namespace.
// CHECK-MESSAGES: :[[@LINE+3]]:8: warning: modification of 'std' namespace
// CHECK-MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
template <>
struct is_error_code_enum<std::detail::X> : std::true_type {};
} // namespace std

// Test member function template specializations.
namespace std {
// CHECK-MESSAGES: :[[@LINE+3]]:18: warning: modification of 'std' namespace
// CHECK_MESSAGES: :[[@LINE-2]]:11: note: 'std' namespace opened here
template <>
bool less<void>::operator()<int &&, float &&>(int &&, float &&) const {
  return true;
}
// CHECK-MESSAGES: :[[@LINE+3]]:18: warning: modification of 'std' namespace
// CHECK_MESSAGES: :[[@LINE-8]]:11: note: 'std' namespace opened here
template <>
bool less<void>::operator()<MyError &&, MyError &&>(MyError &&, MyError &&) const {
  return true;
}
} // namespace std

// Test member class template specializations.
namespace std {
// CHECK-MESSAGES: :[[@LINE+3]]:20: warning: modification of 'std' namespace
// CHECK_MESSAGES: :[[@LINE-2]]:11: note: 'std' namespace opened here
template <>
struct less<void>::X<bool> {};
// CHECK-MESSAGES: :[[@LINE+3]]:20: warning: modification of 'std' namespace
// CHECK_MESSAGES: :[[@LINE-6]]:11: note: 'std' namespace opened here
template <>
struct less<void>::X<MyError> {};
// CHECK-MESSAGES: :[[@LINE+3]]:20: warning: modification of 'std' namespace
// CHECK_MESSAGES: :[[@LINE-10]]:11: note: 'std' namespace opened here
template <typename T>
struct less<void>::X<MyError, T> {};
} // namespace std

// We did not open the 'std' namespace, but still specialized the member
// function of 'std::less'.
// CHECK-MESSAGES: :[[@LINE+3]]:23: warning: modification of 'std' namespace
// no-note: There is no opening of 'std' namespace, hence no note emitted.
template <>
bool std::less<void>::operator()<int &&, int &&>(int &&, int &&) const {
  return true;
}

namespace SpaceA {
namespace SpaceB {
class MapKey {
  int Type = 0;

public:
  MapKey() = default;
  int getType() const { return Type; }
};
} // namespace SpaceB
} // namespace SpaceA

// no-warning: Specializing for 'std::hash' for a program-defined type.
template <>
struct std::hash<::SpaceA::SpaceB::MapKey> {
  // no-warning
  unsigned long operator()(const ::SpaceA::SpaceB::MapKey &K) const {
    return K.getType();
  }
  // no-warning
  bool operator()(const ::SpaceA::SpaceB::MapKey &K1,
                  const ::SpaceA::SpaceB::MapKey &K2) const {
    return K1.getType() < K2.getType();
  }
};

using myint = int;

// The type alias declaration is the same as typedef, does not introduce a
// program-defined type.
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <>
struct std::hash<myint> {
  // no-warning: The warning was already reported for the struct itself.
  unsigned long operator()(const myint &K) const {
    return K;
  }
  // no-warning: The warning was already reported for the struct itself.
  bool operator()(const myint &K1,
                  const myint &K2) const {
    return K1 < K2;
  }
};

// CHECK-MESSAGES: :[[@LINE+2]]:15: warning: modification of 'std' namespace
template <>
struct ::std::hash<long> {
  unsigned long operator()(const long &K) const {
    return K;
  }
};

namespace ranges {
namespace detail {
struct diffmax_t {};
using LongT = long;
} // namespace detail
} // namespace ranges

namespace std {
// no-warning: specialization with an user-defined type
template <>
struct numeric_limits<::ranges::detail::diffmax_t> {
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr ::ranges::detail::diffmax_t max() noexcept {
    return {};
  }
};
inline constexpr bool numeric_limits<::ranges::detail::diffmax_t>::is_signed;
inline constexpr bool numeric_limits<::ranges::detail::diffmax_t>::is_integer;
} // namespace std

namespace std {
// specialization with type alias to non-program-defined-type
// CHECK-MESSAGES: :[[@LINE+3]]:8: warning: modification of 'std' namespace
// CHECK_MESSAGES: :[[@LINE-3]]:11: note: 'std' namespace opened here
template <>
struct numeric_limits<::ranges::detail::LongT> {
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr ::ranges::detail::LongT max() noexcept {
    return 1;
  }
};
inline constexpr bool numeric_limits<::ranges::detail::LongT>::is_signed;
inline constexpr bool numeric_limits<::ranges::detail::LongT>::is_integer;
} // namespace std
