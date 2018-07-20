#ifndef STD_COMPARE_H
#define STD_COMPARE_H

namespace std {
inline namespace __1 {

// exposition only
enum class _EqResult : unsigned char {
  __zero = 0,
  __equal = __zero,
  __equiv = __equal,
  __nonequal = 1,
  __nonequiv = __nonequal
};

enum class _OrdResult : signed char {
  __less = -1,
  __greater = 1
};

enum class _NCmpResult : signed char {
  __unordered = -127
};

struct _CmpUnspecifiedType;
using _CmpUnspecifiedParam = void (_CmpUnspecifiedType::*)();

class weak_equality {
  constexpr explicit weak_equality(_EqResult __val) noexcept : __value_(__val) {}

public:
  static const weak_equality equivalent;
  static const weak_equality nonequivalent;

  friend constexpr bool operator==(weak_equality __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator==(_CmpUnspecifiedParam, weak_equality __v) noexcept;
  friend constexpr bool operator!=(weak_equality __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator!=(_CmpUnspecifiedParam, weak_equality __v) noexcept;
  friend constexpr weak_equality operator<=>(weak_equality __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr weak_equality operator<=>(_CmpUnspecifiedParam, weak_equality __v) noexcept;

  // test helper
  constexpr bool test_eq(weak_equality const &other) const noexcept {
    return __value_ == other.__value_;
  }

private:
  _EqResult __value_;
};

inline constexpr weak_equality weak_equality::equivalent(_EqResult::__equiv);
inline constexpr weak_equality weak_equality::nonequivalent(_EqResult::__nonequiv);
inline constexpr bool operator==(weak_equality __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ == _EqResult::__zero;
}
inline constexpr bool operator==(_CmpUnspecifiedParam, weak_equality __v) noexcept {
  return __v.__value_ == _EqResult::__zero;
}
inline constexpr bool operator!=(weak_equality __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ != _EqResult::__zero;
}
inline constexpr bool operator!=(_CmpUnspecifiedParam, weak_equality __v) noexcept {
  return __v.__value_ != _EqResult::__zero;
}

inline constexpr weak_equality operator<=>(weak_equality __v, _CmpUnspecifiedParam) noexcept {
  return __v;
}
inline constexpr weak_equality operator<=>(_CmpUnspecifiedParam, weak_equality __v) noexcept {
  return __v;
}

class strong_equality {
  explicit constexpr strong_equality(_EqResult __val) noexcept : __value_(__val) {}

public:
  static const strong_equality equal;
  static const strong_equality nonequal;
  static const strong_equality equivalent;
  static const strong_equality nonequivalent;

  // conversion
  constexpr operator weak_equality() const noexcept {
    return __value_ == _EqResult::__zero ? weak_equality::equivalent
                                         : weak_equality::nonequivalent;
  }

  // comparisons
  friend constexpr bool operator==(strong_equality __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator!=(strong_equality __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator==(_CmpUnspecifiedParam, strong_equality __v) noexcept;
  friend constexpr bool operator!=(_CmpUnspecifiedParam, strong_equality __v) noexcept;

  friend constexpr strong_equality operator<=>(strong_equality __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr strong_equality operator<=>(_CmpUnspecifiedParam, strong_equality __v) noexcept;

  // test helper
  constexpr bool test_eq(strong_equality const &other) const noexcept {
    return __value_ == other.__value_;
  }

private:
  _EqResult __value_;
};

inline constexpr strong_equality strong_equality::equal(_EqResult::__equal);
inline constexpr strong_equality strong_equality::nonequal(_EqResult::__nonequal);
inline constexpr strong_equality strong_equality::equivalent(_EqResult::__equiv);
inline constexpr strong_equality strong_equality::nonequivalent(_EqResult::__nonequiv);
constexpr bool operator==(strong_equality __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ == _EqResult::__zero;
}
constexpr bool operator==(_CmpUnspecifiedParam, strong_equality __v) noexcept {
  return __v.__value_ == _EqResult::__zero;
}
constexpr bool operator!=(strong_equality __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ != _EqResult::__zero;
}
constexpr bool operator!=(_CmpUnspecifiedParam, strong_equality __v) noexcept {
  return __v.__value_ != _EqResult::__zero;
}

constexpr strong_equality operator<=>(strong_equality __v, _CmpUnspecifiedParam) noexcept {
  return __v;
}
constexpr strong_equality operator<=>(_CmpUnspecifiedParam, strong_equality __v) noexcept {
  return __v;
}

class partial_ordering {
  using _ValueT = signed char;
  explicit constexpr partial_ordering(_EqResult __v) noexcept
      : __value_(_ValueT(__v)) {}
  explicit constexpr partial_ordering(_OrdResult __v) noexcept
      : __value_(_ValueT(__v)) {}
  explicit constexpr partial_ordering(_NCmpResult __v) noexcept
      : __value_(_ValueT(__v)) {}

  constexpr bool __is_ordered() const noexcept {
    return __value_ != _ValueT(_NCmpResult::__unordered);
  }

public:
  // valid values
  static const partial_ordering less;
  static const partial_ordering equivalent;
  static const partial_ordering greater;
  static const partial_ordering unordered;

  // conversion
  constexpr operator weak_equality() const noexcept {
    return __value_ == 0 ? weak_equality::equivalent : weak_equality::nonequivalent;
  }

  // comparisons
  friend constexpr bool operator==(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator!=(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator<(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator<=(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator>(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator>=(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator==(_CmpUnspecifiedParam, partial_ordering __v) noexcept;
  friend constexpr bool operator!=(_CmpUnspecifiedParam, partial_ordering __v) noexcept;
  friend constexpr bool operator<(_CmpUnspecifiedParam, partial_ordering __v) noexcept;
  friend constexpr bool operator<=(_CmpUnspecifiedParam, partial_ordering __v) noexcept;
  friend constexpr bool operator>(_CmpUnspecifiedParam, partial_ordering __v) noexcept;
  friend constexpr bool operator>=(_CmpUnspecifiedParam, partial_ordering __v) noexcept;

  friend constexpr partial_ordering operator<=>(partial_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr partial_ordering operator<=>(_CmpUnspecifiedParam, partial_ordering __v) noexcept;

  // test helper
  constexpr bool test_eq(partial_ordering const &other) const noexcept {
    return __value_ == other.__value_;
  }

private:
  _ValueT __value_;
};

inline constexpr partial_ordering partial_ordering::less(_OrdResult::__less);
inline constexpr partial_ordering partial_ordering::equivalent(_EqResult::__equiv);
inline constexpr partial_ordering partial_ordering::greater(_OrdResult::__greater);
inline constexpr partial_ordering partial_ordering::unordered(_NCmpResult ::__unordered);
constexpr bool operator==(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__is_ordered() && __v.__value_ == 0;
}
constexpr bool operator<(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__is_ordered() && __v.__value_ < 0;
}
constexpr bool operator<=(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__is_ordered() && __v.__value_ <= 0;
}
constexpr bool operator>(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__is_ordered() && __v.__value_ > 0;
}
constexpr bool operator>=(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__is_ordered() && __v.__value_ >= 0;
}
constexpr bool operator==(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return __v.__is_ordered() && 0 == __v.__value_;
}
constexpr bool operator<(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return __v.__is_ordered() && 0 < __v.__value_;
}
constexpr bool operator<=(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return __v.__is_ordered() && 0 <= __v.__value_;
}
constexpr bool operator>(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return __v.__is_ordered() && 0 > __v.__value_;
}
constexpr bool operator>=(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return __v.__is_ordered() && 0 >= __v.__value_;
}
constexpr bool operator!=(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return !__v.__is_ordered() || __v.__value_ != 0;
}
constexpr bool operator!=(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return !__v.__is_ordered() || __v.__value_ != 0;
}

constexpr partial_ordering operator<=>(partial_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v;
}
constexpr partial_ordering operator<=>(_CmpUnspecifiedParam, partial_ordering __v) noexcept {
  return __v < 0 ? partial_ordering::greater : (__v > 0 ? partial_ordering::less : __v);
}

class weak_ordering {
  using _ValueT = signed char;
  explicit constexpr weak_ordering(_EqResult __v) noexcept : __value_(_ValueT(__v)) {}
  explicit constexpr weak_ordering(_OrdResult __v) noexcept : __value_(_ValueT(__v)) {}

public:
  static const weak_ordering less;
  static const weak_ordering equivalent;
  static const weak_ordering greater;

  // conversions
  constexpr operator weak_equality() const noexcept {
    return __value_ == 0 ? weak_equality::equivalent
                         : weak_equality::nonequivalent;
  }
  constexpr operator partial_ordering() const noexcept {
    return __value_ == 0 ? partial_ordering::equivalent
                         : (__value_ < 0 ? partial_ordering::less : partial_ordering::greater);
  }

  // comparisons
  friend constexpr bool operator==(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator!=(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator<(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator<=(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator>(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator>=(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator==(_CmpUnspecifiedParam, weak_ordering __v) noexcept;
  friend constexpr bool operator!=(_CmpUnspecifiedParam, weak_ordering __v) noexcept;
  friend constexpr bool operator<(_CmpUnspecifiedParam, weak_ordering __v) noexcept;
  friend constexpr bool operator<=(_CmpUnspecifiedParam, weak_ordering __v) noexcept;
  friend constexpr bool operator>(_CmpUnspecifiedParam, weak_ordering __v) noexcept;
  friend constexpr bool operator>=(_CmpUnspecifiedParam, weak_ordering __v) noexcept;

  friend constexpr weak_ordering operator<=>(weak_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr weak_ordering operator<=>(_CmpUnspecifiedParam, weak_ordering __v) noexcept;

  // test helper
  constexpr bool test_eq(weak_ordering const &other) const noexcept {
    return __value_ == other.__value_;
  }

private:
  _ValueT __value_;
};

inline constexpr weak_ordering weak_ordering::less(_OrdResult::__less);
inline constexpr weak_ordering weak_ordering::equivalent(_EqResult::__equiv);
inline constexpr weak_ordering weak_ordering::greater(_OrdResult::__greater);
constexpr bool operator==(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ == 0;
}
constexpr bool operator!=(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ != 0;
}
constexpr bool operator<(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ < 0;
}
constexpr bool operator<=(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ <= 0;
}
constexpr bool operator>(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ > 0;
}
constexpr bool operator>=(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ >= 0;
}
constexpr bool operator==(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return 0 == __v.__value_;
}
constexpr bool operator!=(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return 0 != __v.__value_;
}
constexpr bool operator<(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return 0 < __v.__value_;
}
constexpr bool operator<=(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return 0 <= __v.__value_;
}
constexpr bool operator>(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return 0 > __v.__value_;
}
constexpr bool operator>=(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return 0 >= __v.__value_;
}

constexpr weak_ordering operator<=>(weak_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v;
}
constexpr weak_ordering operator<=>(_CmpUnspecifiedParam, weak_ordering __v) noexcept {
  return __v < 0 ? weak_ordering::greater : (__v > 0 ? weak_ordering::less : __v);
}

class strong_ordering {
  using _ValueT = signed char;
  explicit constexpr strong_ordering(_EqResult __v) noexcept : __value_(static_cast<signed char>(__v)) {}
  explicit constexpr strong_ordering(_OrdResult __v) noexcept : __value_(static_cast<signed char>(__v)) {}

public:
  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering equivalent;
  static const strong_ordering greater;

  // conversions
  constexpr operator weak_equality() const noexcept {
    return __value_ == 0 ? weak_equality::equivalent
                         : weak_equality::nonequivalent;
  }
  constexpr operator strong_equality() const noexcept {
    return __value_ == 0 ? strong_equality::equal
                         : strong_equality::nonequal;
  }
  constexpr operator partial_ordering() const noexcept {
    return __value_ == 0 ? partial_ordering::equivalent
                         : (__value_ < 0 ? partial_ordering::less : partial_ordering::greater);
  }
  constexpr operator weak_ordering() const noexcept {
    return __value_ == 0 ? weak_ordering::equivalent
                         : (__value_ < 0 ? weak_ordering::less : weak_ordering::greater);
  }

  // comparisons
  friend constexpr bool operator==(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator!=(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator<(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator<=(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator>(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator>=(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr bool operator==(_CmpUnspecifiedParam, strong_ordering __v) noexcept;
  friend constexpr bool operator!=(_CmpUnspecifiedParam, strong_ordering __v) noexcept;
  friend constexpr bool operator<(_CmpUnspecifiedParam, strong_ordering __v) noexcept;
  friend constexpr bool operator<=(_CmpUnspecifiedParam, strong_ordering __v) noexcept;
  friend constexpr bool operator>(_CmpUnspecifiedParam, strong_ordering __v) noexcept;
  friend constexpr bool operator>=(_CmpUnspecifiedParam, strong_ordering __v) noexcept;

  friend constexpr strong_ordering operator<=>(strong_ordering __v, _CmpUnspecifiedParam) noexcept;
  friend constexpr strong_ordering operator<=>(_CmpUnspecifiedParam, strong_ordering __v) noexcept;

  // test helper
  constexpr bool test_eq(strong_ordering const &other) const noexcept {
    return __value_ == other.__value_;
  }

private:
  _ValueT __value_;
};

inline constexpr strong_ordering strong_ordering::less(_OrdResult::__less);
inline constexpr strong_ordering strong_ordering::equal(_EqResult::__equal);
inline constexpr strong_ordering strong_ordering::equivalent(_EqResult::__equiv);
inline constexpr strong_ordering strong_ordering::greater(_OrdResult::__greater);

constexpr bool operator==(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ == 0;
}
constexpr bool operator!=(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ != 0;
}
constexpr bool operator<(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ < 0;
}
constexpr bool operator<=(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ <= 0;
}
constexpr bool operator>(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ > 0;
}
constexpr bool operator>=(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v.__value_ >= 0;
}
constexpr bool operator==(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return 0 == __v.__value_;
}
constexpr bool operator!=(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return 0 != __v.__value_;
}
constexpr bool operator<(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return 0 < __v.__value_;
}
constexpr bool operator<=(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return 0 <= __v.__value_;
}
constexpr bool operator>(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return 0 > __v.__value_;
}
constexpr bool operator>=(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return 0 >= __v.__value_;
}

constexpr strong_ordering operator<=>(strong_ordering __v, _CmpUnspecifiedParam) noexcept {
  return __v;
}
constexpr strong_ordering operator<=>(_CmpUnspecifiedParam, strong_ordering __v) noexcept {
  return __v < 0 ? strong_ordering::greater : (__v > 0 ? strong_ordering::less : __v);
}

// named comparison functions
constexpr bool is_eq(weak_equality __cmp) noexcept { return __cmp == 0; }
constexpr bool is_neq(weak_equality __cmp) noexcept { return __cmp != 0; }
constexpr bool is_lt(partial_ordering __cmp) noexcept { return __cmp < 0; }
constexpr bool is_lteq(partial_ordering __cmp) noexcept { return __cmp <= 0; }
constexpr bool is_gt(partial_ordering __cmp) noexcept { return __cmp > 0; }
constexpr bool is_gteq(partial_ordering __cmp) noexcept { return __cmp >= 0; }

} // namespace __1
} // end namespace std

#endif // STD_COMPARE_H
