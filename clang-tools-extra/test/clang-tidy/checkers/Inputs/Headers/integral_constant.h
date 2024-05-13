#ifndef _INTEGRAL_CONSTANT_H_
#define _INTEGRAL_CONSTANT_H_

template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  typedef T value_type;
  typedef integral_constant type; // using injected-class-name
  constexpr operator value_type() const noexcept { return value; }
};

using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;

template <class T, class U>
struct is_same : false_type {};

template <class T>
struct is_same<T, T> : true_type {};

#endif
