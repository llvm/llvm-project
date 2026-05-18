// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

namespace std {
using size_t = decltype(sizeof(0));
template <typename T> struct tuple_size;
template <size_t I, typename T> struct tuple_element;

template <typename T> struct tuple_size<const T> : tuple_size<T> {};

template <size_t I, typename T> struct tuple_element<I, const T> {
  using type = const typename tuple_element<I, T>::type;
};
} // namespace std

using u8 = unsigned char;

template <u8 N> struct Range {
  template <std::size_t I>
  constexpr friend u8 get(Range) noexcept {
    return I;
  }
};

namespace std {
template <u8 N> struct tuple_size<Range<N>> {
  static constexpr std::size_t value = N;
};
template <std::size_t I, u8 N> struct tuple_element<I, Range<N>> {
  using type = u8;
};
} // namespace std

const u8 &f() {
  static constexpr auto [I] = Range<1>();
  return I;
}

// CHECK: @[[TMP:_ZGR.*]] = internal constant i8 0, align 1
// CHECK-LABEL: define {{.*}} @_Z1fv(
// CHECK: ret ptr @[[TMP]]
