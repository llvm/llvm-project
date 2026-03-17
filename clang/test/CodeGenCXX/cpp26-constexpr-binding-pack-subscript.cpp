// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// static constexpr structured binding pack elements used as array
// subscript indices must be constant-folded at the Clang codegen level.
// Without the fix, each element is emitted as a load from a "reference
// temporary" static variable.

namespace std {
  using size_t = decltype(sizeof(0));
  template <typename T> struct tuple_size;
  template <size_t I, typename T> struct tuple_element;
} // namespace std

using u8 = unsigned char;

template <u8 N>
struct Range {
  template <std::size_t I>
  consteval friend u8 get(Range) noexcept { return I; }
};

namespace std {
  template <u8 N>
  struct tuple_size<Range<N>> { static constexpr std::size_t value = N; };
  template <std::size_t I, u8 N>
  struct tuple_element<I, Range<N>> { using type = u8; };
} // namespace std

template <std::size_t N>
struct Array {
  u8 data[N];
  constexpr const u8 &operator[](std::size_t i) const { return data[i]; }
};

template <std::size_t L, std::size_t R>
__attribute__((always_inline)) inline constexpr Array<L + R>
concat(const Array<L> &l, const Array<R> &r) {
  static constexpr auto [...I] = Range<L>{};
  static constexpr auto [...J] = Range<R>{};
  return {{l[I]..., r[J]...}};
}

Array<32> test(const Array<16> &l, const Array<16> &r) {
  return concat(l, r);
}

// The binding-pack indices must not be materialised as "reference temporary"
// static variables at any optimisation level.
// CHECK-LABEL: define {{.*}} @{{.*test.*}}
// CHECK-NOT:   reference temporary
