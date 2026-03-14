// RUN: %clang -std=c++26 -O3 -emit-llvm -S -target x86_64-unknown-linux-gnu -o - %s | FileCheck %s
// RUN: %clang -std=c++26 -emit-llvm -S -target x86_64-unknown-linux-gnu -o - %s | FileCheck %s --check-prefix=IR
// UNSUPPORTED: system-windows

// static constexpr structured binding pack elements used as array
// subscript indices must be constant-folded at the Clang codegen level.
// Without the fix, each element is emitted as a load from a "reference
// temporary" static variable, preventing vectorisation at -O3.

#include <array>
#include <cstdint>

using u8 = std::uint8_t;

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

template <std::size_t L, std::size_t R>
__attribute__((always_inline)) inline constexpr std::array<u8, L + R>
concat(const std::array<u8, L> &l, const std::array<u8, R> &r) {
  static constexpr auto [...I] = Range<L>{};
  static constexpr auto [...J] = Range<R>{};
  return {l[I]..., r[J]...};
}

auto test(const std::array<u8, 16> &l, const std::array<u8, 16> &r) {
  return concat(l, r);
}

// At -O3 the two 16-byte arrays should be copied with a pair of vector
// loads/stores; no scalar byte loop and no reference-temporary indirection.
// CHECK-LABEL: define {{.*}} @{{.*test.*}}
// CHECK-NOT:   reference temporary
// CHECK:       load <16 x i8>
// CHECK:       store <16 x i8>
// CHECK:       load <16 x i8>
// CHECK:       store <16 x i8>
// CHECK:       ret void

// At any optimisation level the binding-pack indices must not be materialised
// as "reference temporary" static variables.
// IR-LABEL: define {{.*}} @{{.*test.*}}
// IR-NOT:   reference temporary
