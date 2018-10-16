//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <string>
#include <tuple>
#include <type_traits>

#include "benchmark/benchmark.h"
#include "test_macros.h"

namespace internal {

template <class D, class E, size_t I>
struct EnumValue : std::integral_constant<E, static_cast<E>(I)> {
  static std::string name() { return std::string("_") + D::Names[I]; }
};

template <class D, class E, size_t ...Idxs>
constexpr auto makeEnumValueTuple(std::index_sequence<Idxs...>) {
  return std::make_tuple(EnumValue<D, E, Idxs>{}...);
}

template <class T>
static auto skip(int) -> decltype(T::skip()) {
  return T::skip();
}
template <class T>
static bool skip(char) {
  return false;
}

template <template <class...> class B, class... U>
void makeBenchmarkImpl(std::tuple<U...> t) {
  using T = B<U...>;
  if (!internal::skip<T>(0))
    benchmark::RegisterBenchmark(T::name().c_str(), T::run);
}

template <template <class...> class B, class... U, class... T, class... Tuples>
void makeBenchmarkImpl(std::tuple<U...>, std::tuple<T...>, Tuples... rest) {
  (internal::makeBenchmarkImpl<B>(std::tuple<U..., T>(), rest...), ...);
}

}  // namespace internal

// CRTP class that enables using enum types as a dimension for
// makeCartesianProductBenchmark below.
// The type passed to `B` will be a std::integral_constant<E, e>, with the
// additional static function `name()` that returns the stringified name of the
// label.
//
// Eg:
// enum class MyEnum { A, B };
// struct AllMyEnum : EnumValuesAsTuple<AllMyEnum, MyEnum, 2> {
//   static constexpr absl::string_view Names[] = {"A", "B"};
// };
template <class Derived, class EnumType, size_t NumLabels>
using EnumValuesAsTuple =
    decltype(internal::makeEnumValueTuple<Derived, EnumType>(
        std::make_index_sequence<NumLabels>{}));

// Instantiates B<T0, T1, ..., TN> where <Ti...> are the combinations in the
// cartesian product of `Tuples...`
// B<T...> requires:
//  - static std::string name(): The name of the benchmark.
//  - static void run(benchmark::State&): The body of the benchmark.
// It can also optionally provide:
//  - static bool skip(): When `true`, skips the combination. Default is false.
//
// Returns int to facilitate registration. The return value is unspecified.
template <template <class...> class B, class... Tuples>
int makeCartesianProductBenchmark() {
  internal::makeBenchmarkImpl<B>(std::tuple<>(), Tuples()...);
  return 0;
}

// When `opaque` is true, this function hides the runtime state of `value` from
// the optimizer.
// It returns `value`.
template <class T>
TEST_ALWAYS_INLINE inline T maybeOpaque(T value, bool opaque) {
  if (opaque) benchmark::DoNotOptimize(value);
  return value;
}

