//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Algorithmic complexity tests for both std::set_intersection and std::ranges::set_intersection

// template<InputIterator InIter1, InputIterator InIter2, typename OutIter>
//   requires OutputIterator<OutIter, InIter1::reference>
//         && OutputIterator<OutIter, InIter2::reference>
//         && HasLess<InIter2::value_type, InIter1::value_type>
//         && HasLess<InIter1::value_type, InIter2::value_type>
//   constexpr OutIter       // constexpr after C++17
//   set_intersection(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
//                    OutIter result);
//
// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          weakly_incrementable O, class Comp = ranges::less,
//          class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<I1, I2, O, Comp, Proj1, Proj2>
//   constexpr set_intersection_result<I1, I2, O>
//     set_intersection(I1 first1, S1 last1, I2 first2, S2 last2, O result,
//                      Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                         // since C++20
//
// template<input_range R1, input_range R2, weakly_incrementable O,
//          class Comp = ranges::less, class Proj1 = identity, class Proj2 = identity>
//   requires mergeable<iterator_t<R1>, iterator_t<R2>, O, Comp, Proj1, Proj2>
//   constexpr set_intersection_result<borrowed_iterator_t<R1>, borrowed_iterator_t<R2>, O>
//     set_intersection(R1&& r1, R2&& r2, O result,
//                      Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});                         // since C++20

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>

#include "test_iterators.h"

namespace {

// __debug_less will perform an additional comparison in an assertion
static constexpr unsigned std_less_comparison_count_multiplier() noexcept {
#if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
  return 2;
#else
  return 1;
#endif
}

struct [[nodiscard]] OperationCounts {
  std::size_t comparisons{};
  struct PerInput {
    std::size_t proj{};
    IteratorOpCounts iterops;

    [[nodiscard]] constexpr bool isNotBetterThan(const PerInput& other) {
      return proj >= other.proj && iterops.increments + iterops.decrements + iterops.zero_moves >=
                                       other.iterops.increments + other.iterops.decrements + other.iterops.zero_moves;
    }
  };
  std::array<PerInput, 2> in;

  [[nodiscard]] constexpr bool isNotBetterThan(const OperationCounts& expect) {
    return std_less_comparison_count_multiplier() * comparisons >= expect.comparisons &&
           in[0].isNotBetterThan(expect.in[0]) && in[1].isNotBetterThan(expect.in[1]);
  }
};

template <std::size_t ResultSize>
struct counted_set_intersection_result {
  std::array<int, ResultSize> result;
  OperationCounts opcounts;

  constexpr counted_set_intersection_result() = default;

  constexpr explicit counted_set_intersection_result(std::array<int, ResultSize>&& contents) : result{contents} {}

  constexpr void assertNotBetterThan(const counted_set_intersection_result& other) {
    assert(result == other.result);
    assert(opcounts.isNotBetterThan(other.opcounts));
  }
};

template <std::size_t ResultSize>
counted_set_intersection_result(std::array<int, ResultSize>) -> counted_set_intersection_result<ResultSize>;

template <template <class...> class InIterType1,
          template <class...>
          class InIterType2,
          class OutIterType,
          std::size_t ResultSize,
          std::ranges::input_range R1,
          std::ranges::input_range R2>
constexpr counted_set_intersection_result<ResultSize> counted_set_intersection(const R1& in1, const R2& in2) {
  counted_set_intersection_result<ResultSize> out;

  const auto comp = [&out](int x, int y) {
    ++out.opcounts.comparisons;
    return x < y;
  };

  operation_counting_iterator b1(InIterType1<decltype(in1.begin())>(in1.begin()), &out.opcounts.in[0].iterops);
  operation_counting_iterator e1(InIterType1<decltype(in1.end()) >(in1.end()), &out.opcounts.in[0].iterops);

  operation_counting_iterator b2(InIterType2<decltype(in2.begin())>(in2.begin()), &out.opcounts.in[1].iterops);
  operation_counting_iterator e2(InIterType2<decltype(in2.end()) >(in2.end()), &out.opcounts.in[1].iterops);

  std::set_intersection(b1, e1, b2, e2, OutIterType(out.result.data()), comp);

  return out;
}

template <template <class...> class InIterType1,
          template <class...>
          class InIterType2,
          class OutIterType,
          std::size_t ResultSize,
          std::ranges::input_range R1,
          std::ranges::input_range R2>
constexpr counted_set_intersection_result<ResultSize> counted_ranges_set_intersection(const R1& in1, const R2& in2) {
  counted_set_intersection_result<ResultSize> out;

  const auto comp = [&out](int x, int y) {
    ++out.opcounts.comparisons;
    return x < y;
  };

  const auto proj1 = [&out](const int& i) {
    ++out.opcounts.in[0].proj;
    return i;
  };

  const auto proj2 = [&out](const int& i) {
    ++out.opcounts.in[1].proj;
    return i;
  };

  operation_counting_iterator b1(InIterType1<decltype(in1.begin())>(in1.begin()), &out.opcounts.in[0].iterops);
  operation_counting_iterator e1(InIterType1<decltype(in1.end()) >(in1.end()), &out.opcounts.in[0].iterops);

  operation_counting_iterator b2(InIterType2<decltype(in2.begin())>(in2.begin()), &out.opcounts.in[1].iterops);
  operation_counting_iterator e2(InIterType2<decltype(in2.end()) >(in2.end()), &out.opcounts.in[1].iterops);

  std::ranges::subrange r1{b1, sentinel_wrapper<decltype(e1)>{e1}};
  std::ranges::subrange r2{b2, sentinel_wrapper<decltype(e2)>{e2}};
  std::same_as<std::ranges::set_intersection_result<decltype(e1), decltype(e2), OutIterType>> decltype(auto) result =
      std::ranges::set_intersection(r1, r2, OutIterType{out.result.data()}, comp, proj1, proj2);
  assert(base(result.in1) == base(e1));
  assert(base(result.in2) == base(e2));
  assert(base(result.out) == out.result.data() + out.result.size());

  return out;
}

template <template <typename...> class In1, template <typename...> class In2, class Out>
constexpr void testComplexityParameterizedIter() {
  // Worst-case complexity:
  // Let N=(last1 - first1) and M=(last2 - first2)
  // At most 2*(N+M) - 1 comparisons and applications of each projection.
  // At most 2*(N+M) iterator mutations.
  {
    std::array r1{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    std::array r2{2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    counted_set_intersection_result<0> expected;
    expected.opcounts.comparisons              = 37;
    expected.opcounts.in[0].proj               = 37;
    expected.opcounts.in[0].iterops.increments = 30;
    expected.opcounts.in[0].iterops.decrements = 0;
    expected.opcounts.in[1]                    = expected.opcounts.in[0];

    expected.assertNotBetterThan(counted_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
    expected.assertNotBetterThan(counted_ranges_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
  }

  {
    std::array r1{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    std::array r2{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

    counted_set_intersection_result expected(std::array{1, 3, 5, 7, 9, 11, 13, 15, 17, 19});
    expected.opcounts.comparisons              = 38;
    expected.opcounts.in[0].proj               = 38;
    expected.opcounts.in[0].iterops.increments = 30;
    expected.opcounts.in[0].iterops.decrements = 0;
    expected.opcounts.in[1]                    = expected.opcounts.in[0];

    expected.assertNotBetterThan(counted_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
    expected.assertNotBetterThan(counted_ranges_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
  }

  // Lower complexity when there is low overlap between ranges: we can make 2*log(X) comparisons when one range
  // has X elements that can be skipped over (and then 1 more to confirm that the value we found is equal).
  {
    std::array r1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::array r2{15};

    counted_set_intersection_result expected(std::array{15});
    expected.opcounts.comparisons              = 9;
    expected.opcounts.in[0].proj               = 9;
    expected.opcounts.in[0].iterops.increments = 23;
    expected.opcounts.in[0].iterops.decrements = 0;
    expected.opcounts.in[1].proj               = 9;
    expected.opcounts.in[1].iterops.increments = 1;
    expected.opcounts.in[1].iterops.decrements = 0;

    expected.assertNotBetterThan(counted_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
    expected.assertNotBetterThan(counted_ranges_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
  }

  {
    std::array r1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::array r2{0, 16};
    counted_set_intersection_result<0> expected;

    expected.opcounts.comparisons              = 10;
    expected.opcounts.in[0].proj               = 10;
    expected.opcounts.in[0].iterops.increments = 24;
    expected.opcounts.in[0].iterops.decrements = 0;
    expected.opcounts.in[1].proj               = 10;
    expected.opcounts.in[1].iterops.increments = 4;
    expected.opcounts.in[1].iterops.decrements = 0;

    expected.assertNotBetterThan(counted_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
    expected.assertNotBetterThan(counted_ranges_set_intersection<In1, In2, Out, expected.result.size()>(r1, r2));
  }
}

template <template <typename...> class In2, class Out>
constexpr void testComplexityParameterizedIterPermutateIn1() {
  //common_input_iterator
  testComplexityParameterizedIter<forward_iterator, In2, Out>();
  testComplexityParameterizedIter<bidirectional_iterator, In2, Out>();
  testComplexityParameterizedIter<random_access_iterator, In2, Out>();
}

template <class Out>
constexpr void testComplexityParameterizedIterPermutateIn1In2() {
  testComplexityParameterizedIterPermutateIn1<forward_iterator, Out>();
  testComplexityParameterizedIterPermutateIn1<bidirectional_iterator, Out>();
  testComplexityParameterizedIterPermutateIn1<random_access_iterator, Out>();
}

constexpr bool testComplexity() {
  testComplexityParameterizedIterPermutateIn1In2<forward_iterator<int*>>();
  testComplexityParameterizedIterPermutateIn1In2<bidirectional_iterator<int*>>();
  testComplexityParameterizedIterPermutateIn1In2<random_access_iterator<int*>>();
  return true;
}

template <template <typename...> class In1, template <typename...> class In2, class Out>
constexpr void testComplexityGuaranteesParameterizedIter() {
  // now a more generic validation of the complexity guarantees when searching for a single value
  for (unsigned range_size = 1; range_size < 20; ++range_size) {
    std::ranges::iota_view<int, int> r1(0, range_size);
    for (int i : r1) {
      // At most 2 * ((last1 - first1) + (last2 - first2)) - 1 comparisons
      counted_set_intersection_result<1> expected(std::array{i});
      expected.opcounts.comparisons              = 2 * (r1.size() + 1) - 1;
      expected.opcounts.in[0].proj               = expected.opcounts.comparisons;
      expected.opcounts.in[1].proj               = expected.opcounts.comparisons;
      expected.opcounts.in[0].iterops.increments = 2 * r1.size();
      expected.opcounts.in[1].iterops.increments = 2;
      expected.opcounts.in[0].iterops.decrements = expected.opcounts.in[0].iterops.increments;
      expected.opcounts.in[1].iterops.decrements = expected.opcounts.in[1].iterops.increments;

      expected.assertNotBetterThan(
          counted_set_intersection<In1, In2, Out, expected.result.size()>(r1, expected.result));
      expected.assertNotBetterThan(
          counted_ranges_set_intersection<In1, In2, Out, expected.result.size()>(r1, expected.result));
    }
  }
}

template <template <typename...> class In2, class Out>
constexpr void testComplexityGuaranteesParameterizedIterPermutateIn1() {
  //common_input_iterator
  testComplexityGuaranteesParameterizedIter<forward_iterator, In2, Out>();
  testComplexityGuaranteesParameterizedIter<bidirectional_iterator, In2, Out>();
  testComplexityGuaranteesParameterizedIter<random_access_iterator, In2, Out>();
}

template <class Out>
constexpr void testComplexityGuaranteesParameterizedIterPermutateIn1In2() {
  testComplexityGuaranteesParameterizedIterPermutateIn1<forward_iterator, Out>();
  testComplexityGuaranteesParameterizedIterPermutateIn1<bidirectional_iterator, Out>();
  testComplexityGuaranteesParameterizedIterPermutateIn1<random_access_iterator, Out>();
}

constexpr bool testComplexityGuarantees() {
  testComplexityGuaranteesParameterizedIterPermutateIn1In2<forward_iterator<int*>>();
  testComplexityGuaranteesParameterizedIterPermutateIn1In2<bidirectional_iterator<int*>>();
  testComplexityGuaranteesParameterizedIterPermutateIn1In2<random_access_iterator<int*>>();
  return true;
}

constexpr bool testComplexityBasic() {
  // Complexity: At most 2 * ((last1 - first1) + (last2 - first2)) - 1 comparisons and applications of each projection.
  std::array<int, 5> r1{1, 3, 5, 7, 9};
  std::array<int, 5> r2{2, 4, 6, 8, 10};
  std::array<int, 0> expected{};

  const std::size_t maxOperation = std_less_comparison_count_multiplier() * (2 * (r1.size() + r2.size()) - 1);

  // std::set_intersection
  {
    std::array<int, 0> out{};
    std::size_t numberOfComp = 0;

    const auto comp = [&numberOfComp](int x, int y) {
      ++numberOfComp;
      return x < y;
    };

    std::set_intersection(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), comp);

    assert(std::ranges::equal(out, expected));
    assert(numberOfComp <= maxOperation);
  }

  // ranges::set_intersection iterator overload
  {
    std::array<int, 0> out{};
    std::size_t numberOfComp  = 0;
    std::size_t numberOfProj1 = 0;
    std::size_t numberOfProj2 = 0;

    const auto comp = [&numberOfComp](int x, int y) {
      ++numberOfComp;
      return x < y;
    };

    const auto proj1 = [&numberOfProj1](int d) {
      ++numberOfProj1;
      return d;
    };

    const auto proj2 = [&numberOfProj2](int d) {
      ++numberOfProj2;
      return d;
    };

    std::ranges::set_intersection(r1.begin(), r1.end(), r2.begin(), r2.end(), out.data(), comp, proj1, proj2);

    assert(std::ranges::equal(out, expected));
    assert(numberOfComp <= maxOperation);
    assert(numberOfProj1 <= maxOperation);
    assert(numberOfProj2 <= maxOperation);
  }

  // ranges::set_intersection range overload
  {
    std::array<int, 0> out{};
    std::size_t numberOfComp  = 0;
    std::size_t numberOfProj1 = 0;
    std::size_t numberOfProj2 = 0;

    const auto comp = [&numberOfComp](int x, int y) {
      ++numberOfComp;
      return x < y;
    };

    const auto proj1 = [&numberOfProj1](int d) {
      ++numberOfProj1;
      return d;
    };

    const auto proj2 = [&numberOfProj2](int d) {
      ++numberOfProj2;
      return d;
    };

    std::ranges::set_intersection(r1, r2, out.data(), comp, proj1, proj2);

    assert(std::ranges::equal(out, expected));
    assert(numberOfComp < maxOperation);
    assert(numberOfProj1 < maxOperation);
    assert(numberOfProj2 < maxOperation);
  }
  return true;
}

} // unnamed namespace

int main(int, char**) {
  testComplexityBasic();
  testComplexity();
  testComplexityGuarantees();

  static_assert(testComplexityBasic());
  static_assert(testComplexity());

  // we hit maximum constexpr evaluation step limit even if we split this into
  // the 3 types of the first type layer, so let's skip the constexpr validation
  // static_assert(testComplexityGuarantees());

  return 0;
}
