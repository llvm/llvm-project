//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <forward_list>
#include <iterator>
#include <set>
#include <vector>

#include "common.h"

namespace {

// types of containers we'll want to test, covering interesting iterator types
struct VectorContainer {
  template <typename... Args>
  using type = std::vector<Args...>;

  static constexpr const char* Name = "Vector";
};

struct SetContainer {
  template <typename... Args>
  using type = std::set<Args...>;

  static constexpr const char* Name = "Set";
};

struct ForwardListContainer {
  template <typename... Args>
  using type = std::forward_list<Args...>;

  static constexpr const char* Name = "ForwardList";
};

using AllContainerTypes = std::tuple<VectorContainer, SetContainer, ForwardListContainer>;

// set_intersection performance may depend on where matching values lie
enum class OverlapPosition {
    Nowhere,
    Front,
    Back,
    Interlaced,
};

struct AllOverlapPositions : EnumValuesAsTuple<AllOverlapPositions, OverlapPosition, 4> {
  static constexpr const char* Names[] = {
      "Nowhere", "Front", "Back", "Interlaced"};
};

// functor that moves elements from an iterator range into a new Container instance
template <typename Container>
struct MoveInto {};

template <typename T>
struct MoveInto<std::vector<T>> {
    template <class It>
    [[nodiscard]] static std::vector<T> operator()(It first, It last) {
        std::vector<T> out;
        std::move(first, last, std::back_inserter(out));
        return out;
    }
};

template <typename T>
struct MoveInto<std::forward_list<T>> {
    template <class It>
    [[nodiscard]] static std::forward_list<T> operator()(It first, It last) {
        std::forward_list<T> out;
        std::move(first, last, std::front_inserter(out));
        out.reverse();
        return out;
    }
};

template <typename T>
struct MoveInto<std::set<T>> {
    template <class It>
    [[nodiscard]] static std::set<T> operator()(It first, It last) {
        std::set<T> out;
        std::move(first, last, std::inserter(out, out.begin()));
        return out;
    }
};

// lightweight wrapping around fillValues() which puts a little effort into
// making that would be contiguous when sorted non-contiguous in memory
template <typename T>
std::vector<T> getVectorOfRandom(size_t N) {
  std::vector<T> V;
  fillValues(V, N, Order::Random);
  sortValues(V, Order::Random);
  return std::vector<T>(V);
}

// forward_iterator wrapping which, for each increment, moves the underlying iterator forward Stride elements
template <typename Wrapped>
struct StridedFwdIt {
  Wrapped Base;
  unsigned Stride;

  using iterator_category = std::forward_iterator_tag;
  using difference_type = typename Wrapped::difference_type;
  using value_type = typename Wrapped::value_type;
  using pointer = typename Wrapped::pointer;
  using reference = typename Wrapped::reference;

  StridedFwdIt(Wrapped B, unsigned Stride_) : Base(B), Stride(Stride_) { assert(Stride != 0); }

  StridedFwdIt operator++() { for (unsigned I=0; I<Stride; ++I) ++Base; return *this; }
  StridedFwdIt operator++(int) { auto Tmp = *this; ++*this; return Tmp; }
  value_type& operator*() { return *Base; }
  const value_type& operator*() const { return *Base; }
  value_type& operator->() { return *Base; }
  const value_type& operator->() const { return *Base; }
  bool operator==(const StridedFwdIt& o) const { return Base==o.Base; }
  bool operator!=(const StridedFwdIt& o) const { return !operator==(o); }
};
template <typename Wrapped> StridedFwdIt(Wrapped, unsigned) -> StridedFwdIt<Wrapped>;


// realistically, data won't all be nicely contiguous in a container
// we'll go through some effort to ensure that it's shuffled through memory
template <class Container>
std::pair<Container, Container> genCacheUnfriendlyData(size_t Size1, size_t Size2, OverlapPosition Pos) {
  using ValueType = typename Container::value_type;
  const MoveInto<Container> moveInto;
  const auto SrcSize = Pos == OverlapPosition::Nowhere ? Size1 + Size2 : std::max(Size1, Size2);
  std::vector<ValueType> Src = getVectorOfRandom<ValueType>(SrcSize);

  if (Pos == OverlapPosition::Nowhere) {
    std::sort(Src.begin(), Src.end());
    return std::make_pair(
        moveInto(Src.begin(), Src.begin() + Size1),
        moveInto(Src.begin() + Size1, Src.end()));
  }

  // all other overlap types will have to copy some part of the data, but if
  // we copy after sorting it will likely have high cache locality, so we sort
  // each copy separately
  auto Copy = Src;
  std::sort(Src.begin(), Src.end());
  std::sort(Copy.begin(), Copy.end());

  switch(Pos) {
    case OverlapPosition::Nowhere:
      break;

    case OverlapPosition::Front:
      return std::make_pair(
          moveInto(Src.begin(), Src.begin() + Size1),
          moveInto(Copy.begin(), Copy.begin() + Size2));

    case OverlapPosition::Back:
      return std::make_pair(
          moveInto(Src.begin() + (Src.size() - Size1), Src.end()),
          moveInto(Copy.begin() + (Copy.size() - Size2), Copy.end()));

    case OverlapPosition::Interlaced:
      const auto Stride1 = Size1 < Size2 ? Size2/Size1 : 1;
      const auto Stride2 = Size2 < Size1 ? Size1/Size2 : 1;
      return std::make_pair(
          moveInto(StridedFwdIt(Src.begin(), Stride1), StridedFwdIt(Src.end(), Stride1)),
          moveInto(StridedFwdIt(Copy.begin(), Stride2), StridedFwdIt(Copy.end(), Stride2)));
  }
  abort();
  return std::pair<Container, Container>();
}


template <class ValueType, class Container, class Overlap>
struct SetIntersection {
  using ContainerType = typename Container::template type<Value<ValueType>>;
  size_t Size1;
  size_t Size2;

  SetIntersection(size_t M, size_t N) : Size1(M), Size2(N) {}

  void run(benchmark::State& state) const {
    state.PauseTiming();
    auto Input = genCacheUnfriendlyData<ContainerType>(Size1, Size2, Overlap());
    std::vector<Value<ValueType>> out(std::min(Size1, Size2));

    size_t cmp;
    auto trackingLess = [&cmp](const Value<ValueType>& lhs, const Value<ValueType>& rhs) {
        ++cmp;
        return std::less<Value<ValueType>>{}(lhs, rhs);
    };

    const auto BatchSize =  std::max(size_t{16}, (2*TestSetElements) / (Size1+Size2));
    state.ResumeTiming();

    for (const auto& _ : state) {
      while (state.KeepRunningBatch(BatchSize)) {
        for (unsigned i=0; i<BatchSize; ++i) {
          const auto& [C1, C2] = Input;
          auto outIter = std::set_intersection(C1.begin(), C1.end(), C2.begin(), C2.end(), out.begin(), trackingLess);
          benchmark::DoNotOptimize(outIter);
          state.counters["Comparisons"] = cmp;
        }
      }
    }
  }

  std::string name() const {
    return std::string("SetIntersection") + Overlap::name() + '_' + Container::Name +
        ValueType::name() + '_' + std::to_string(Size1) + '_' + std::to_string(Size2);
  }
};

} // namespace

int main(int argc, char** argv) {/**/
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  makeCartesianProductBenchmark<SetIntersection, AllValueTypes, AllContainerTypes, AllOverlapPositions>(Quantities, Quantities);
  benchmark::RunSpecifiedBenchmarks();
}
