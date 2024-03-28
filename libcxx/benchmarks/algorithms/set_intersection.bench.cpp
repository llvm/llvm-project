//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>
#include <set>
#include <stdlib.h>
#include <vector>

#include "common.h"
#include "test_iterators.h"

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

using AllContainerTypes = std::tuple<VectorContainer, SetContainer>;

// set_intersection performance may depend on where matching values lie
enum class OverlapPosition {
  None,
  Front,
  // performance-wise, matches at the back are identical to ones at the front
  Interlaced,
};

struct AllOverlapPositions : EnumValuesAsTuple<AllOverlapPositions, OverlapPosition, 3> {
  static constexpr const char* Names[] = {"None", "Front", "Interlaced"};
};

// functor that moves elements from an iterator range into a new Container instance
template <typename Container>
struct MoveInto {
  template <class It>
  [[nodiscard]] static Container operator()(It first, It last) {
    Container out;
    std::move(first, last, std::inserter(out, out.begin()));
    return out;
  }
};

// lightweight wrapping around fillValues() which puts a little effort into
// making that would be contiguous when sorted non-contiguous in memory
template <typename T>
std::vector<T> getVectorOfRandom(size_t N) {
  std::vector<T> v;
  fillValues(v, N, Order::Random);
  sortValues(v, Order::Random);
  return std::vector<T>(v);
}

// forward_iterator wrapping which, for each increment, moves the underlying iterator forward Stride elements
template <typename Wrapped>
struct StridedFwdIt {
  Wrapped base_;
  unsigned stride_;

  using iterator_category = std::forward_iterator_tag;
  using difference_type   = typename Wrapped::difference_type;
  using value_type        = typename Wrapped::value_type;
  using pointer           = typename Wrapped::pointer;
  using reference         = typename Wrapped::reference;

  StridedFwdIt(Wrapped base, unsigned stride) : base_(base), stride_(stride) { assert(stride_ != 0); }

  StridedFwdIt operator++() {
    for (unsigned i = 0; i < stride_; ++i)
      ++base_;
    return *this;
  }
  StridedFwdIt operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }
  value_type& operator*() { return *base_; }
  const value_type& operator*() const { return *base_; }
  value_type& operator->() { return *base_; }
  const value_type& operator->() const { return *base_; }
  bool operator==(const StridedFwdIt& o) const { return base_ == o.base_; }
  bool operator!=(const StridedFwdIt& o) const { return !operator==(o); }
};
template <typename Wrapped>
StridedFwdIt(Wrapped, unsigned) -> StridedFwdIt<Wrapped>;

// realistically, data won't all be nicely contiguous in a container
// we'll go through some effort to ensure that it's shuffled through memory
template <class Container>
std::pair<Container, Container> genCacheUnfriendlyData(size_t size1, size_t size2, OverlapPosition pos) {
  using ValueType = typename Container::value_type;
  const MoveInto<Container> move_into;
  const auto src_size        = pos == OverlapPosition::None ? size1 + size2 : std::max(size1, size2);
  std::vector<ValueType> src = getVectorOfRandom<ValueType>(src_size);

  if (pos == OverlapPosition::None) {
    std::sort(src.begin(), src.end());
    return std::make_pair(move_into(src.begin(), src.begin() + size1), move_into(src.begin() + size1, src.end()));
  }

  // all other overlap types will have to copy some part of the data, but if
  // we copy after sorting it will likely have high cache locality, so we sort
  // each copy separately
  auto copy = src;
  std::sort(src.begin(), src.end());
  std::sort(copy.begin(), copy.end());

  switch (pos) {
  case OverlapPosition::None:
    break;

  case OverlapPosition::Front:
    return std::make_pair(move_into(src.begin(), src.begin() + size1), move_into(copy.begin(), copy.begin() + size2));

  case OverlapPosition::Interlaced:
    const auto stride1 = size1 < size2 ? size2 / size1 : 1;
    const auto stride2 = size2 < size1 ? size1 / size2 : 1;
    return std::make_pair(move_into(StridedFwdIt(src.begin(), stride1), StridedFwdIt(src.end(), stride1)),
                          move_into(StridedFwdIt(copy.begin(), stride2), StridedFwdIt(copy.end(), stride2)));
  }
  abort();
  return std::pair<Container, Container>();
}

// use environment variable to enable additional counters: instrumentation will
// impact CPU utilisation, let's give the user the option
static const bool TRACK_COUNTERS = getenv("TRACK_COUNTERS") != nullptr;

template <class ValueType, class Container, class Overlap>
struct SetIntersection {
  using ContainerType = typename Container::template type<Value<ValueType>>;
  size_t size1_;
  size_t size2_;

  SetIntersection(size_t size1, size_t size2) : size1_(size1), size2_(size2) {}

  bool skip() const noexcept {
    // let's save some time and skip simmetrical runs
    return size1_ < size2_;
  }

  void run(benchmark::State& state) const {
    state.PauseTiming();
    auto input = genCacheUnfriendlyData<ContainerType>(size1_, size2_, Overlap());
    std::vector<Value<ValueType>> out(std::min(size1_, size2_));

    const auto BATCH_SIZE = std::max(size_t{512}, (2 * TestSetElements) / (size1_ + size2_));
    state.ResumeTiming();

    for (const auto& _ : state) {
      while (state.KeepRunningBatch(BATCH_SIZE)) {
        for (unsigned i = 0; i < BATCH_SIZE; ++i) {
          const auto& [c1, c2] = input;
          if (TRACK_COUNTERS) {
            size_t cmp{}, strides{}, displacement{};
            auto tracking_less = [&cmp](const Value<ValueType>& lhs, const Value<ValueType>& rhs) {
              ++cmp;
              return std::less<Value<ValueType>>{}(lhs, rhs);
            };
            stride_counting_iterator b1(c1.begin(), &strides, &displacement);
            stride_counting_iterator e1(c1.end(), &strides, &displacement);
            stride_counting_iterator b2(c2.begin(), &strides, &displacement);
            stride_counting_iterator e2(c2.end(), &strides, &displacement);
            auto res = std::set_intersection(b1, e1, b2, e2, out.begin(), tracking_less);
            benchmark::DoNotOptimize(res);
            state.counters["comparisons"]       = cmp;
            state.counters["iter_strides"]      = strides;
            state.counters["iter_displacement"] = displacement;

          } else {
            auto res = std::set_intersection(c1.begin(), c1.end(), c2.begin(), c2.end(), out.begin());
            benchmark::DoNotOptimize(res);
          }
        }
      }
    }
  }

  std::string name() const {
    return std::string("SetIntersection") + Overlap::name() + '_' + Container::Name + ValueType::name() + '_' +
           std::to_string(size1_) + '_' + std::to_string(size2_);
  }
};

} // namespace

int main(int argc, char** argv) { /**/
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  const std::vector<size_t> Quantities = {
    1 << 0,
    1 << 4,
    1 << 8,
    1 << 14,
// Running each benchmark in parallel consumes too much memory with MSAN
// and can lead to the test process being killed.
#if !TEST_HAS_FEATURE(memory_sanitizer)
    1 << 18
#endif
  };

  makeCartesianProductBenchmark<SetIntersection, AllValueTypes, AllContainerTypes, AllOverlapPositions>(
      Quantities, Quantities);
  benchmark::RunSpecifiedBenchmarks();
}
