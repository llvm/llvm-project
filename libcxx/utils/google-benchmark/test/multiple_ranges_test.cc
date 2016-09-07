#include "benchmark/benchmark.h"

#include <set>
#include <cassert>

class MultipleRangesFixture : public ::benchmark::Fixture {
 public:
  MultipleRangesFixture()
      : expectedValues({
        {1, 3, 5}, {1, 3, 8}, {1, 3, 15}, {2, 3, 5}, {2, 3, 8}, {2, 3, 15},
        {1, 4, 5}, {1, 4, 8}, {1, 4, 15}, {2, 4, 5}, {2, 4, 8}, {2, 4, 15},
        {1, 7, 5}, {1, 7, 8}, {1, 7, 15}, {2, 7, 5}, {2, 7, 8}, {2, 7, 15},
        {7, 6, 3}
      })
  {
  }

  void SetUp(const ::benchmark::State& state) {
    std::vector<int> ranges = {state.range(0), state.range(1), state.range(2)};

    assert(expectedValues.find(ranges) != expectedValues.end());

    actualValues.insert(ranges);
  }

  virtual ~MultipleRangesFixture() {
    assert(actualValues.size() == expectedValues.size());
  }
  
  std::set<std::vector<int>> expectedValues;
  std::set<std::vector<int>> actualValues;
};


BENCHMARK_DEFINE_F(MultipleRangesFixture, Empty)(benchmark::State& state) {
  while (state.KeepRunning()) {
    int product = state.range(0) * state.range(1) * state.range(2);
    for (int x = 0; x < product; x++) {
      benchmark::DoNotOptimize(x);
    }
  }
}

BENCHMARK_REGISTER_F(MultipleRangesFixture, Empty)->RangeMultiplier(2)
    ->Ranges({{1, 2}, {3, 7}, {5, 15}})->Args({7, 6, 3});

void BM_CheckDefaultArgument(benchmark::State& state) {
  // Test that the 'range()' without an argument is the same as 'range(0)'.
  assert(state.range() == state.range(0));
  assert(state.range() != state.range(1));
  while (state.KeepRunning()) {}
}
BENCHMARK(BM_CheckDefaultArgument)->Ranges({{1, 5}, {6, 10}});

static void BM_MultipleRanges(benchmark::State& st) {
    while (st.KeepRunning()) {}
}
BENCHMARK(BM_MultipleRanges)->Ranges({{5, 5}, {6, 6}});


BENCHMARK_MAIN()
