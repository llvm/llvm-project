#include "benchmark/benchmark.h"

#include <cstdlib>
#include <map>

namespace {

std::map<int, int> ConstructRandomMap(int size) {
  std::map<int, int> m;
  for (int i = 0; i < size; ++i) {
    m.insert(std::make_pair(rand() % size, rand() % size));
  }
  return m;
}

}  // namespace

// Basic version.
static void BM_MapLookup(benchmark::State& state) {
  const int size = state.range_x();
  while (state.KeepRunning()) {
    state.PauseTiming();
    std::map<int, int> m = ConstructRandomMap(size);
    state.ResumeTiming();
    for (int i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(m.find(rand() % size));
    }
  }
  state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_MapLookup)->Range(1 << 3, 1 << 12);

// Using fixtures.
class MapFixture : public ::benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& st) {
    m = ConstructRandomMap(st.range_x());
  }

  void TearDown(const ::benchmark::State&) {
    m.clear();
  }

  std::map<int, int> m;
};

BENCHMARK_DEFINE_F(MapFixture, Lookup)(benchmark::State& state) {
  const int size = state.range_x();
  while (state.KeepRunning()) {
    for (int i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(m.find(rand() % size));
    }
  }
  state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK_REGISTER_F(MapFixture, Lookup)->Range(1<<3, 1<<12);

BENCHMARK_MAIN()
