#include <unordered_set>
#include <vector>
#include <cstdint>

#include "benchmark/benchmark_api.h"

template <class IntT>
std::vector<IntT> getInputs(size_t N) {
    std::vector<IntT> inputs;
    for (size_t i=0; i < N; ++i) {
        inputs.push_back(i);
    }
    return inputs;
}

template <class Container, class Inputs>
void BM_SetInsert(benchmark::State& st, Container c, Inputs const& in) {
    const auto end = in.end();
    while (st.KeepRunning()) {
        c.clear();
        for (auto it = in.begin(); it != end; ++it) {
            benchmark::DoNotOptimize(c.insert(*it));
        }
        benchmark::DoNotOptimize(c);
    }
}
BENCHMARK_CAPTURE(BM_SetInsert, uint32_insert,
    std::unordered_set<uint32_t>{}, getInputs<uint32_t>(1024));

template <class Container, class Inputs>
void BM_SetFind(benchmark::State& st, Container c, Inputs const& in) {
    c.insert(in.begin(), in.end());
    const auto end = in.end();
    while (st.KeepRunning()) {
        for (auto it = in.begin(); it != end; ++it) {
            benchmark::DoNotOptimize(c.find(*it));
        }
    }
}
BENCHMARK_CAPTURE(BM_SetFind, uint32_lookup,
    std::unordered_set<uint32_t>{}, getInputs<uint32_t>(1024));


BENCHMARK_MAIN()
