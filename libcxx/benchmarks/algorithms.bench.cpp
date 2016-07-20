#include <unordered_set>
#include <vector>
#include <cstdint>

#include "benchmark/benchmark_api.h"
#include "GenerateInput.hpp"

constexpr std::size_t TestNumInputs = 1024;

template <class GenInputs>
void BM_Sort(benchmark::State& st, GenInputs gen) {
    using ValueType = typename decltype(gen(0))::value_type;
    const auto in = gen(st.range_x());
    std::vector<ValueType> inputs[5];
    auto reset_inputs = [&]() {
        for (auto& C : inputs) {
            C = in;
            benchmark::DoNotOptimize(C.data());
        }
    };
    reset_inputs();
    while (st.KeepRunning()) {
        for (auto& I : inputs) {
            std::sort(I.data(), I.data() + I.size());
            benchmark::DoNotOptimize(I.data());
        }
        st.PauseTiming();
        reset_inputs();
        benchmark::ClobberMemory();
        st.ResumeTiming();
    }
}

BENCHMARK_CAPTURE(BM_Sort, random_uint32,
    getRandomIntegerInputs<uint32_t>)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, sorted_ascending_uint32,
    getSortedIntegerInputs<uint32_t>)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, sorted_descending_uint32,
    getReverseSortedIntegerInputs<uint32_t>)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, single_element_uint32,
    getDuplicateIntegerInputs<uint32_t>)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, pipe_organ_uint32,
    getPipeOrganIntegerInputs<uint32_t>)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, random_strings,
    getRandomStringInputs)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, sorted_ascending_strings,
    getSortedStringInputs)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, sorted_descending_strings,
    getReverseSortedStringInputs)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Sort, single_element_strings,
    getDuplicateStringInputs)->Arg(TestNumInputs);


BENCHMARK_MAIN()
