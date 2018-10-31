#include <unordered_set>
#include <vector>
#include <cstdint>

#include "benchmark/benchmark.h"
#include "GenerateInput.hpp"

constexpr std::size_t TestNumInputs = 1024;

template <class GenInputs>
void BM_Sort(benchmark::State& st, GenInputs gen) {
    using ValueType = typename decltype(gen(0))::value_type;
    const auto in = gen(st.range(0));
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

template <typename GenInputs, typename Alg>
void do_binary_search_benchmark(benchmark::State& st, GenInputs gen, Alg alg)
{
    using ValueType = typename decltype(gen(0))::value_type;
    auto in = gen(st.range(0));
    std::sort(in.begin(), in.end());

    const auto every_10_percentile = [&]() -> std::vector<ValueType*>  {
        size_t step = in.size() / 10;

        if (step == 0) {
            st.SkipWithError("Input doesn't contain enough elements");
            return {};
        }

        std::vector<ValueType*> res;
        for (size_t i = 0; i < in.size(); i += step)
            res.push_back(&in[i]);

        return res;
    }();

    for (auto _ : st)
    {
        for (auto* test : every_10_percentile)
          benchmark::DoNotOptimize(alg(in.begin(), in.end(), *test));
    }
}

template <typename GenInputs>
void BM_LowerBound(benchmark::State& st, GenInputs gen)
{
    do_binary_search_benchmark(st, gen, [](auto f, auto l, const auto& v) {
        return std::lower_bound(f, l, v);
    });
}

BENCHMARK_CAPTURE(BM_LowerBound, random_int32, getRandomIntegerInputs<int32_t>)
    ->Arg(TestNumInputs)                    // Small int32_t vector
    ->Arg(TestNumInputs * TestNumInputs);   // Big int32_t   vector

BENCHMARK_CAPTURE(BM_LowerBound, random_int64, getRandomIntegerInputs<int64_t>)
    ->Arg(TestNumInputs);  // Small int64_t vector. Should also represent pointers.

BENCHMARK_CAPTURE(BM_LowerBound, random_strings, getRandomStringInputs)
    ->Arg(TestNumInputs);  // Small string vector. What happens if the comparison is not very cheap.

template <typename GenInputs>
void BM_EqualRange(benchmark::State& st, GenInputs gen)
{
    do_binary_search_benchmark(st, gen, [](auto f, auto l, const auto& v) {
        return std::equal_range(f, l, v);
    });
}

BENCHMARK_CAPTURE(BM_EqualRange, random_int32, getRandomIntegerInputs<int32_t>)
    ->Arg(TestNumInputs)                    // Small int32_t vector
    ->Arg(TestNumInputs * TestNumInputs);   // Big int32_t   vector

BENCHMARK_CAPTURE(BM_EqualRange, random_int64, getRandomIntegerInputs<int64_t>)
    ->Arg(TestNumInputs);  // Small int64_t vector. Should also represent pointers.

BENCHMARK_CAPTURE(BM_EqualRange, random_strings, getRandomStringInputs)
    ->Arg(TestNumInputs);  // Small string vector. What happens if the comparison is not very cheap.

BENCHMARK_MAIN();
