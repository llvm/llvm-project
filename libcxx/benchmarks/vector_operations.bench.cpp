#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "benchmark/benchmark.h"

#include "ContainerBenchmarks.h"
#include "GenerateInput.h"

using namespace ContainerBenchmarks;

constexpr std::size_t TestNumInputs = 1024;

BENCHMARK_CAPTURE(BM_ConstructSize, vector_byte, std::vector<unsigned char>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_CopyConstruct, vector_int, std::vector<int>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_Assignment, vector_int, std::vector<int>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructSizeValue, vector_byte, std::vector<unsigned char>{}, 0)->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructIterIter, vector_char, std::vector<char>{}, getRandomIntegerInputs<char>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructIterIter, vector_size_t, std::vector<size_t>{}, getRandomIntegerInputs<size_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructIterIter, vector_string, std::vector<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange, vector_char, std::vector<char>{}, getRandomIntegerInputs<char>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange, vector_size_t, std::vector<size_t>{}, getRandomIntegerInputs<size_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange, vector_string, std::vector<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_MAIN();
