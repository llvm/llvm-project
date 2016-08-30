#undef NDEBUG
#include "benchmark/benchmark.h"
#include "output_test.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace {

#define ADD_COMPLEXITY_CASES(...) \
    int CONCAT(dummy, __LINE__) = AddComplexityTest(__VA_ARGS__)

int AddComplexityTest(std::string big_o_test_name,
                      std::string rms_test_name, std::string big_o) {
  SetSubstitutions({
        {"%bigo_name", big_o_test_name},
        {"%rms_name", rms_test_name},
        {"%bigo_str", "[ ]*" + std::string(dec_re) + " " + big_o},
        {"%bigo", big_o},
        {"%rms", "[ ]*[0-9]+ %"}
  });
  AddCases(TC_ConsoleOut, {
    {"^%bigo_name %bigo_str %bigo_str[ ]*$"},
    {"^%bigo_name", MR_Not}, // Assert we we didn't only matched a name.
    {"^%rms_name %rms %rms[ ]*$", MR_Next}
  });
  AddCases(TC_JSONOut, {
    {"\"name\": \"%bigo_name\",$"},
    {"\"cpu_coefficient\": [0-9]+,$", MR_Next},
    {"\"real_coefficient\": [0-9]{1,5},$", MR_Next},
    {"\"big_o\": \"%bigo\",$", MR_Next},
    {"\"time_unit\": \"ns\"$", MR_Next},
    {"}", MR_Next},
    {"\"name\": \"%rms_name\",$"},
    {"\"rms\": [0-9]+%$", MR_Next},
    {"}", MR_Next}
  });
  AddCases(TC_CSVOut, {
    {"^\"%bigo_name\",,%float,%float,%bigo,,,,,$"},
    {"^\"%bigo_name\"", MR_Not},
    {"^\"%rms_name\",,%float,%float,,,,,,$", MR_Next}
  });
  return 0;
}

}  // end namespace

// ========================================================================= //
// --------------------------- Testing BigO O(1) --------------------------- //
// ========================================================================= //

void BM_Complexity_O1(benchmark::State& state) {
  while (state.KeepRunning()) {
      for (int i=0; i < 1024; ++i) {
          benchmark::DoNotOptimize(&i);
      }
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O1) -> Range(1, 1<<18) -> Complexity(benchmark::o1);
BENCHMARK(BM_Complexity_O1) -> Range(1, 1<<18) -> Complexity();
BENCHMARK(BM_Complexity_O1) -> Range(1, 1<<18) -> Complexity([](int){return 1.0; });

const char* big_o_1_test_name = "BM_Complexity_O1_BigO";
const char* rms_o_1_test_name = "BM_Complexity_O1_RMS";
const char* enum_big_o_1 = "\\([0-9]+\\)";
// FIXME: Tolerate both '(1)' and 'lgN' as output when the complexity is auto deduced.
// See https://github.com/google/benchmark/issues/272
const char* auto_big_o_1 = "(\\([0-9]+\\))|(lgN)";
const char* lambda_big_o_1 = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(big_o_1_test_name, rms_o_1_test_name, enum_big_o_1);

// Add auto enum tests
ADD_COMPLEXITY_CASES(big_o_1_test_name, rms_o_1_test_name, auto_big_o_1);

// Add lambda tests
ADD_COMPLEXITY_CASES(big_o_1_test_name, rms_o_1_test_name, lambda_big_o_1);

// ========================================================================= //
// --------------------------- Testing BigO O(N) --------------------------- //
// ========================================================================= //

std::vector<int> ConstructRandomVector(int size) {
  std::vector<int> v;
  v.reserve(size);
  for (int i = 0; i < size; ++i) {
    v.push_back(std::rand() % size);
  }
  return v;
}

void BM_Complexity_O_N(benchmark::State& state) {
  auto v = ConstructRandomVector(state.range(0));
  const int item_not_in_vector = state.range(0)*2; // Test worst case scenario (item not in vector)
  while (state.KeepRunning()) {
      benchmark::DoNotOptimize(std::find(v.begin(), v.end(), item_not_in_vector));
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O_N) -> RangeMultiplier(2) -> Range(1<<10, 1<<16) -> Complexity(benchmark::oN);
BENCHMARK(BM_Complexity_O_N) -> RangeMultiplier(2) -> Range(1<<10, 1<<16) -> Complexity([](int n) -> double{return n; });
BENCHMARK(BM_Complexity_O_N) -> RangeMultiplier(2) -> Range(1<<10, 1<<16) -> Complexity();

const char* big_o_n_test_name = "BM_Complexity_O_N_BigO";
const char* rms_o_n_test_name = "BM_Complexity_O_N_RMS";
const char* enum_auto_big_o_n = "N";
const char* lambda_big_o_n = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(big_o_n_test_name, rms_o_n_test_name, enum_auto_big_o_n);

// Add lambda tests
ADD_COMPLEXITY_CASES(big_o_n_test_name, rms_o_n_test_name, lambda_big_o_n);

// ========================================================================= //
// ------------------------- Testing BigO O(N*lgN) ------------------------- //
// ========================================================================= //

static void BM_Complexity_O_N_log_N(benchmark::State& state) {
  auto v = ConstructRandomVector(state.range(0));
  while (state.KeepRunning()) {
      std::sort(v.begin(), v.end());
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O_N_log_N) -> RangeMultiplier(2) -> Range(1<<10, 1<<16) -> Complexity(benchmark::oNLogN);
BENCHMARK(BM_Complexity_O_N_log_N) -> RangeMultiplier(2) -> Range(1<<10, 1<<16) -> Complexity([](int n) {return n * std::log2(n); });
BENCHMARK(BM_Complexity_O_N_log_N) -> RangeMultiplier(2) -> Range(1<<10, 1<<16) -> Complexity();

const char* big_o_n_lg_n_test_name = "BM_Complexity_O_N_log_N_BigO";
const char* rms_o_n_lg_n_test_name = "BM_Complexity_O_N_log_N_RMS";
const char* enum_auto_big_o_n_lg_n = "NlgN";
const char* lambda_big_o_n_lg_n = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(big_o_n_lg_n_test_name, rms_o_n_lg_n_test_name, enum_auto_big_o_n_lg_n);

// Add lambda tests
ADD_COMPLEXITY_CASES(big_o_n_lg_n_test_name, rms_o_n_lg_n_test_name, lambda_big_o_n_lg_n);


// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char* argv[]) {
  RunOutputTests(argc, argv);
}

