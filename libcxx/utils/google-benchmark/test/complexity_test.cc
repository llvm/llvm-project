
#undef NDEBUG
#include "benchmark/benchmark.h"
#include "../src/check.h" // NOTE: check.h is for internal use only!
#include "../src/re.h"    // NOTE: re.h is for internal use only
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>

namespace {

// ========================================================================= //
// -------------------------- Testing Case --------------------------------- //
// ========================================================================= //

enum MatchRules {
  MR_Default, // Skip non-matching lines until a match is found.
  MR_Next    // Match must occur on the next line.
};

struct TestCase {
  std::string regex;
  int match_rule;

  TestCase(std::string re, int rule = MR_Default) : regex(re), match_rule(rule) {}

  void Check(std::stringstream& remaining_output) const {
    benchmark::Regex r;
    std::string err_str;
    r.Init(regex, &err_str);
    CHECK(err_str.empty()) << "Could not construct regex \"" << regex << "\""
                           << " got Error: " << err_str;

    std::string near = "<EOF>";
    std::string line;
    bool first = true;
    while (remaining_output.eof() == false) {
        CHECK(remaining_output.good());
        std::getline(remaining_output, line);
        // Keep the first line as context.
        if (first) {
            near = line;
            first = false;
        }
        if (r.Match(line)) return;
        CHECK(match_rule != MR_Next) << "Expected line \"" << line
                                     << "\" to match regex \"" << regex << "\""
                                     << "\nstarted matching at line: \"" << near << "\"";
    }

    CHECK(remaining_output.eof() == false)
        << "End of output reached before match for regex \"" << regex
        << "\" was found"
        << "\nstarted matching at line: \"" << near << "\"";
  }
};

std::vector<TestCase> ConsoleOutputTests;
std::vector<TestCase> JSONOutputTests;
std::vector<TestCase> CSVOutputTests;

// ========================================================================= //
// -------------------------- Test Helpers --------------------------------- //
// ========================================================================= //

class TestReporter : public benchmark::BenchmarkReporter {
public:
  TestReporter(std::vector<benchmark::BenchmarkReporter*> reps)
      : reporters_(reps)  {}

  virtual bool ReportContext(const Context& context) {
    bool last_ret = false;
    bool first = true;
    for (auto rep : reporters_) {
      bool new_ret = rep->ReportContext(context);
      CHECK(first || new_ret == last_ret)
          << "Reports return different values for ReportContext";
      first = false;
      last_ret = new_ret;
    }
    return last_ret;
  }

  virtual void ReportRuns(const std::vector<Run>& report) {
    for (auto rep : reporters_)
      rep->ReportRuns(report);
  }

  virtual void Finalize() {
      for (auto rep : reporters_)
        rep->Finalize();
  }

private:
  std::vector<benchmark::BenchmarkReporter*> reporters_;
};


#define CONCAT2(x, y) x##y
#define CONCAT(x, y) CONCAT2(x, y)

#define ADD_CASES(...) \
    int CONCAT(dummy, __LINE__) = AddCases(__VA_ARGS__)

int AddCases(std::vector<TestCase>* out, std::initializer_list<TestCase> const& v) {
  for (auto const& TC : v)
    out->push_back(TC);
  return 0;
}

template <class First>
std::string join(First f) { return f; }

template <class First, class ...Args>
std::string join(First f, Args&&... args) {
    return std::string(std::move(f)) + "[ ]+" + join(std::forward<Args>(args)...);
}

std::string dec_re = "[0-9]*[.]?[0-9]+([eE][-+][0-9]+)?";

#define ADD_COMPLEXITY_CASES(...) \
    int CONCAT(dummy, __LINE__) = AddComplexityTest(__VA_ARGS__)

int AddComplexityTest(std::vector<TestCase>* console_out, std::vector<TestCase>* json_out,
                      std::vector<TestCase>* csv_out, std::string big_o_test_name, 
                      std::string rms_test_name, std::string big_o) {
  std::string big_o_str = dec_re + " " + big_o;
  AddCases(console_out, {
    {join("^" + big_o_test_name + "", big_o_str, big_o_str) + "[ ]*$"},
    {join("^" + rms_test_name + "", "[0-9]+ %", "[0-9]+ %") + "[ ]*$"}
  });
  AddCases(json_out, {
    {"\"name\": \"" + big_o_test_name + "\",$"},
    {"\"cpu_coefficient\": [0-9]+,$", MR_Next},
    {"\"real_coefficient\": [0-9]{1,5},$", MR_Next},
    {"\"big_o\": \"" + big_o + "\",$", MR_Next},
    {"\"time_unit\": \"ns\"$", MR_Next},
    {"}", MR_Next},
    {"\"name\": \"" + rms_test_name + "\",$"},
    {"\"rms\": [0-9]+%$", MR_Next},
    {"}", MR_Next}
  });
  AddCases(csv_out, {
    {"^\"" + big_o_test_name + "\",," + dec_re + "," + dec_re + "," + big_o + ",,,,,$"},
    {"^\"" + rms_test_name + "\",," + dec_re + "," + dec_re + ",,,,,,$", MR_Next}
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
const char* enum_auto_big_o_1 = "\\([0-9]+\\)";
const char* lambda_big_o_1 = "f\\(N\\)";

// Add enum tests
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests, 
                     big_o_1_test_name, rms_o_1_test_name, enum_auto_big_o_1);

// Add auto enum tests
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests,
                     big_o_1_test_name, rms_o_1_test_name, enum_auto_big_o_1);

// Add lambda tests
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests, 
                     big_o_1_test_name, rms_o_1_test_name, lambda_big_o_1);

// ========================================================================= //
// --------------------------- Testing BigO O(N) --------------------------- //
// ========================================================================= //

std::vector<int> ConstructRandomVector(int size) {
  std::vector<int> v;
  v.reserve(size);
  for (int i = 0; i < size; ++i) {
    v.push_back(rand() % size);
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
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests, 
                     big_o_n_test_name, rms_o_n_test_name, enum_auto_big_o_n);

// Add lambda tests
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests, 
                     big_o_n_test_name, rms_o_n_test_name, lambda_big_o_n);

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
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests, 
                     big_o_n_lg_n_test_name, rms_o_n_lg_n_test_name, enum_auto_big_o_n_lg_n);

// Add lambda tests
ADD_COMPLEXITY_CASES(&ConsoleOutputTests, &JSONOutputTests, &CSVOutputTests, 
                     big_o_n_lg_n_test_name, rms_o_n_lg_n_test_name, lambda_big_o_n_lg_n);


// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //


int main(int argc, char* argv[]) {
  benchmark::Initialize(&argc, argv);
  benchmark::ConsoleReporter CR(benchmark::ConsoleReporter::OO_None);
  benchmark::JSONReporter JR;
  benchmark::CSVReporter CSVR;
  struct ReporterTest {
    const char* name;
    std::vector<TestCase>& output_cases;
    benchmark::BenchmarkReporter& reporter;
    std::stringstream out_stream;
    std::stringstream err_stream;

    ReporterTest(const char* n,
                 std::vector<TestCase>& out_tc,
                 benchmark::BenchmarkReporter& br)
        : name(n), output_cases(out_tc), reporter(br) {
        reporter.SetOutputStream(&out_stream);
        reporter.SetErrorStream(&err_stream);
    }
  } TestCases[] = {
      {"ConsoleReporter", ConsoleOutputTests, CR},
      {"JSONReporter", JSONOutputTests, JR},
      {"CSVReporter", CSVOutputTests, CSVR}
  };

  // Create the test reporter and run the benchmarks.
  std::cout << "Running benchmarks...\n";
  TestReporter test_rep({&CR, &JR, &CSVR});
  benchmark::RunSpecifiedBenchmarks(&test_rep);

  for (auto& rep_test : TestCases) {
      std::string msg = std::string("\nTesting ") + rep_test.name + " Output\n";
      std::string banner(msg.size() - 1, '-');
      std::cout << banner << msg << banner << "\n";

      std::cerr << rep_test.err_stream.str();
      std::cout << rep_test.out_stream.str();

      for (const auto& TC : rep_test.output_cases)
        TC.Check(rep_test.out_stream);

      std::cout << "\n";
  }
  return 0;
}

