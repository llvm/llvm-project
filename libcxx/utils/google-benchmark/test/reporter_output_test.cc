
#undef NDEBUG
#include "benchmark/benchmark.h"
#include "../src/check.h" // NOTE: check.h is for internal use only!
#include "../src/re.h" // NOTE: re.h is for internal use only
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>

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

    std::string line;
    while (remaining_output.eof() == false) {
        CHECK(remaining_output.good());
        std::getline(remaining_output, line);
        if (r.Match(line)) return;
        CHECK(match_rule != MR_Next) << "Expected line \"" << line
                                     << "\" to match regex \"" << regex << "\"";
    }

    CHECK(remaining_output.eof() == false)
        << "End of output reached before match for regex \"" << regex
        << "\" was found";
  }
};

std::vector<TestCase> ConsoleOutputTests;
std::vector<TestCase> JSONOutputTests;
std::vector<TestCase> CSVOutputTests;

std::vector<TestCase> ConsoleErrorTests;
std::vector<TestCase> JSONErrorTests;
std::vector<TestCase> CSVErrorTests;

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

}  // end namespace

// ========================================================================= //
// ---------------------- Testing Prologue Output -------------------------- //
// ========================================================================= //

ADD_CASES(&ConsoleOutputTests, {
    {join("^Benchmark", "Time", "CPU", "Iterations$"), MR_Next},
    {"^[-]+$", MR_Next}
});
ADD_CASES(&CSVOutputTests, {
  {"name,iterations,real_time,cpu_time,time_unit,bytes_per_second,items_per_second,"
    "label,error_occurred,error_message"}
});

// ========================================================================= //
// ------------------------ Testing Basic Output --------------------------- //
// ========================================================================= //

void BM_basic(benchmark::State& state) {
  while (state.KeepRunning()) {}
}
BENCHMARK(BM_basic);

ADD_CASES(&ConsoleOutputTests, {
    {"^BM_basic[ ]+[0-9]{1,5} ns[ ]+[0-9]{1,5} ns[ ]+[0-9]+$"}
});
ADD_CASES(&JSONOutputTests, {
    {"\"name\": \"BM_basic\",$"},
    {"\"iterations\": [0-9]+,$", MR_Next},
    {"\"real_time\": [0-9]{1,5},$", MR_Next},
    {"\"cpu_time\": [0-9]{1,5},$", MR_Next},
    {"\"time_unit\": \"ns\"$", MR_Next},
    {"}", MR_Next}
});
ADD_CASES(&CSVOutputTests, {
    {"^\"BM_basic\",[0-9]+," + dec_re + "," + dec_re + ",ns,,,,,$"}
});

// ========================================================================= //
// ------------------------ Testing Error Output --------------------------- //
// ========================================================================= //

void BM_error(benchmark::State& state) {
    state.SkipWithError("message");
    while(state.KeepRunning()) {}
}
BENCHMARK(BM_error);
ADD_CASES(&ConsoleOutputTests, {
    {"^BM_error[ ]+ERROR OCCURRED: 'message'$"}
});
ADD_CASES(&JSONOutputTests, {
    {"\"name\": \"BM_error\",$"},
    {"\"error_occurred\": true,$", MR_Next},
    {"\"error_message\": \"message\",$", MR_Next}
});

ADD_CASES(&CSVOutputTests, {
    {"^\"BM_error\",,,,,,,,true,\"message\"$"}
});


// ========================================================================= //
// ----------------------- Testing Complexity Output ----------------------- //
// ========================================================================= //

void BM_Complexity_O1(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O1)->Range(1, 1<<18)->Complexity(benchmark::o1);

std::string bigOStr = "[0-9]+\\.[0-9]+ \\([0-9]+\\)";

ADD_CASES(&ConsoleOutputTests, {
   {join("^BM_Complexity_O1_BigO", bigOStr, bigOStr) + "[ ]*$"},
   {join("^BM_Complexity_O1_RMS", "[0-9]+ %", "[0-9]+ %") + "[ ]*$"}
});


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
    std::vector<TestCase>& error_cases;
    benchmark::BenchmarkReporter& reporter;
    std::stringstream out_stream;
    std::stringstream err_stream;

    ReporterTest(const char* n,
                 std::vector<TestCase>& out_tc,
                 std::vector<TestCase>& err_tc,
                 benchmark::BenchmarkReporter& br)
        : name(n), output_cases(out_tc), error_cases(err_tc), reporter(br) {
        reporter.SetOutputStream(&out_stream);
        reporter.SetErrorStream(&err_stream);
    }
  } TestCases[] = {
      {"ConsoleReporter", ConsoleOutputTests, ConsoleErrorTests, CR},
      {"JSONReporter", JSONOutputTests, JSONErrorTests, JR},
      {"CSVReporter", CSVOutputTests, CSVErrorTests, CSVR}
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

      for (const auto& TC : rep_test.error_cases)
        TC.Check(rep_test.err_stream);
      for (const auto& TC : rep_test.output_cases)
        TC.Check(rep_test.out_stream);

      std::cout << "\n";
  }
  return 0;
}
