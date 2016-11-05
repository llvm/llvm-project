
#undef NDEBUG
#include <cassert>
#include <vector>

#include "../src/check.h"  // NOTE: check.h is for internal use only!
#include "benchmark/benchmark.h"

namespace {

class TestReporter : public benchmark::ConsoleReporter {
 public:
  virtual void ReportRuns(const std::vector<Run>& report) {
    all_runs_.insert(all_runs_.end(), begin(report), end(report));
    ConsoleReporter::ReportRuns(report);
  }

  std::vector<Run> all_runs_;
};

struct TestCase {
  std::string name;
  const char* label;
  // Note: not explicit as we rely on it being converted through ADD_CASES.
  TestCase(const char* xname) : TestCase(xname, nullptr) {}
  TestCase(const char* xname, const char* xlabel)
      : name(xname), label(xlabel) {}

  typedef benchmark::BenchmarkReporter::Run Run;

  void CheckRun(Run const& run) const {
    CHECK(name == run.benchmark_name) << "expected " << name << " got "
                                      << run.benchmark_name;
    if (label) {
      CHECK(run.report_label == label) << "expected " << label << " got "
                                       << run.report_label;
    } else {
      CHECK(run.report_label == "");
    }
  }
};

std::vector<TestCase> ExpectedResults;

int AddCases(std::initializer_list<TestCase> const& v) {
  for (auto N : v) {
    ExpectedResults.push_back(N);
  }
  return 0;
}

#define CONCAT(x, y) CONCAT2(x, y)
#define CONCAT2(x, y) x##y
#define ADD_CASES(...) int CONCAT(dummy, __LINE__) = AddCases({__VA_ARGS__})

}  // end namespace

typedef benchmark::internal::Benchmark* ReturnVal;

//----------------------------------------------------------------------------//
// Test RegisterBenchmark with no additional arguments
//----------------------------------------------------------------------------//
void BM_function(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_function);
ReturnVal dummy = benchmark::RegisterBenchmark(
    "BM_function_manual_registration", BM_function);
ADD_CASES({"BM_function"}, {"BM_function_manual_registration"});

//----------------------------------------------------------------------------//
// Test RegisterBenchmark with additional arguments
// Note: GCC <= 4.8 do not support this form of RegisterBenchmark because they
//       reject the variadic pack expansion of lambda captures.
//----------------------------------------------------------------------------//
#ifndef BENCHMARK_HAS_NO_VARIADIC_REGISTER_BENCHMARK

void BM_extra_args(benchmark::State& st, const char* label) {
  while (st.KeepRunning()) {
  }
  st.SetLabel(label);
}
int RegisterFromFunction() {
  std::pair<const char*, const char*> cases[] = {
      {"test1", "One"}, {"test2", "Two"}, {"test3", "Three"}};
  for (auto const& c : cases)
    benchmark::RegisterBenchmark(c.first, &BM_extra_args, c.second);
  return 0;
}
int dummy2 = RegisterFromFunction();
ADD_CASES({"test1", "One"}, {"test2", "Two"}, {"test3", "Three"});

#endif  // BENCHMARK_HAS_NO_VARIADIC_REGISTER_BENCHMARK

//----------------------------------------------------------------------------//
// Test RegisterBenchmark with different callable types
//----------------------------------------------------------------------------//

struct CustomFixture {
  void operator()(benchmark::State& st) {
    while (st.KeepRunning()) {
    }
  }
};

void TestRegistrationAtRuntime() {
#ifdef BENCHMARK_HAS_CXX11
  {
    CustomFixture fx;
    benchmark::RegisterBenchmark("custom_fixture", fx);
    AddCases({"custom_fixture"});
  }
#endif
#ifndef BENCHMARK_HAS_NO_VARIADIC_REGISTER_BENCHMARK
  {
    int x = 42;
    auto capturing_lam = [=](benchmark::State& st) {
      while (st.KeepRunning()) {
      }
      st.SetLabel(std::to_string(x));
    };
    benchmark::RegisterBenchmark("lambda_benchmark", capturing_lam);
    AddCases({{"lambda_benchmark", "42"}});
  }
#endif
}

int main(int argc, char* argv[]) {
  TestRegistrationAtRuntime();

  benchmark::Initialize(&argc, argv);

  TestReporter test_reporter;
  benchmark::RunSpecifiedBenchmarks(&test_reporter);

  typedef benchmark::BenchmarkReporter::Run Run;
  auto EB = ExpectedResults.begin();

  for (Run const& run : test_reporter.all_runs_) {
    assert(EB != ExpectedResults.end());
    EB->CheckRun(run);
    ++EB;
  }
  assert(EB == ExpectedResults.end());

  return 0;
}
