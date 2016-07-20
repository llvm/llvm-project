
#undef NDEBUG
#include "benchmark/benchmark.h"
#include "../src/check.h" // NOTE: check.h is for internal use only!
#include <cassert>
#include <vector>

namespace {

class TestReporter : public benchmark::ConsoleReporter {
 public:
  virtual bool ReportContext(const Context& context) {
    return ConsoleReporter::ReportContext(context);
  };

  virtual void ReportRuns(const std::vector<Run>& report) {
    all_runs_.insert(all_runs_.end(), begin(report), end(report));
    ConsoleReporter::ReportRuns(report);
  }

  TestReporter()  {}
  virtual ~TestReporter() {}

  mutable std::vector<Run> all_runs_;
};

struct TestCase {
  std::string name;
  bool error_occurred;
  std::string error_message;

  typedef benchmark::BenchmarkReporter::Run Run;

  void CheckRun(Run const& run) const {
    CHECK(name == run.benchmark_name) << "expected " << name << " got " << run.benchmark_name;
    CHECK(error_occurred == run.error_occurred);
    CHECK(error_message == run.error_message);
    if (error_occurred) {
      //CHECK(run.iterations == 0);
    } else {
      CHECK(run.iterations != 0);
    }
  }
};

std::vector<TestCase> ExpectedResults;

int AddCases(const char* base_name, std::initializer_list<TestCase> const& v) {
  for (auto TC : v) {
    TC.name = base_name + TC.name;
    ExpectedResults.push_back(std::move(TC));
  }
  return 0;
}

#define CONCAT(x, y) CONCAT2(x, y)
#define CONCAT2(x, y) x##y
#define ADD_CASES(...) \
int CONCAT(dummy, __LINE__) = AddCases(__VA_ARGS__)

}  // end namespace


void BM_error_before_running(benchmark::State& state) {
  state.SkipWithError("error message");
  while (state.KeepRunning()) {
    assert(false);
  }
}
BENCHMARK(BM_error_before_running);
ADD_CASES("BM_error_before_running",
          {{"", true, "error message"}});

void BM_error_during_running(benchmark::State& state) {
  int first_iter = true;
  while (state.KeepRunning()) {
    if (state.range_x() == 1 && state.thread_index <= (state.threads / 2)) {
      assert(first_iter);
      first_iter = false;
      state.SkipWithError("error message");
    } else {
      state.PauseTiming();
      state.ResumeTiming();
    }
  }
}
BENCHMARK(BM_error_during_running)->Arg(1)->Arg(2)->ThreadRange(1, 8);
ADD_CASES(
    "BM_error_during_running",
    {{"/1/threads:1", true, "error message"},
    {"/1/threads:2", true, "error message"},
    {"/1/threads:4", true, "error message"},
    {"/1/threads:8", true, "error message"},
    {"/2/threads:1", false, ""},
    {"/2/threads:2", false, ""},
    {"/2/threads:4", false, ""},
    {"/2/threads:8", false, ""}}
);

void BM_error_after_running(benchmark::State& state) {
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(state.iterations());
  }
  if (state.thread_index <= (state.threads / 2))
    state.SkipWithError("error message");
}
BENCHMARK(BM_error_after_running)->ThreadRange(1, 8);
ADD_CASES(
    "BM_error_after_running",
    {{"/threads:1", true, "error message"},
    {"/threads:2", true, "error message"},
    {"/threads:4", true, "error message"},
    {"/threads:8", true, "error message"}}
);

void BM_error_while_paused(benchmark::State& state) {
  bool first_iter = true;
  while (state.KeepRunning()) {
    if (state.range_x() == 1 && state.thread_index <= (state.threads / 2)) {
      assert(first_iter);
      first_iter = false;
      state.PauseTiming();
      state.SkipWithError("error message");
    } else {
      state.PauseTiming();
      state.ResumeTiming();
    }
  }
}
BENCHMARK(BM_error_while_paused)->Arg(1)->Arg(2)->ThreadRange(1, 8);
ADD_CASES(
    "BM_error_while_paused",
    {{"/1/threads:1", true, "error message"},
    {"/1/threads:2", true, "error message"},
    {"/1/threads:4", true, "error message"},
    {"/1/threads:8", true, "error message"},
    {"/2/threads:1", false, ""},
    {"/2/threads:2", false, ""},
    {"/2/threads:4", false, ""},
    {"/2/threads:8", false, ""}}
);


int main(int argc, char* argv[]) {
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
