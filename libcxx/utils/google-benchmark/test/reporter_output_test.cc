
#undef NDEBUG
#include <utility>

#include "benchmark/benchmark.h"
#include "output_test.h"

// ========================================================================= //
// ---------------------- Testing Prologue Output -------------------------- //
// ========================================================================= //

ADD_CASES(TC_ConsoleOut, {{"^[-]+$", MR_Next},
                          {"^Benchmark %s Time %s CPU %s Iterations$", MR_Next},
                          {"^[-]+$", MR_Next}});
static int AddContextCases() {
  AddCases(TC_ConsoleErr,
           {
               {"%int[-/]%int[-/]%int %int:%int:%int$", MR_Default},
               {"Running .*/reporter_output_test(\\.exe)?$", MR_Next},
               {"Run on \\(%int X %float MHz CPU s\\)", MR_Next},
           });
  AddCases(TC_JSONOut,
           {{"^\\{", MR_Default},
            {"\"context\":", MR_Next},
            {"\"date\": \"", MR_Next},
            {"\"executable\": \".*/reporter_output_test(\\.exe)?\",", MR_Next},
            {"\"num_cpus\": %int,$", MR_Next},
            {"\"mhz_per_cpu\": %float,$", MR_Next},
            {"\"cpu_scaling_enabled\": ", MR_Next},
            {"\"caches\": \\[$", MR_Next}});
  auto const& Caches = benchmark::CPUInfo::Get().caches;
  if (!Caches.empty()) {
    AddCases(TC_ConsoleErr, {{"CPU Caches:$", MR_Next}});
  }
  for (size_t I = 0; I < Caches.size(); ++I) {
    std::string num_caches_str =
        Caches[I].num_sharing != 0 ? " \\(x%int\\)$" : "$";
    AddCases(
        TC_ConsoleErr,
        {{"L%int (Data|Instruction|Unified) %intK" + num_caches_str, MR_Next}});
    AddCases(TC_JSONOut, {{"\\{$", MR_Next},
                          {"\"type\": \"", MR_Next},
                          {"\"level\": %int,$", MR_Next},
                          {"\"size\": %int,$", MR_Next},
                          {"\"num_sharing\": %int$", MR_Next},
                          {"}[,]{0,1}$", MR_Next}});
  }

  AddCases(TC_JSONOut, {{"],$"}});
  return 0;
}
int dummy_register = AddContextCases();
ADD_CASES(TC_CSVOut, {{"%csv_header"}});

// ========================================================================= //
// ------------------------ Testing Basic Output --------------------------- //
// ========================================================================= //

void BM_basic(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_basic);

ADD_CASES(TC_ConsoleOut, {{"^BM_basic %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_basic\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_basic\",%csv_report$"}});

// ========================================================================= //
// ------------------------ Testing Bytes per Second Output ---------------- //
// ========================================================================= //

void BM_bytes_per_second(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.SetBytesProcessed(1);
}
BENCHMARK(BM_bytes_per_second);

ADD_CASES(TC_ConsoleOut,
          {{"^BM_bytes_per_second %console_report +%float[kM]{0,1}B/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_bytes_per_second\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bytes_per_second\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_bytes_per_second\",%csv_bytes_report$"}});

// ========================================================================= //
// ------------------------ Testing Items per Second Output ---------------- //
// ========================================================================= //

void BM_items_per_second(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.SetItemsProcessed(1);
}
BENCHMARK(BM_items_per_second);

ADD_CASES(TC_ConsoleOut,
          {{"^BM_items_per_second %console_report +%float[kM]{0,1} items/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_items_per_second\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"items_per_second\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_items_per_second\",%csv_items_report$"}});

// ========================================================================= //
// ------------------------ Testing Label Output --------------------------- //
// ========================================================================= //

void BM_label(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.SetLabel("some label");
}
BENCHMARK(BM_label);

ADD_CASES(TC_ConsoleOut, {{"^BM_label %console_report some label$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_label\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"label\": \"some label\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_label\",%csv_label_report_begin\"some "
                       "label\"%csv_label_report_end$"}});

// ========================================================================= //
// ------------------------ Testing Error Output --------------------------- //
// ========================================================================= //

void BM_error(benchmark::State& state) {
  state.SkipWithError("message");
  for (auto _ : state) {
  }
}
BENCHMARK(BM_error);
ADD_CASES(TC_ConsoleOut, {{"^BM_error[ ]+ERROR OCCURRED: 'message'$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_error\",$"},
                       {"\"error_occurred\": true,$", MR_Next},
                       {"\"error_message\": \"message\",$", MR_Next}});

ADD_CASES(TC_CSVOut, {{"^\"BM_error\",,,,,,,,true,\"message\"$"}});

// ========================================================================= //
// ------------------------ Testing No Arg Name Output -----------------------
// //
// ========================================================================= //

void BM_no_arg_name(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_no_arg_name)->Arg(3);
ADD_CASES(TC_ConsoleOut, {{"^BM_no_arg_name/3 %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_no_arg_name/3\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_no_arg_name/3\",%csv_report$"}});

// ========================================================================= //
// ------------------------ Testing Arg Name Output ----------------------- //
// ========================================================================= //

void BM_arg_name(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_arg_name)->ArgName("first")->Arg(3);
ADD_CASES(TC_ConsoleOut, {{"^BM_arg_name/first:3 %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_arg_name/first:3\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_arg_name/first:3\",%csv_report$"}});

// ========================================================================= //
// ------------------------ Testing Arg Names Output ----------------------- //
// ========================================================================= //

void BM_arg_names(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_arg_names)->Args({2, 5, 4})->ArgNames({"first", "", "third"});
ADD_CASES(TC_ConsoleOut,
          {{"^BM_arg_names/first:2/5/third:4 %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_arg_names/first:2/5/third:4\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_arg_names/first:2/5/third:4\",%csv_report$"}});

// ========================================================================= //
// ----------------------- Testing Complexity Output ----------------------- //
// ========================================================================= //

void BM_Complexity_O1(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_Complexity_O1)->Range(1, 1 << 18)->Complexity(benchmark::o1);
SET_SUBSTITUTIONS({{"%bigOStr", "[ ]* %float \\([0-9]+\\)"},
                   {"%RMS", "[ ]*[0-9]+ %"}});
ADD_CASES(TC_ConsoleOut, {{"^BM_Complexity_O1_BigO %bigOStr %bigOStr[ ]*$"},
                          {"^BM_Complexity_O1_RMS %RMS %RMS[ ]*$"}});

// ========================================================================= //
// ----------------------- Testing Aggregate Output ------------------------ //
// ========================================================================= //

// Test that non-aggregate data is printed by default
void BM_Repeat(benchmark::State& state) {
  for (auto _ : state) {
  }
}
// need two repetitions min to be able to output any aggregate output
BENCHMARK(BM_Repeat)->Repetitions(2);
ADD_CASES(TC_ConsoleOut, {{"^BM_Repeat/repeats:2 %console_report$"},
                          {"^BM_Repeat/repeats:2 %console_report$"},
                          {"^BM_Repeat/repeats:2_mean %console_report$"},
                          {"^BM_Repeat/repeats:2_median %console_report$"},
                          {"^BM_Repeat/repeats:2_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Repeat/repeats:2\",$"},
                       {"\"name\": \"BM_Repeat/repeats:2\",$"},
                       {"\"name\": \"BM_Repeat/repeats:2_mean\",$"},
                       {"\"name\": \"BM_Repeat/repeats:2_median\",$"},
                       {"\"name\": \"BM_Repeat/repeats:2_stddev\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Repeat/repeats:2\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:2\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:2_mean\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:2_median\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:2_stddev\",%csv_report$"}});
// but for two repetitions, mean and median is the same, so let's repeat..
BENCHMARK(BM_Repeat)->Repetitions(3);
ADD_CASES(TC_ConsoleOut, {{"^BM_Repeat/repeats:3 %console_report$"},
                          {"^BM_Repeat/repeats:3 %console_report$"},
                          {"^BM_Repeat/repeats:3 %console_report$"},
                          {"^BM_Repeat/repeats:3_mean %console_report$"},
                          {"^BM_Repeat/repeats:3_median %console_report$"},
                          {"^BM_Repeat/repeats:3_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Repeat/repeats:3\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3_mean\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3_median\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3_stddev\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Repeat/repeats:3\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3_mean\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3_median\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3_stddev\",%csv_report$"}});
// median differs between even/odd number of repetitions, so just to be sure
BENCHMARK(BM_Repeat)->Repetitions(4);
ADD_CASES(TC_ConsoleOut, {{"^BM_Repeat/repeats:4 %console_report$"},
                          {"^BM_Repeat/repeats:4 %console_report$"},
                          {"^BM_Repeat/repeats:4 %console_report$"},
                          {"^BM_Repeat/repeats:4 %console_report$"},
                          {"^BM_Repeat/repeats:4_mean %console_report$"},
                          {"^BM_Repeat/repeats:4_median %console_report$"},
                          {"^BM_Repeat/repeats:4_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Repeat/repeats:4\",$"},
                       {"\"name\": \"BM_Repeat/repeats:4\",$"},
                       {"\"name\": \"BM_Repeat/repeats:4\",$"},
                       {"\"name\": \"BM_Repeat/repeats:4\",$"},
                       {"\"name\": \"BM_Repeat/repeats:4_mean\",$"},
                       {"\"name\": \"BM_Repeat/repeats:4_median\",$"},
                       {"\"name\": \"BM_Repeat/repeats:4_stddev\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Repeat/repeats:4\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:4\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:4\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:4\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:4_mean\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:4_median\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:4_stddev\",%csv_report$"}});

// Test that a non-repeated test still prints non-aggregate results even when
// only-aggregate reports have been requested
void BM_RepeatOnce(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_RepeatOnce)->Repetitions(1)->ReportAggregatesOnly();
ADD_CASES(TC_ConsoleOut, {{"^BM_RepeatOnce/repeats:1 %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_RepeatOnce/repeats:1\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_RepeatOnce/repeats:1\",%csv_report$"}});

// Test that non-aggregate data is not reported
void BM_SummaryRepeat(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_SummaryRepeat)->Repetitions(3)->ReportAggregatesOnly();
ADD_CASES(TC_ConsoleOut,
          {{".*BM_SummaryRepeat/repeats:3 ", MR_Not},
           {"^BM_SummaryRepeat/repeats:3_mean %console_report$"},
           {"^BM_SummaryRepeat/repeats:3_median %console_report$"},
           {"^BM_SummaryRepeat/repeats:3_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{".*BM_SummaryRepeat/repeats:3 ", MR_Not},
                       {"\"name\": \"BM_SummaryRepeat/repeats:3_mean\",$"},
                       {"\"name\": \"BM_SummaryRepeat/repeats:3_median\",$"},
                       {"\"name\": \"BM_SummaryRepeat/repeats:3_stddev\",$"}});
ADD_CASES(TC_CSVOut, {{".*BM_SummaryRepeat/repeats:3 ", MR_Not},
                      {"^\"BM_SummaryRepeat/repeats:3_mean\",%csv_report$"},
                      {"^\"BM_SummaryRepeat/repeats:3_median\",%csv_report$"},
                      {"^\"BM_SummaryRepeat/repeats:3_stddev\",%csv_report$"}});

void BM_RepeatTimeUnit(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_RepeatTimeUnit)
    ->Repetitions(3)
    ->ReportAggregatesOnly()
    ->Unit(benchmark::kMicrosecond);
ADD_CASES(TC_ConsoleOut,
          {{".*BM_RepeatTimeUnit/repeats:3 ", MR_Not},
           {"^BM_RepeatTimeUnit/repeats:3_mean %console_us_report$"},
           {"^BM_RepeatTimeUnit/repeats:3_median %console_us_report$"},
           {"^BM_RepeatTimeUnit/repeats:3_stddev %console_us_report$"}});
ADD_CASES(TC_JSONOut, {{".*BM_RepeatTimeUnit/repeats:3 ", MR_Not},
                       {"\"name\": \"BM_RepeatTimeUnit/repeats:3_mean\",$"},
                       {"\"time_unit\": \"us\",?$"},
                       {"\"name\": \"BM_RepeatTimeUnit/repeats:3_median\",$"},
                       {"\"time_unit\": \"us\",?$"},
                       {"\"name\": \"BM_RepeatTimeUnit/repeats:3_stddev\",$"},
                       {"\"time_unit\": \"us\",?$"}});
ADD_CASES(TC_CSVOut,
          {{".*BM_RepeatTimeUnit/repeats:3 ", MR_Not},
           {"^\"BM_RepeatTimeUnit/repeats:3_mean\",%csv_us_report$"},
           {"^\"BM_RepeatTimeUnit/repeats:3_median\",%csv_us_report$"},
           {"^\"BM_RepeatTimeUnit/repeats:3_stddev\",%csv_us_report$"}});

// ========================================================================= //
// -------------------- Testing user-provided statistics ------------------- //
// ========================================================================= //

const auto UserStatistics = [](const std::vector<double>& v) {
  return v.back();
};
void BM_UserStats(benchmark::State& state) {
  for (auto _ : state) {
  }
}
// clang-format off
BENCHMARK(BM_UserStats)
  ->Repetitions(3)
  ->ComputeStatistics("", UserStatistics);
// clang-format on

// check that user-provided stats is calculated, and is after the default-ones
// empty string as name is intentional, it would sort before anything else
ADD_CASES(TC_ConsoleOut, {{"^BM_UserStats/repeats:3 %console_report$"},
                          {"^BM_UserStats/repeats:3 %console_report$"},
                          {"^BM_UserStats/repeats:3 %console_report$"},
                          {"^BM_UserStats/repeats:3_mean %console_report$"},
                          {"^BM_UserStats/repeats:3_median %console_report$"},
                          {"^BM_UserStats/repeats:3_stddev %console_report$"},
                          {"^BM_UserStats/repeats:3_ %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_UserStats/repeats:3\",$"},
                       {"\"name\": \"BM_UserStats/repeats:3\",$"},
                       {"\"name\": \"BM_UserStats/repeats:3\",$"},
                       {"\"name\": \"BM_UserStats/repeats:3_mean\",$"},
                       {"\"name\": \"BM_UserStats/repeats:3_median\",$"},
                       {"\"name\": \"BM_UserStats/repeats:3_stddev\",$"},
                       {"\"name\": \"BM_UserStats/repeats:3_\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_UserStats/repeats:3\",%csv_report$"},
                      {"^\"BM_UserStats/repeats:3\",%csv_report$"},
                      {"^\"BM_UserStats/repeats:3\",%csv_report$"},
                      {"^\"BM_UserStats/repeats:3_mean\",%csv_report$"},
                      {"^\"BM_UserStats/repeats:3_median\",%csv_report$"},
                      {"^\"BM_UserStats/repeats:3_stddev\",%csv_report$"},
                      {"^\"BM_UserStats/repeats:3_\",%csv_report$"}});

// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char* argv[]) { RunOutputTests(argc, argv); }
