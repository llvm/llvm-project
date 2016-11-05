
#undef NDEBUG
#include <utility>

#include "benchmark/benchmark.h"
#include "output_test.h"

// ========================================================================= //
// ---------------------- Testing Prologue Output -------------------------- //
// ========================================================================= //

ADD_CASES(TC_ConsoleOut, {{"^Benchmark %s Time %s CPU %s Iterations$", MR_Next},
                          {"^[-]+$", MR_Next}});
ADD_CASES(TC_CSVOut,
          {{"name,iterations,real_time,cpu_time,time_unit,bytes_per_second,"
            "items_per_second,label,error_occurred,error_message"}});

// ========================================================================= //
// ------------------------ Testing Basic Output --------------------------- //
// ========================================================================= //

void BM_basic(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_basic);

ADD_CASES(TC_ConsoleOut, {{"^BM_basic %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_basic\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %int,$", MR_Next},
                       {"\"cpu_time\": %int,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_basic\",%csv_report$"}});

// ========================================================================= //
// ------------------------ Testing Bytes per Second Output ---------------- //
// ========================================================================= //

void BM_bytes_per_second(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
  state.SetBytesProcessed(1);
}
BENCHMARK(BM_bytes_per_second);

ADD_CASES(TC_ConsoleOut,
          {{"^BM_bytes_per_second %console_report +%floatB/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_bytes_per_second\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %int,$", MR_Next},
                       {"\"cpu_time\": %int,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bytes_per_second\": %int$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_bytes_per_second\",%csv_bytes_report$"}});

// ========================================================================= //
// ------------------------ Testing Items per Second Output ---------------- //
// ========================================================================= //

void BM_items_per_second(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
  state.SetItemsProcessed(1);
}
BENCHMARK(BM_items_per_second);

ADD_CASES(TC_ConsoleOut,
          {{"^BM_items_per_second %console_report +%float items/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_items_per_second\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %int,$", MR_Next},
                       {"\"cpu_time\": %int,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"items_per_second\": %int$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_items_per_second\",%csv_items_report$"}});

// ========================================================================= //
// ------------------------ Testing Label Output --------------------------- //
// ========================================================================= //

void BM_label(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
  state.SetLabel("some label");
}
BENCHMARK(BM_label);

ADD_CASES(TC_ConsoleOut, {{"^BM_label %console_report some label$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_label\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %int,$", MR_Next},
                       {"\"cpu_time\": %int,$", MR_Next},
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
  while (state.KeepRunning()) {
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
  while (state.KeepRunning()) {
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
  while (state.KeepRunning()) {
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
  while (state.KeepRunning()) {
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
  while (state.KeepRunning()) {
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
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_Repeat)->Repetitions(3);
ADD_CASES(TC_ConsoleOut, {{"^BM_Repeat/repeats:3 %console_report$"},
                          {"^BM_Repeat/repeats:3 %console_report$"},
                          {"^BM_Repeat/repeats:3 %console_report$"},
                          {"^BM_Repeat/repeats:3_mean %console_report$"},
                          {"^BM_Repeat/repeats:3_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Repeat/repeats:3\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3_mean\",$"},
                       {"\"name\": \"BM_Repeat/repeats:3_stddev\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Repeat/repeats:3\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3_mean\",%csv_report$"},
                      {"^\"BM_Repeat/repeats:3_stddev\",%csv_report$"}});

// Test that a non-repeated test still prints non-aggregate results even when
// only-aggregate reports have been requested
void BM_RepeatOnce(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_RepeatOnce)->Repetitions(1)->ReportAggregatesOnly();
ADD_CASES(TC_ConsoleOut, {{"^BM_RepeatOnce/repeats:1 %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_RepeatOnce/repeats:1\",$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_RepeatOnce/repeats:1\",%csv_report$"}});

// Test that non-aggregate data is not reported
void BM_SummaryRepeat(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_SummaryRepeat)->Repetitions(3)->ReportAggregatesOnly();
ADD_CASES(TC_ConsoleOut,
          {{".*BM_SummaryRepeat/repeats:3 ", MR_Not},
           {"^BM_SummaryRepeat/repeats:3_mean %console_report$"},
           {"^BM_SummaryRepeat/repeats:3_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{".*BM_SummaryRepeat/repeats:3 ", MR_Not},
                       {"\"name\": \"BM_SummaryRepeat/repeats:3_mean\",$"},
                       {"\"name\": \"BM_SummaryRepeat/repeats:3_stddev\",$"}});
ADD_CASES(TC_CSVOut, {{".*BM_SummaryRepeat/repeats:3 ", MR_Not},
                      {"^\"BM_SummaryRepeat/repeats:3_mean\",%csv_report$"},
                      {"^\"BM_SummaryRepeat/repeats:3_stddev\",%csv_report$"}});

void BM_RepeatTimeUnit(benchmark::State& state) {
  while (state.KeepRunning()) {
  }
}
BENCHMARK(BM_RepeatTimeUnit)
    ->Repetitions(3)
    ->ReportAggregatesOnly()
    ->Unit(benchmark::kMicrosecond);
ADD_CASES(TC_ConsoleOut,
          {{".*BM_RepeatTimeUnit/repeats:3 ", MR_Not},
           {"^BM_RepeatTimeUnit/repeats:3_mean %console_us_report$"},
           {"^BM_RepeatTimeUnit/repeats:3_stddev %console_us_report$"}});
ADD_CASES(TC_JSONOut, {{".*BM_RepeatTimeUnit/repeats:3 ", MR_Not},
                       {"\"name\": \"BM_RepeatTimeUnit/repeats:3_mean\",$"},
                       {"\"time_unit\": \"us\",?$"},
                       {"\"name\": \"BM_RepeatTimeUnit/repeats:3_stddev\",$"},
                       {"\"time_unit\": \"us\",?$"}});
ADD_CASES(TC_CSVOut,
          {{".*BM_RepeatTimeUnit/repeats:3 ", MR_Not},
           {"^\"BM_RepeatTimeUnit/repeats:3_mean\",%csv_us_report$"},
           {"^\"BM_RepeatTimeUnit/repeats:3_stddev\",%csv_us_report$"}});

// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char* argv[]) { RunOutputTests(argc, argv); }
