// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark/reporter.h"
#include "complexity.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "check.h"
#include "colorprint.h"
#include "commandlineflags.h"
#include "internal_macros.h"
#include "string_util.h"
#include "walltime.h"

namespace benchmark {

bool ConsoleReporter::ReportContext(const Context& context) {
  name_field_width_ = context.name_field_width;

  PrintBasicContext(&GetErrorStream(), context);

#ifdef BENCHMARK_OS_WINDOWS
  if (color_output_ && &std::cout != &GetOutputStream()) {
      GetErrorStream() << "Color printing is only supported for stdout on windows."
                          " Disabling color printing\n";
      color_output_ = false;
  }
#endif
  std::string str = FormatString("%-*s %13s %13s %10s\n",
                             static_cast<int>(name_field_width_), "Benchmark",
                             "Time", "CPU", "Iterations");
  GetOutputStream() << str << std::string(str.length() - 1, '-') << "\n";

  return true;
}

void ConsoleReporter::ReportRuns(const std::vector<Run>& reports) {
  for (const auto& run : reports)
    PrintRunData(run);
}

static void  IgnoreColorPrint(std::ostream& out, LogColor,
                               const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    out << FormatString(fmt, args);
    va_end(args);
}

void ConsoleReporter::PrintRunData(const Run& result) {
  typedef void(PrinterFn)(std::ostream&, LogColor, const char*, ...);
  auto& Out = GetOutputStream();
  PrinterFn* printer = color_output_ ? (PrinterFn*)ColorPrintf
                                     : IgnoreColorPrint;
  auto name_color =
      (result.report_big_o || result.report_rms) ? COLOR_BLUE : COLOR_GREEN;
  printer(Out, name_color, "%-*s ", name_field_width_,
              result.benchmark_name.c_str());

  if (result.error_occurred) {
    printer(Out, COLOR_RED, "ERROR OCCURRED: \'%s\'",
                result.error_message.c_str());
    printer(Out, COLOR_DEFAULT, "\n");
    return;
  }
  // Format bytes per second
  std::string rate;
  if (result.bytes_per_second > 0) {
    rate = StrCat(" ", HumanReadableNumber(result.bytes_per_second), "B/s");
  }

  // Format items per second
  std::string items;
  if (result.items_per_second > 0) {
    items = StrCat(" ", HumanReadableNumber(result.items_per_second),
                   " items/s");
 }

  const double real_time = result.GetAdjustedRealTime();
  const double cpu_time = result.GetAdjustedCPUTime();

  if (result.report_big_o) {
    std::string big_o = GetBigOString(result.complexity);
    printer(Out, COLOR_YELLOW, "%10.2f %s %10.2f %s ", real_time,
                big_o.c_str(), cpu_time, big_o.c_str());
  } else if (result.report_rms) {
    printer(Out, COLOR_YELLOW, "%10.0f %% %10.0f %% ", real_time * 100,
                cpu_time * 100);
  } else {
    const char* timeLabel = GetTimeUnitString(result.time_unit);
    printer(Out, COLOR_YELLOW, "%10.0f %s %10.0f %s ", real_time, timeLabel,
                cpu_time, timeLabel);
  }

  if (!result.report_big_o && !result.report_rms) {
    printer(Out, COLOR_CYAN, "%10lld", result.iterations);
  }

  if (!rate.empty()) {
    printer(Out, COLOR_DEFAULT, " %*s", 13, rate.c_str());
  }

  if (!items.empty()) {
    printer(Out, COLOR_DEFAULT, " %*s", 18, items.c_str());
  }

  if (!result.report_label.empty()) {
    printer(Out, COLOR_DEFAULT, " %s", result.report_label.c_str());
  }

  printer(Out, COLOR_DEFAULT, "\n");
}

}  // end namespace benchmark
