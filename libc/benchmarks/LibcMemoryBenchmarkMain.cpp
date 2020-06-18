//===-- Benchmark ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcMemoryBenchmarkMain.h"
#include "JSON.h"
#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace llvm {
namespace libc_benchmarks {

static cl::opt<std::string>
    Configuration("conf", cl::desc("Specify configuration filename"),
                  cl::value_desc("filename"), cl::init(""));

static cl::opt<std::string> Output("o", cl::desc("Specify output filename"),
                                   cl::value_desc("filename"), cl::init("-"));

extern std::unique_ptr<BenchmarkRunner>
getRunner(const StudyConfiguration &Conf);

void Main() {
#ifndef NDEBUG
  static_assert(
      false,
      "For reproducibility benchmarks should not be compiled in DEBUG mode.");
#endif
  checkRequirements();
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Configuration);
  if (!MB)
    report_fatal_error(
        Twine("Could not open configuration file: ").concat(Configuration));
  auto ErrorOrStudy = ParseJsonStudy((*MB)->getBuffer());
  if (!ErrorOrStudy)
    report_fatal_error(ErrorOrStudy.takeError());

  const auto StudyPrototype = *ErrorOrStudy;

  Study S;
  S.Host = HostState::get();
  S.Options = StudyPrototype.Options;
  S.Configuration = StudyPrototype.Configuration;

  const auto Runs = S.Configuration.Runs;
  const auto &SR = S.Configuration.Size;
  std::unique_ptr<BenchmarkRunner> Runner = getRunner(S.Configuration);
  const size_t TotalSteps =
      Runner->getFunctionNames().size() * Runs * ((SR.To - SR.From) / SR.Step);
  size_t Steps = 0;
  for (auto FunctionName : Runner->getFunctionNames()) {
    FunctionMeasurements FM;
    FM.Name = std::string(FunctionName);
    for (size_t Run = 0; Run < Runs; ++Run) {
      for (uint32_t Size = SR.From; Size <= SR.To; Size += SR.Step) {
        const auto Result = Runner->benchmark(S.Options, FunctionName, Size);
        Measurement Measurement;
        Measurement.Runtime = Result.BestGuess;
        Measurement.Size = Size;
        FM.Measurements.push_back(Measurement);
        outs() << format("%3d%% run: %2d / %2d size: %5d ",
                         (Steps * 100 / TotalSteps), Run, Runs, Size)
               << FunctionName
               << "                                                  \r";
        ++Steps;
      }
    }
    S.Functions.push_back(std::move(FM));
  }

  std::error_code EC;
  raw_fd_ostream FOS(Output, EC);
  if (EC)
    report_fatal_error(Twine("Could not open file: ")
                           .concat(EC.message())
                           .concat(", ")
                           .concat(Output));
  json::OStream JOS(FOS);
  SerializeToJson(S, JOS);
}

} // namespace libc_benchmarks
} // namespace llvm

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::libc_benchmarks::Main();
  return EXIT_SUCCESS;
}
