//===-- TestTracer.cpp - Tracing unit tests ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TestTracer.h"
#include "support/Trace.h"
#include "llvm/ADT/StringRef.h"
#include <mutex>

namespace clang {
namespace clangd {
namespace trace {

void TestTracer::record(const Metric &Metric, double Value,
                        llvm::StringRef Label) {
  std::lock_guard<std::mutex> Lock(Mu);
  Measurements[Metric.Name][Label].push_back(Value);
}

std::vector<double> TestTracer::takeMetric(llvm::StringRef Metric,
                                           llvm::StringRef Label) {
  std::lock_guard<std::mutex> Lock(Mu);
  auto LabelsIt = Measurements.find(Metric);
  if (LabelsIt == Measurements.end())
    return {};
  auto &Labels = LabelsIt->getValue();
  auto ValuesIt = Labels.find(Label);
  if (ValuesIt == Labels.end())
    return {};
  auto Res = std::move(ValuesIt->getValue());
  ValuesIt->getValue().clear();
  return Res;
}
} // namespace trace
} // namespace clangd
} // namespace clang
