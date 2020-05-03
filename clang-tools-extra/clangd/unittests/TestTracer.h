//===-- TestTracer.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allows setting up a fake tracer for tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTFS_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTFS_H

#include "support/Trace.h"
#include "llvm/ADT/StringMap.h"
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace trace {

/// A RAII Tracer that can be used by tests.
class TestTracer : public EventTracer {
public:
  TestTracer() : Session(*this) {}
  /// Stores all the measurements to be returned with take later on.
  void record(const Metric &Metric, double Value,
              llvm::StringRef Label) override;

  /// Returns recorded measurements for \p Metric and clears them.
  std::vector<double> take(std::string Metric, std::string Label = "");

private:
  struct Measure {
    std::string Label;
    double Value;
  };
  /// Measurements recorded per metric.
  llvm::StringMap<std::vector<Measure>> Measurements;
  Session Session;
};

} // namespace trace
} // namespace clangd
} // namespace clang
#endif
