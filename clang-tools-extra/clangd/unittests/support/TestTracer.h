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

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_SUPPORT_TESTTRACER_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_SUPPORT_TESTTRACER_H

#include "support/Trace.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace trace {

/// A RAII Tracer that can be used by tests.
class TestTracer : public EventTracer {
public:
  TestTracer() : S(*this) {}
  /// Stores all the measurements to be returned with take later on.
  void record(const Metric &Metric, double Value,
              llvm::StringRef Label) override;

  /// Returns recorded measurements for \p Metric and clears them.
  std::vector<double> takeMetric(llvm::StringRef Metric,
                                 llvm::StringRef Label = "");

private:
  std::mutex Mu;
  /// Measurements recorded per metric per label.
  llvm::StringMap<llvm::StringMap<std::vector<double>>> Measurements;
  Session S;
};

} // namespace trace
} // namespace clangd
} // namespace clang
#endif
