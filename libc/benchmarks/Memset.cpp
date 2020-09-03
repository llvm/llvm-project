//===-- Benchmark memset implementation -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcBenchmark.h"
#include "LibcMemoryBenchmark.h"
#include "LibcMemoryBenchmarkMain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace __llvm_libc {
void *memset(void *, int, size_t);
} // namespace __llvm_libc

namespace llvm {
namespace libc_benchmarks {

// The context encapsulates the buffers, parameters and the measure.
struct MemsetContext : public BenchmarkRunner {
  using FunctionPrototype = void *(*)(void *, int, size_t);

  struct ParameterType {
    uint16_t DstOffset = 0;
  };

  explicit MemsetContext(const StudyConfiguration &Conf)
      : OD(Conf), DstBuffer(Conf.BufferSize), MemsetValue(Conf.MemsetValue),
        PP(*this) {}

  // Needed by the ParameterProvider to update the current batch of parameter.
  void Randomize(MutableArrayRef<ParameterType> Parameters) {
    for (auto &P : Parameters) {
      P.DstOffset = OD(Gen);
    }
  }

  ArrayRef<StringRef> getFunctionNames() const override {
    static std::array<StringRef, 1> kFunctionNames = {"memset"};
    return kFunctionNames;
  }

  BenchmarkResult benchmark(const BenchmarkOptions &Options,
                            StringRef FunctionName, size_t Size) override {
    FunctionPrototype Function = StringSwitch<FunctionPrototype>(FunctionName)
                                     .Case("memset", &__llvm_libc::memset);
    return llvm::libc_benchmarks::benchmark(
        Options, PP, [this, Function, Size](ParameterType p) {
          Function(DstBuffer + p.DstOffset, MemsetValue, Size);
          return DstBuffer + p.DstOffset;
        });
  }

private:
  std::default_random_engine Gen;
  OffsetDistribution OD;
  AlignedBuffer DstBuffer;
  const uint8_t MemsetValue;
  SmallParameterProvider<MemsetContext> PP;
};

std::unique_ptr<BenchmarkRunner> getRunner(const StudyConfiguration &Conf) {
  return std::make_unique<MemsetContext>(Conf);
}

} // namespace libc_benchmarks
} // namespace llvm
