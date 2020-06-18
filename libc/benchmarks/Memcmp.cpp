//===-- Benchmark memcmp implementation -----------------------------------===//
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

namespace llvm {
namespace libc_benchmarks {

// The context encapsulates the buffers, parameters and the measure.
struct MemcmpContext : public BenchmarkRunner {
  using FunctionPrototype = int (*)(const void *, const void *, size_t);

  struct ParameterType {
    uint16_t Offset = 0;
  };

  explicit MemcmpContext(const StudyConfiguration &Conf)
      : MOD(Conf), OD(Conf), ABuffer(Conf.BufferSize), BBuffer(Conf.BufferSize),
        PP(*this) {
    std::uniform_int_distribution<char> Dis;
    // Generate random buffer A.
    for (size_t I = 0; I < Conf.BufferSize; ++I)
      ABuffer[I] = Dis(Gen);
    // Copy buffer A to B.
    ::memcpy(BBuffer.begin(), ABuffer.begin(), Conf.BufferSize);
    if (Conf.MemcmpMismatchAt == 0)
      return; // all same.
    else if (Conf.MemcmpMismatchAt == 1)
      for (char &c : BBuffer)
        ++c; // all different.
    else
      for (const auto I : MOD.getMismatchIndices())
        ++BBuffer[I];
  }

  // Needed by the ParameterProvider to update the current batch of parameter.
  void Randomize(MutableArrayRef<ParameterType> Parameters) {
    if (MOD)
      for (auto &P : Parameters)
        P.Offset = MOD(Gen, CurrentSize);
    else
      for (auto &P : Parameters)
        P.Offset = OD(Gen);
  }

  ArrayRef<StringRef> getFunctionNames() const override {
    static std::array<StringRef, 1> kFunctionNames = {"memcmp"};
    return kFunctionNames;
  }

  BenchmarkResult benchmark(const BenchmarkOptions &Options,
                            StringRef FunctionName, size_t Size) override {
    CurrentSize = Size;
    // FIXME: Add `bcmp` once we're guaranteed that the function is provided.
    FunctionPrototype Function =
        StringSwitch<FunctionPrototype>(FunctionName).Case("memcmp", &::memcmp);
    return llvm::libc_benchmarks::benchmark(
        Options, PP, [this, Function, Size](ParameterType p) {
          return Function(ABuffer + p.Offset, BBuffer + p.Offset, Size);
        });
  }

private:
  std::default_random_engine Gen;
  MismatchOffsetDistribution MOD;
  OffsetDistribution OD;
  size_t CurrentSize = 0;
  AlignedBuffer ABuffer;
  AlignedBuffer BBuffer;
  SmallParameterProvider<MemcmpContext> PP;
};

std::unique_ptr<BenchmarkRunner> getRunner(const StudyConfiguration &Conf) {
  return std::make_unique<MemcmpContext>(Conf);
}

} // namespace libc_benchmarks
} // namespace llvm
