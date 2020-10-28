//===-- Benchmark memcpy implementation -----------------------------------===//
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
#include <memory>

namespace __llvm_libc {
extern void *memcpy(void *__restrict, const void *__restrict, size_t);
} // namespace __llvm_libc

namespace llvm {
namespace libc_benchmarks {

// The context encapsulates the buffers, parameters and the measure.
struct MemcpyContext : public BenchmarkRunner {
  using FunctionPrototype = void *(*)(void *, const void *, size_t);

  struct ParameterType {
    uint16_t SrcOffset = 0;
    uint16_t DstOffset = 0;
  };

  explicit MemcpyContext(const StudyConfiguration &Conf)
      : OD(Conf), SrcBuffer(Conf.BufferSize), DstBuffer(Conf.BufferSize),
        PP(*this) {}

  // Needed by the ParameterProvider to update the current batch of parameter.
  void Randomize(MutableArrayRef<ParameterType> Parameters) {
    for (auto &P : Parameters) {
      P.DstOffset = OD(Gen);
      P.SrcOffset = OD(Gen);
    }
  }

  ArrayRef<StringRef> getFunctionNames() const override {
    static std::array<StringRef, 1> kFunctionNames = {"memcpy"};
    return kFunctionNames;
  }

  BenchmarkResult benchmark(const BenchmarkOptions &Options,
                            StringRef FunctionName, size_t Size) override {
    FunctionPrototype Function = StringSwitch<FunctionPrototype>(FunctionName)
                                     .Case("memcpy", &__llvm_libc::memcpy);
    return llvm::libc_benchmarks::benchmark(
        Options, PP, [this, Function, Size](ParameterType p) {
          Function(DstBuffer + p.DstOffset, SrcBuffer + p.SrcOffset, Size);
          return DstBuffer + p.DstOffset;
        });
  }

private:
  std::default_random_engine Gen;
  OffsetDistribution OD;
  AlignedBuffer SrcBuffer;
  AlignedBuffer DstBuffer;
  SmallParameterProvider<MemcpyContext> PP;
};

std::unique_ptr<BenchmarkRunner> getRunner(const StudyConfiguration &Conf) {
  return std::make_unique<MemcpyContext>(Conf);
}

} // namespace libc_benchmarks
} // namespace llvm
