#include <cstdint>
#include <vector>

#include "profile/MemProfData.inc"
#include "gtest/gtest.h"

namespace llvm {
namespace memprof {
namespace {
TEST(MemProf, F16EncodeDecode) {
  const std::vector<uint64_t> TestCases = {
      0, 100, 4095, 4096, 5000, 8191, 65535, 1000000, 134213640, 200000000,
  };

  for (const uint64_t TestCase : TestCases) {
    const uint16_t Encoded = encodeHistogramCount(TestCase);
    const uint64_t Decoded = decodeHistogramCount(Encoded);

    const uint64_t MaxRepresentable = static_cast<uint64_t>(MaxMantissa)
                                      << MaxExponent;

    if (TestCase >= MaxRepresentable) {
      EXPECT_EQ(Decoded, MaxRepresentable);
    } else if (TestCase <= MaxMantissa) {
      EXPECT_EQ(Decoded, TestCase);
    } else {
      // The decoded value should be close to the original value.
      // The error should be less than 1/1024 for larger numbers.
      EXPECT_NEAR(Decoded, TestCase, static_cast<double>(TestCase) / 1024.0);
    }
  }
}
} // namespace
} // namespace memprof
} // namespace llvm
