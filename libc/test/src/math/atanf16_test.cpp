#include "src/math/atanf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAtanf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7c00U;

// Range: [-Inf, 0]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xfc00U;

TEST_F(LlvmLibcAtanf16Test, PositiveRange) {
  for (uint16_t v = POS_START; v <= POS_STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   LIBC_NAMESPACE::atanf16(x), 0.5);
  }
}

TEST_F(LlvmLibcAtanf16Test, NegativeRange) {
  for (uint16_t v = NEG_START; v <= NEG_STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Atan, x,
                                   LIBC_NAMESPACE::atanf16(x), 0.5);
  }
}
