#include "src/math/fabs.h"
#include "src/sys/random/getrandom.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcGetRandomTest, InvalidFlag) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  static constexpr size_t SIZE = 256;
  char data[SIZE];
  errno = 0;
  ASSERT_THAT(__llvm_libc::getrandom(data, SIZE, -1), Fails(EINVAL));
  errno = 0;
}

TEST(LlvmLibcGetRandomTest, InvalidBuffer) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

  errno = 0;
  ASSERT_THAT(__llvm_libc::getrandom(nullptr, 65536, 0), Fails(EFAULT));
  errno = 0;
}

TEST(LlvmLibcGetRandomTest, PiEstimation) {
  static constexpr size_t LIMIT = 10000000;
  static constexpr double PI = 3.14159265358979;

  auto generator = []() {
    uint16_t data;
    __llvm_libc::getrandom(&data, sizeof(data), 0);
    return data;
  };

  auto sample = [&]() {
    auto x = static_cast<double>(generator()) / 65536.0;
    auto y = static_cast<double>(generator()) / 65536.0;
    return x * x + y * y < 1.0;
  };

  double counter = 0;
  for (size_t i = 0; i < LIMIT; ++i) {
    if (sample()) {
      counter += 1.0;
    }
  }
  counter = counter / LIMIT * 4.0;
  ASSERT_TRUE(__llvm_libc::fabs(counter - PI) < 0.1);
}
