//===---------------------------------------------------------------------===//
// statistics_test - Unit tests for src/statistics.cc
//===---------------------------------------------------------------------===//

#include "../src/string_util.h"
#include "gtest/gtest.h"

namespace {
TEST(StringUtilTest, stoul) {
  {
    size_t pos = 0;
    EXPECT_EQ(0, benchmark::stoul("0", &pos));
    EXPECT_EQ(1, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(7, benchmark::stoul("7", &pos));
    EXPECT_EQ(1, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(135, benchmark::stoul("135", &pos));
    EXPECT_EQ(3, pos);
  }
#if ULONG_MAX == 0xFFFFFFFFul
  {
    size_t pos = 0;
    EXPECT_EQ(0xFFFFFFFFul, benchmark::stoul("4294967295", &pos));
    EXPECT_EQ(10, pos);
  }
#elif ULONG_MAX == 0xFFFFFFFFFFFFFFFFul
  {
    size_t pos = 0;
    EXPECT_EQ(0xFFFFFFFFFFFFFFFFul, benchmark::stoul("18446744073709551615", &pos));
    EXPECT_EQ(20, pos);
  }
#endif
  {
    size_t pos = 0;
    EXPECT_EQ(10, benchmark::stoul("1010", &pos, 2));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(520, benchmark::stoul("1010", &pos, 8));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(1010, benchmark::stoul("1010", &pos, 10));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(4112, benchmark::stoul("1010", &pos, 16));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(0xBEEF, benchmark::stoul("BEEF", &pos, 16));
    EXPECT_EQ(4, pos);
  }
  {
    ASSERT_THROW(benchmark::stoul("this is a test"), std::invalid_argument);
  }
}

TEST(StringUtilTest, stoi) {
  {
    size_t pos = 0;
    EXPECT_EQ(0, benchmark::stoi("0", &pos));
    EXPECT_EQ(1, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(-17, benchmark::stoi("-17", &pos));
    EXPECT_EQ(3, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(1357, benchmark::stoi("1357", &pos));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(10, benchmark::stoi("1010", &pos, 2));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(520, benchmark::stoi("1010", &pos, 8));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(1010, benchmark::stoi("1010", &pos, 10));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(4112, benchmark::stoi("1010", &pos, 16));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(0xBEEF, benchmark::stoi("BEEF", &pos, 16));
    EXPECT_EQ(4, pos);
  }
  {
    ASSERT_THROW(benchmark::stoi("this is a test"), std::invalid_argument);
  }
}

TEST(StringUtilTest, stod) {
  {
    size_t pos = 0;
    EXPECT_EQ(0.0, benchmark::stod("0", &pos));
    EXPECT_EQ(1, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(-84.0, benchmark::stod("-84", &pos));
    EXPECT_EQ(3, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(1234.0, benchmark::stod("1234", &pos));
    EXPECT_EQ(4, pos);
  }
  {
    size_t pos = 0;
    EXPECT_EQ(1.5, benchmark::stod("1.5", &pos));
    EXPECT_EQ(3, pos);
  }
  {
    size_t pos = 0;
    /* Note: exactly representable as double */
    EXPECT_EQ(-1.25e+9, benchmark::stod("-1.25e+9", &pos));
    EXPECT_EQ(8, pos);
  }
  {
    ASSERT_THROW(benchmark::stod("this is a test"), std::invalid_argument);
  }
}

}  // end namespace
