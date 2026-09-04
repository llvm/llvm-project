//===-- Worst case test template for math functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_EXHAUSTIVE_WORST_CASE_TEST_H
#define LLVM_LIBC_TEST_SRC_MATH_EXHAUSTIVE_WORST_CASE_TEST_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename OutType, typename InType = OutType>
using UnaryOp = OutType(InType);

template <typename OutType, typename InType = OutType>
using BinaryOp = OutType(InType, InType);

namespace detail {

LIBC_INLINE std::string find_file(const std::string &filename) {
  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  if (file_exists(filename))
    return filename;

  if (const char *env = std::getenv("LIBC_WORST_CASE_DIR")) {
    std::string path = std::string(env) + "/" + filename;
    if (file_exists(path))
      return path;
  }

#ifdef LIBC_MATH_EXHAUSTIVE_TEST_DIR
  {
    std::string path =
        std::string(LIBC_MATH_EXHAUSTIVE_TEST_DIR) + "/" + filename;
    if (file_exists(path))
      return path;
  }
#endif

  const char *candidate_dirs[] = {
      "libc/test/src/math/exhaustive",
      "../libc/test/src/math/exhaustive",
      "../../libc/test/src/math/exhaustive",
      "../../../libc/test/src/math/exhaustive",
      "test/src/math/exhaustive",
      "../test/src/math/exhaustive",
  };
  for (const char *dir : candidate_dirs) {
    std::string path = std::string(dir) + "/" + filename;
    if (file_exists(path))
      return path;
  }

  return "";
}

template <typename FloatType>
bool load_binary_file(const std::string &path, std::vector<FloatType> &inputs) {
  std::ifstream file(path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open())
    return false;
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size <= 0 || (size % sizeof(FloatType) != 0))
    return false;
  size_t count = static_cast<size_t>(size) / sizeof(FloatType);
  inputs.resize(count);
  if (!file.read(reinterpret_cast<char *>(inputs.data()), size))
    return false;
  return true;
}

template <typename FloatType>
bool load_wc_stream(std::istream &in, std::vector<FloatType> &inputs) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  std::string line;
  while (std::getline(in, line)) {
    size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos)
      line.erase(comment_pos);

    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
      continue;
    size_t end = line.find_last_not_of(" \t\r\n");
    std::string token = line.substr(start, end - start + 1);

    size_t first_delim = token.find_first_of(" \t,");
    if (first_delim != std::string::npos)
      token = token.substr(0, first_delim);

    if (token == "snan" || token == "+snan") {
      inputs.push_back(FPBits(uint64_t(0x7ff4000000000000ULL)).get_val());
    } else if (token == "-snan") {
      inputs.push_back(FPBits(uint64_t(0xfff4000000000000ULL)).get_val());
    } else if (token == "qnan" || token == "+qnan" || token == "nan" ||
               token == "+nan") {
      inputs.push_back(FPBits::quiet_nan().get_val());
    } else if (token == "-qnan" || token == "-nan") {
      inputs.push_back(FPBits::quiet_nan(LIBC_NAMESPACE::Sign::NEG).get_val());
    } else if (token == "inf" || token == "+inf") {
      inputs.push_back(FPBits::inf().get_val());
    } else if (token == "-inf") {
      inputs.push_back(FPBits::inf(LIBC_NAMESPACE::Sign::NEG).get_val());
    } else if (token == "+0" || token == "0") {
      inputs.push_back(FloatType(0.0));
    } else if (token == "-0") {
      inputs.push_back(FloatType(-0.0));
    } else {
      char *endptr = nullptr;
      double val = std::strtod(token.c_str(), &endptr);
      if (endptr != token.c_str()) {
        inputs.push_back(static_cast<FloatType>(val));
      }
    }
  }
  return !inputs.empty();
}

template <typename FloatType>
LIBC_INLINE bool load_input_file(const std::string &name,
                                 std::vector<FloatType> &inputs) {
  // 1. Try finding <name>.bin
  std::string bin_name = name;
  if (bin_name.size() < 4 || bin_name.substr(bin_name.size() - 4) != ".bin") {
    if (bin_name.size() >= 3 && bin_name.substr(bin_name.size() - 3) == ".wc")
      bin_name = bin_name.substr(0, bin_name.size() - 3);
    bin_name += ".bin";
  }
  std::string bin_path = find_file(bin_name);
  if (!bin_path.empty()) {
    if (load_binary_file(bin_path, inputs)) {
      std::cout << "Loaded " << inputs.size() << " test cases from binary file "
                << bin_path << std::endl;
      return true;
    }
  }

  // 2. Try finding <name>.wc
  std::string wc_name = name;
  if (wc_name.size() < 3 || wc_name.substr(wc_name.size() - 3) != ".wc") {
    if (wc_name.size() >= 4 && wc_name.substr(wc_name.size() - 4) == ".bin")
      wc_name = wc_name.substr(0, wc_name.size() - 4);
    wc_name += ".wc";
  }
  std::string wc_path = find_file(wc_name);
  if (!wc_path.empty()) {
    std::ifstream file(wc_path.c_str());
    if (file.is_open() && load_wc_stream(file, inputs)) {
      std::cout << "Loaded " << inputs.size() << " test cases from text file "
                << wc_path << std::endl;
      return true;
    }
  }

  std::cerr << "Error: Could not locate worst-case file for '" << name << "'\n";
  return false;
}

} // namespace detail

template <typename OutType, typename InType, mpfr::Operation Op,
          UnaryOp<OutType, InType> Func, unsigned Tolerance = 0>
struct UnaryOpWorstCaseChecker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = InType;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using StorageType = typename FPBits::StorageType;

  uint64_t check(const FloatType *inputs, size_t count,
                 mpfr::RoundingMode rounding, bool test_symmetric) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return count;

    uint64_t failed = 0;
    for (size_t i = 0; i < count; ++i) {
      FloatType x = inputs[i];
      bool correct = TEST_MPFR_MATCH_ROUNDING_SILENTLY(
          Op, x, Func(x), static_cast<double>(Tolerance) + 0.5, rounding);
      failed += (!correct);
      if (!correct) {
        EXPECT_MPFR_MATCH_ROUNDING(Op, x, Func(x), 0.5, rounding);
      }
      if (test_symmetric) {
        FloatType neg_x = -x;
        bool correct_neg = TEST_MPFR_MATCH_ROUNDING_SILENTLY(
            Op, neg_x, Func(neg_x), static_cast<double>(Tolerance) + 0.5,
            rounding);
        failed += (!correct_neg);
        if (!correct_neg) {
          EXPECT_MPFR_MATCH_ROUNDING(Op, neg_x, Func(neg_x), 0.5, rounding);
        }
      }
    }
    return failed;
  }
};

template <typename Checker, size_t ChunkSize = 10000>
struct LlvmLibcWorstCaseMathTest : public virtual LIBC_NAMESPACE::testing::Test,
                                   public Checker {
  using FloatType = typename Checker::FloatType;

  void test_inputs(mpfr::RoundingMode rounding,
                   const std::vector<FloatType> &inputs,
                   bool test_symmetric = false) {
    const size_t total = inputs.size();
    if (total == 0) {
      std::cout << "No inputs to test.\n";
      return;
    }

    int n_threads = std::thread::hardware_concurrency();
#ifdef LIBC_TEST_MAX_CONCURRENCY
    if (n_threads <= 0 || n_threads > LIBC_TEST_MAX_CONCURRENCY)
      n_threads = LIBC_TEST_MAX_CONCURRENCY;
#endif
    if (n_threads < 1)
      n_threads = 1;

    std::vector<std::thread> thread_list;
    std::mutex mx_cur_val;
    std::mutex mx_print;
    size_t current_idx = 0;
    std::atomic<uint64_t> failed(0);
    int current_percent = -1;

    for (int i = 0; i < n_threads; ++i) {
      thread_list.emplace_back([&, this]() {
        while (true) {
          size_t begin_idx = 0;
          size_t end_idx = 0;
          int new_percent = -1;
          {
            std::lock_guard<std::mutex> lock(mx_cur_val);
            if (current_idx >= total)
              return;
            begin_idx = current_idx;
            current_idx = std::min(current_idx + ChunkSize, total);
            end_idx = current_idx;
            int pc = static_cast<int>((100 * end_idx) / total);
            if (current_percent != pc) {
              new_percent = pc;
              current_percent = pc;
            }
          }

          if (new_percent >= 0) {
            std::lock_guard<std::mutex> lock(mx_print);
            std::cout << new_percent << "% is in process     \r" << std::flush;
          }

          uint64_t failed_in_chunk =
              Checker::check(&inputs[begin_idx], end_idx - begin_idx, rounding,
                             test_symmetric);
          if (failed_in_chunk > 0) {
            failed.fetch_add(failed_in_chunk);
          }
        }
      });
    }

    for (auto &thread : thread_list) {
      if (thread.joinable())
        thread.join();
    }

    std::cout << std::endl;
    std::cout << "Test " << ((failed > 0) ? "FAILED" : "PASSED") << " ("
              << failed.load() << " failures out of "
              << (total * (test_symmetric ? 2 : 1)) << " tests)" << std::endl;
    ASSERT_EQ(failed.load(), uint64_t(0));
  }

  void test_inputs_all_roundings(const std::vector<FloatType> &inputs,
                                 bool test_symmetric = false) {
    std::cout << "-- Testing for FE_TONEAREST (" << inputs.size()
              << " inputs) --" << std::endl;
    test_inputs(mpfr::RoundingMode::Nearest, inputs, test_symmetric);

    std::cout << "-- Testing for FE_UPWARD (" << inputs.size() << " inputs) --"
              << std::endl;
    test_inputs(mpfr::RoundingMode::Upward, inputs, test_symmetric);

    std::cout << "-- Testing for FE_DOWNWARD (" << inputs.size()
              << " inputs) --" << std::endl;
    test_inputs(mpfr::RoundingMode::Downward, inputs, test_symmetric);

    std::cout << "-- Testing for FE_TOWARDZERO (" << inputs.size()
              << " inputs) --" << std::endl;
    test_inputs(mpfr::RoundingMode::TowardZero, inputs, test_symmetric);
  }

  void test_file(const std::string &filename_or_func,
                 mpfr::RoundingMode rounding, bool test_symmetric = false) {
    std::vector<FloatType> inputs;
    ASSERT_TRUE(detail::load_input_file(filename_or_func, inputs));
    test_inputs(rounding, inputs, test_symmetric);
  }

  void test_file_all_roundings(const std::string &filename_or_func,
                               bool test_symmetric = false) {
    std::vector<FloatType> inputs;
    ASSERT_TRUE(detail::load_input_file(filename_or_func, inputs));
    test_inputs_all_roundings(inputs, test_symmetric);
  }
};

template <typename FloatType, mpfr::Operation Op, UnaryOp<FloatType> Func,
          unsigned Tolerance = 0>
using LlvmLibcUnaryOpWorstCaseMathTest = LlvmLibcWorstCaseMathTest<
    UnaryOpWorstCaseChecker<FloatType, FloatType, Op, Func, Tolerance>>;

#endif // LLVM_LIBC_TEST_SRC_MATH_EXHAUSTIVE_WORST_CASE_TEST_H
