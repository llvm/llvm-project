//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test fails because Clang no longer enables -fdelayed-template-parsing
// by default on Windows with C++20 (#69431).
// XFAIL: msvc && (clang-18 || clang-19 || clang-20)

// UNSUPPORTED: c++03, c++11

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "fuzz.h"

template <class IntT>
std::vector<IntT> GetValues(const std::uint8_t *data, std::size_t size) {
  std::vector<IntT> result;
  while (size >= sizeof(IntT)) {
    IntT tmp;
    std::memcpy(&tmp, data, sizeof(IntT));
    size -= sizeof(IntT);
    data += sizeof(IntT);
    result.push_back(tmp);
  }
  return result;
}

template <class Dist>
struct ParamTypeHelper {
  using ParamT = typename Dist::param_type;
  using ResultT = typename Dist::result_type;
  static_assert(std::is_same<ResultT, typename ParamT::distribution_type::result_type>::value, "");

  static ParamT Create(const std::uint8_t* data, std::size_t size, bool &OK) {
    constexpr bool select_vector_result = std::is_constructible<ParamT, ResultT*, ResultT*, ResultT*>::value;
    constexpr bool select_vector_double = std::is_constructible<ParamT, double*, double*>::value;
    constexpr int selector = select_vector_result ? 0 : (select_vector_double ? 1 : 2);
    return DispatchAndCreate(std::integral_constant<int, selector>{}, data, size, OK);
  }

  // Vector result
  static ParamT DispatchAndCreate(std::integral_constant<int, 0>, const std::uint8_t *data, std::size_t size, bool &OK) {
    auto Input = GetValues<ResultT>(data, size);
    OK = false;
    if (Input.size() < 10)
      return ParamT{};
    OK = true;
    auto Beg = Input.begin();
    auto End = Input.end();
    auto Mid = Beg + ((End - Beg) / 2);

    assert(Mid - Beg <= (End  -  Mid));
    ParamT p(Beg, Mid, Mid);
    return p;
  }

  // Vector double
  static ParamT DispatchAndCreate(std::integral_constant<int, 1>, const std::uint8_t *data, std::size_t size, bool &OK) {
    auto Input = GetValues<double>(data, size);

    OK = true;
    auto Beg = Input.begin();
    auto End = Input.end();

    ParamT p(Beg, End);
    return p;
  }

  // Default
  static ParamT DispatchAndCreate(std::integral_constant<int, 2>, const std::uint8_t *data, std::size_t size, bool &OK) {
    OK = false;
    if (size < sizeof(ParamT))
      return ParamT{};
    OK = true;
    ParamT input;
    std::memcpy(&input, data, sizeof(ParamT));
    return input;
  }
};

template <class IntT>
struct ParamTypeHelper<std::poisson_distribution<IntT>> {
  using Dist = std::poisson_distribution<IntT>;
  using ParamT = typename Dist::param_type;
  using ResultT = typename Dist::result_type;

  static ParamT Create(const std::uint8_t *data, std::size_t size, bool& OK) {
    OK = false;
    auto vals = GetValues<double>(data, size);
    if (vals.empty() || std::isnan(vals[0]) || std::isnan(std::abs(vals[0])) || vals[0] < 0)
      return ParamT{};
    OK = true;
    return ParamT{vals[0]};
  }
};

template <class IntT>
struct ParamTypeHelper<std::geometric_distribution<IntT>> {
  using Dist = std::geometric_distribution<IntT>;
  using ParamT = typename Dist::param_type;
  using ResultT = typename Dist::result_type;

  static ParamT Create(const std::uint8_t *data, std::size_t size, bool& OK) {
    OK = false;
    auto vals = GetValues<double>(data, size);
    if (vals.empty() || std::isnan(vals[0]) || vals[0] < 0 )
      return ParamT{};
    OK = true;
    return ParamT{vals[0]};
  }
};

template <class IntT>
struct ParamTypeHelper<std::lognormal_distribution<IntT>> {
  using Dist = std::lognormal_distribution<IntT>;
  using ParamT = typename Dist::param_type;
  using ResultT = typename Dist::result_type;

  static ParamT Create(const std::uint8_t *data, std::size_t size, bool& OK) {
    OK = false;
    auto vals = GetValues<ResultT>(data, size);
    if (vals.size() < 2 )
      return ParamT{};
    OK = true;
    return ParamT{vals[0], vals[1]};
  }
};

template <>
struct ParamTypeHelper<std::bernoulli_distribution> {
  using Dist = std::bernoulli_distribution;
  using ParamT = Dist::param_type;
  using ResultT = Dist::result_type;

  static ParamT Create(const std::uint8_t *data, std::size_t size, bool& OK) {
    OK = false;
    auto vals = GetValues<double>(data, size);
    if (vals.empty())
      return ParamT{};
    OK = true;
    return ParamT{vals[0]};
  }
};

template <class Distribution>
int helper(const std::uint8_t *data, std::size_t size) {
  std::mt19937 engine;
  using ParamT = typename Distribution::param_type;
  bool OK;
  ParamT p = ParamTypeHelper<Distribution>::Create(data, size, OK);
  if (!OK)
    return 0;
  Distribution d(p);
  volatile auto res = d(engine);
  if (std::isnan(res)) {
    // FIXME(llvm.org/PR44289):
    // Investigate why these distributions are returning NaN and decide
    // if that's what we want them to be doing.
    //
    // Make this assert false (or return non-zero).
    return 0;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
  return helper<std::uniform_int_distribution<std::int16_t>>(data, size)       ||
         helper<std::uniform_real_distribution<float>>(data, size)             ||
         helper<std::bernoulli_distribution>(data, size)                       ||
         helper<std::poisson_distribution<std::int16_t>>(data, size)           ||
         helper<std::geometric_distribution<std::int16_t>>(data, size)         ||
         helper<std::binomial_distribution<std::int16_t>>(data, size)          ||
         helper<std::negative_binomial_distribution<std::int16_t>>(data, size) ||
         helper<std::exponential_distribution<float>>(data, size)              ||
         helper<std::gamma_distribution<float>>(data, size)                    ||
         helper<std::weibull_distribution<float>>(data, size)                  ||
         helper<std::extreme_value_distribution<float>>(data, size)            ||
         helper<std::normal_distribution<float>>(data, size)                   ||
         helper<std::lognormal_distribution<float>>(data, size)                ||
         helper<std::chi_squared_distribution<float>>(data, size)              ||
         helper<std::cauchy_distribution<float>>(data, size)                   ||
         helper<std::fisher_f_distribution<float>>(data, size)                 ||
         helper<std::student_t_distribution<float>>(data, size)                ||
         helper<std::discrete_distribution<std::int16_t>>(data, size)          ||
         helper<std::piecewise_constant_distribution<float>>(data, size)       ||
         helper<std::piecewise_linear_distribution<float>>(data, size);
}
