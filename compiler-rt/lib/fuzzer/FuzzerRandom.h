//===- FuzzerRandom.h - Internal header for the Fuzzer ----------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// fuzzer::Random
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_RANDOM_H
#define LLVM_FUZZER_RANDOM_H

#include <random>

namespace fuzzer {
class Random : public std::minstd_rand {
 public:
  Random(unsigned int seed) : std::minstd_rand(seed) {}
  result_type operator()() { return this->std::minstd_rand::operator()(); }
  size_t Rand() { return this->operator()(); }
  size_t RandBool() { return Rand() % 2; }
  size_t operator()(size_t n) { return n ? Rand() % n : 0; }
  intptr_t operator()(intptr_t From, intptr_t To) {
    assert(From < To);
    intptr_t RangeSize = To - From + 1;
    return operator()(RangeSize) + From;
  }
};

}  // namespace fuzzer

#endif  // LLVM_FUZZER_RANDOM_H
