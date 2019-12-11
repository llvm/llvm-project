// -*- C++ -*-
//===-------------------------- fuzzing.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_FUZZING
#define _LIBCPP_FUZZING

#include <cstddef> // for size_t
#include <cstdint> // for uint8_t

namespace fuzzing {

//  These all return 0 on success; != 0 on failure
    int sort             (const uint8_t *data, size_t size);
    int stable_sort      (const uint8_t *data, size_t size);
    int partition        (const uint8_t *data, size_t size);
    int partition_copy   (const uint8_t *data, size_t size);
    int stable_partition (const uint8_t *data, size_t size);
	int unique           (const uint8_t *data, size_t size);
	int unique_copy      (const uint8_t *data, size_t size);

//  partition and stable_partition take Bi-Di iterators.
//  Should test those, too
    int nth_element       (const uint8_t *data, size_t size);
    int partial_sort      (const uint8_t *data, size_t size);
    int partial_sort_copy (const uint8_t *data, size_t size);

//  Heap operations
    int make_heap        (const uint8_t *data, size_t size);
    int push_heap        (const uint8_t *data, size_t size);
    int pop_heap         (const uint8_t *data, size_t size);

//  Various flavors of regex
    int regex_ECMAScript (const uint8_t *data, size_t size);
    int regex_POSIX      (const uint8_t *data, size_t size);
    int regex_extended   (const uint8_t *data, size_t size);
    int regex_awk        (const uint8_t *data, size_t size);
    int regex_grep       (const uint8_t *data, size_t size);
    int regex_egrep      (const uint8_t *data, size_t size);

//	Searching
	int search                      (const uint8_t *data, size_t size);
// 	int search_boyer_moore          (const uint8_t *data, size_t size);
// 	int search_boyer_moore_horspool (const uint8_t *data, size_t size);

//	Set operations
// 	int includes                 (const uint8_t *data, size_t size);
// 	int set_union                (const uint8_t *data, size_t size);
// 	int set_intersection         (const uint8_t *data, size_t size);
// 	int set_difference           (const uint8_t *data, size_t size);
// 	int set_symmetric_difference (const uint8_t *data, size_t size);
// 	int merge                    (const uint8_t *data, size_t size);

// Random numbers
  int uniform_int_distribution(const uint8_t*, size_t);
  int uniform_real_distribution(const uint8_t*, size_t);
  int bernoulli_distribution(const uint8_t*, size_t);
  int poisson_distribution(const uint8_t*, size_t);
  int geometric_distribution(const uint8_t*, size_t);
  int binomial_distribution(const uint8_t*, size_t);
  int negative_binomial_distribution(const uint8_t*, size_t);
  int exponential_distribution(const uint8_t*, size_t);
  int gamma_distribution(const uint8_t*, size_t);
  int weibull_distribution(const uint8_t*, size_t);
  int extreme_value_distribution(const uint8_t*, size_t);
  int normal_distribution(const uint8_t*, size_t);
  int lognormal_distribution(const uint8_t*, size_t);
  int chi_squared_distribution(const uint8_t*, size_t);
  int cauchy_distribution(const uint8_t*, size_t);
  int fisher_f_distribution(const uint8_t*, size_t);
  int student_t_distribution(const uint8_t*, size_t);
  int discrete_distribution(const uint8_t*, size_t);
  int piecewise_constant_distribution(const uint8_t*, size_t);
  int piecewise_linear_distribution(const uint8_t*, size_t);

} // namespace fuzzing

#endif // _LIBCPP_FUZZING
