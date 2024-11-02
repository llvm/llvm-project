//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// ensure that all result alias declarations are defined

#include <algorithm>
#include <memory>
#include <type_traits>

using namespace std::ranges;

static_assert(std::is_same_v<in_fun_result<int, long>, for_each_result<int, long>>);
static_assert(std::is_same_v<in_fun_result<int, long>, for_each_n_result<int, long>>);

static_assert(std::is_same_v<in_in_result<int, long>, mismatch_result<int, long>>);
static_assert(std::is_same_v<in_in_result<int, long>, swap_ranges_result<int, long>>);

static_assert(std::is_same_v<in_out_result<int, long>, copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, copy_n_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, copy_if_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, copy_backward_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, move_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, move_backward_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, partial_sort_copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, remove_copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, remove_copy_if_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, replace_copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, replace_copy_if_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, reverse_copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, rotate_copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, set_difference_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, unary_transform_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, uninitialized_copy_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, uninitialized_copy_n_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, uninitialized_move_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, uninitialized_move_n_result<int, long>>);
static_assert(std::is_same_v<in_out_result<int, long>, unique_copy_result<int, long>>);

static_assert(std::is_same_v<in_in_out_result<int, long, char>, binary_transform_result<int, long, char>>);
static_assert(std::is_same_v<in_in_out_result<int, long, char>, merge_result<int, long, char>>);
static_assert(std::is_same_v<in_in_out_result<int, long, char>, set_symmetric_difference_result<int, long, char>>);
static_assert(std::is_same_v<in_in_out_result<int, long, char>, set_union_result<int, long, char>>);
static_assert(std::is_same_v<in_in_out_result<int, long, char>, set_intersection_result<int, long, char>>);

static_assert(std::is_same_v<in_out_out_result<int, long, char>, partition_copy_result<int, long, char>>);

static_assert(std::is_same_v<min_max_result<int>, minmax_result<int>>);
static_assert(std::is_same_v<min_max_result<int>, minmax_element_result<int>>);

static_assert(std::is_same_v<in_found_result<int>, next_permutation_result<int>>);
static_assert(std::is_same_v<in_found_result<int>, prev_permutation_result<int>>);

// static_assert(std::is_same_v<out_value_result<int>, iota_result<int>>);
