//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>
//
// template <class T, class Alloc, class... Args>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   Args&&... args) noexcept;
// template <class T, class Alloc, class Tuple1, class Tuple2>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc, piecewise_construct_t,
//                                                   Tuple1&& x, Tuple2&& y)
//                                                   noexcept;
// template <class T, class Alloc>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc) noexcept;
// template <class T, class Alloc, class U, class V>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   U&& u, V&& v) noexcept;
// template <class T, class Alloc, class U, class V>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   const pair<U, V>& pr) noexcept;
// template <class T, class Alloc, class U, class V>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   pair<U, V>&& pr) noexcept;

#include <memory>
#include <cassert>

#include "test_macros.h"

enum ConstructionStatus {
    default_constructed,
    leading_alloc_default_constructed,
    trailing_alloc_default_constructed,
    int_constructed,
    leading_alloc_int_constructed,
    trailing_alloc_int_constructed
};

struct UsesLeadingAllocConstruction {
    using allocator_type = std::allocator<int>;

    UsesLeadingAllocConstruction() : status(ConstructionStatus::default_constructed) {}
    UsesLeadingAllocConstruction(int) : status(ConstructionStatus::int_constructed) {}
    UsesLeadingAllocConstruction(std::allocator_arg_t, const allocator_type&)
        : status(ConstructionStatus::leading_alloc_default_constructed) {}
    UsesLeadingAllocConstruction(std::allocator_arg_t, const allocator_type&, int)
        : status(ConstructionStatus::leading_alloc_int_constructed) {}

    ConstructionStatus status;
};

struct UsesTrailingAllocConstruction {
    using allocator_type = std::allocator<int>;

    UsesTrailingAllocConstruction() : status(ConstructionStatus::default_constructed) {}
    UsesTrailingAllocConstruction(int) : status(ConstructionStatus::int_constructed) {}
    UsesTrailingAllocConstruction(const allocator_type&)
        : status(ConstructionStatus::trailing_alloc_default_constructed) {}
    UsesTrailingAllocConstruction(int, const allocator_type&)
        : status(ConstructionStatus::trailing_alloc_int_constructed) {}

    ConstructionStatus status;
};

struct DoesNotUseAllocator {
    DoesNotUseAllocator() : status(ConstructionStatus::default_constructed) {}
    DoesNotUseAllocator(int) : status(ConstructionStatus::int_constructed) {}

    ConstructionStatus status;
};

template <class T, class ExpectedArgsTuple, class Alloc, class... Args>
void test_args(const ExpectedArgsTuple& expected_tuple, const Alloc& alloc, Args&&... args) {
    ASSERT_NOEXCEPT(std::uses_allocator_construction_args<T>(alloc, std::forward<Args>(args)...));
    auto construction_args = std::uses_allocator_construction_args<T>(alloc, std::forward<Args>(args)...);
    ASSERT_SAME_TYPE(decltype(construction_args), ExpectedArgsTuple);
    assert(construction_args == expected_tuple);
}

template <class... Args>
void test(Args... args) {
    std::allocator<int> alloc;
    std::tuple<Args...> arguments(args...);
    auto leading_arguments = std::tuple_cat(std::tuple<std::allocator_arg_t,
                                                       const std::allocator<int>&>{std::allocator_arg, alloc},
                                            arguments);
    auto trailing_arguments = std::tuple_cat(arguments, std::tuple<const std::allocator<int>&>{alloc});

    // test_args<DoesNotUseAllocator>(arguments, alloc, args...);
    test_args<UsesLeadingAllocConstruction>(leading_arguments, alloc, args...);
    // test_args<UsesTrailingAllocConstruction>(trailing_arguments, alloc, args...);
}

// template <class... Args1, class... Args2>
// void test_pair(std::tuple<Args1...> args1, std::tuple<Args2...> args2) {
    // using DoesNotUsePair = std::pair<DoesNotUseAllocator, DoesNotUseAllocator>;
    // using LeadingDoesNotUsePair = std::pair<UsesLeadingAllocConstruction, DoesNotUseAllocator>;
    // using DoesNotUseLeadingPair = std::pair<DoesNotUseAllocator, UsesLeadingAllocConstruction>;
    // using LeadingPair = std::pair<UsesLeadingAllocConstruction, UsesLeadingAllocConstruction>;
    // using TrailingDoesNotUsePair = std::pair<UsesTrailingAllocConstruction, DoesNotUseAllocator>;
    // using DoesNotUseTrailingPair = std::pair<DoesNotUseAllocator, UsesTrailingAllocConstruction>;
    // using TrailingPair = std::pair<UsesTrailingAllocConstruction, UsesTrailingAllocConstruction>;
    // using LeadingTrailingPair = std::pair<UsesLeadingAllocConstruction, UsesTrailingAllocConstruction>;
    // using TrailingLeadingPair = std::pair<UsesTrailingAllocConstruction, UsesLeadingAllocConstruction>;

    // std::allocator<int> alloc;
    // auto args1_tuple = std::tuple{args1};
    // auto args2_tuple = std::tuple{args2};
    // auto leading = std::tuple<std::allocator_arg_t, const std::allocator<int>&>{std::allocator_arg, alloc};
    // auto leading_args1 = std::tuple{std::tuple_cat(leading, args1_tuple)};
    // auto leading_args2 = std::tuple{std::tuple_cat(leading, args2_tuple)};
    // auto trailing = std::tuple<const std::allocator<int>&>{alloc};
    // auto trailing_args1 = std::tuple{std::tuple_cat(args1, trailing)};
    // auto trailing_args2 = std::tuple{std::tuple_cat(args2, trailing)};
    // auto piecewise_arg = std::tuple{std::piecewise_construct};

    // auto does_not_use_args = std::tuple_cat(piecewise_arg, args1_tuple, args2_tuple);
    // auto leading_does_not_use_args = std::tuple_cat(piecewise_arg, leading_args1, args2);
    // auto does_not_use_leading_args = std::tuple_cat(piecewise_arg, args1, leading_args2);
    // auto leading_args = std::tuple_cat(piecewise_arg, leading_args1, leading_args2);
    // auto trailing_does_not_use_args = std::tuple_cat(piecewise_arg, trailing_args1, args2);
    // auto does_not_use_trailing_args = std::tuple_cat(piecewise_arg, args1, trailing_args2);
    // auto trailing_args = std::tuple_cat(piecewise_arg, trailing_args1, trailing_args2);
    // auto leading_trailing_args = std::tuple_cat(piecewise_arg, leading_args1, trailing_args2);
    // auto trailing_leading_args = std::tuple_cat(piecewise_arg, trailing_args1, leading_args2);

    // test_args<DoesNotUsePair>(does_not_use_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<LeadingDoesNotUsePair>(leading_does_not_use_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<DoesNotUseLeadingPair>(does_not_use_leading_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<LeadingPair>(leading_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<TrailingDoesNotUsePair>(trailing_does_not_use_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<DoesNotUseTrailingPair>(does_not_use_trailing_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<TrailingPair>(trailing_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<LeadingTrailingPair>(leading_trailing_args, alloc, std::piecewise_construct, args1, args2);
    // test_args<TrailingLeadingPair>(trailing_leading_args, alloc, std::piecewise_construct, args1, args2);

    // test_args<DoesNotUsePair>(piecewise_arg, alloc);
    // test_args<LeadingDoesNotUsePair>(std::tuple_cat(piecewise_arg, std::tuple{leading}, std::tuple{std::tuple{}}), alloc);
    // test_args<DoesNotUseLeadingPair>(std::tuple_cat(piecewise_arg, std::tuple{std::tuple{}}, std::tuple{leading}), alloc);
    // test_args<LeadingPair>(std::tuple_cat(piecewise_arg, std::tuple{leading}, std::tuple{leading}), alloc);
    // test_args<TrailingDoesNotUsePair>(std::tuple_cat(piecewise_arg, std::tuple{trailing}, std::tuple{std::tuple{}}), alloc);
    // test_args<DoesNotUseTrailingPair>(std::tuple_cat(piecewise_arg, std::tuple{std::tuple{}}, std::tuple{trailing}), alloc);
    // test_args<TrailingPair>(std::tuple_cat(piecewise_arg, std::tuple{trailing}, std::tuple{trailing}), alloc);
    // test_args<LeadingTrailingPair>(std::tuple_cat(piecewise_arg, std::tuple{leading}, std::tuple{trailing}), alloc);
    // test_args<TrailingLeadingPair>(std::tuple_cat(piecewise_arg, std::tuple{trailing}, std::tuple{leading}), alloc);

    // int u_arg = 0;
    // std::tuple u_arg_tuple{u_arg};
    // auto leading_u_arg = std::tuple{std::tuple_cat(leading, u_arg_tuple)};
    // auto trailing_u_arg = std::tuple{std::tuple_cat(u_arg_tuple, trailing)};
    // test_args<DoesNotUsePair>(std::tuple_cat(piecewise_arg, std::tuple{u_arg_tuple}, std::tuple{u_arg_tuple}), alloc, u_arg, u_arg);
    // test_args<LeadingDoesNotUsePair>(std::tuple_cat(piecewise_arg, leading_u_arg, std::tuple{u_arg_tuple}), alloc, u_arg, u_arg);
    // test_args<DoesNotUseLeadingPair>(std::tuple_cat(piecewise_arg, std::tuple{u_arg_tuple}, leading_u_arg), alloc, u_arg, u_arg);
    // test_args<LeadingPair>(std::tuple_cat(piecewise_arg, leading_u_arg, leading_u_arg), alloc, u_arg, u_arg);
    // test_args<TrailingDoesNotUsePair>(std::tuple_cat(piecewise_arg, trailing_u_arg, std::tuple{u_arg_tuple}), alloc, u_arg, u_arg);
    // test_args<DoesNotUseTrailingPair>(std::tuple_cat(piecewise_arg, std::tuple{u_arg_tuple}, trailing_u_arg), alloc, u_arg, u_arg);
    // test_args<TrailingPair>(std::tuple_cat(piecewise_arg, trailing_u_arg, trailing_u_arg), alloc, u_arg, u_arg);
    // test_args<LeadingTrailingPair>(std::tuple_cat(piecewise_arg, leading_u_arg, trailing_u_arg), alloc, u_arg, u_arg);
    // test_args<TrailingPair>(std::tuple_cat(piecewise_arg, trailing_u_arg, leading_u_arg), alloc, u_arg, u_arg);

    // auto trailing_u_pair = std::tuple{std::tuple_cat(u_pair_tuple, leading)};
    // test_args<DoesNotUsePair>(std::tuple_cat(piecewise_arg, u_pair_tuple, u_pair_tuple), alloc, u_pair);
    // test_args<LeadingDoesNotUsePair>(std::tuple_cat(piecewise_arg, leading_u_pair, u_pair_tuple), alloc, u_pair);
// }

int main(int, char**) {
    test();
    // test(1);
}