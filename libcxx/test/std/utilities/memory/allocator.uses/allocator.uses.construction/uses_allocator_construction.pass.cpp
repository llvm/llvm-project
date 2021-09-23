//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

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
// template <class T, class Alloc, class... Args>
//   constexpr T make_obj_using_allocator(const Alloc& alloc, Args&&... args);
// template <class T, class Alloc, class... Args>
//   constexpr T* uninitialized_construct_using_allocator(T* p, const Alloc& alloc,
//                                                        Args&&... args);

#include <memory>
#include <cassert>
#include <tuple>

#include "test_macros.h"
#include "uses_alloc_types.h"

using AllocatorType = std::allocator<int>;
using UsesLeading = UsesAllocatorV1<AllocatorType, /*N args = */1>;
using DefaultConstructibleUsesLeading = UsesAllocatorV1<AllocatorType, /*N args = */0>;
using UsesTrailing = UsesAllocatorV2<AllocatorType, /*N args = */1>;
using DefaultConstructibleUsesTrailing = UsesAllocatorV2<AllocatorType, /*N args = */0>;
using UsesBoth = UsesAllocatorV3<AllocatorType, /*N args = */1>;
using DefaultConstructibleUsesBoth = UsesAllocatorV3<AllocatorType, /*N args = */0>;
using NotUses = NotUsesAllocator<AllocatorType, /*N args = */1>;
using DefaultConstructibleNotUses = NotUsesAllocator<AllocatorType, /*N args = */0>;

using TupleOfAll = std::tuple<UsesLeading, UsesTrailing, UsesBoth, NotUses>;
AllocatorType alloc;

template <typename Tuple1, class Tuple2,
          typename Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple1>>::value>>
void check_tuples(Tuple1&&, Tuple2&&);

template <class T>
struct IsTuple : std::false_type {};

template <class... Args>
struct IsTuple<std::tuple<Args...>> : std::true_type {};

void check_single_arg(std::allocator_arg_t, std::allocator_arg_t) {}
void check_single_arg(std::piecewise_construct_t, std::piecewise_construct_t) {}

void check_single_arg(const std::allocator<int>& alloc1, const std::allocator<int>& alloc2) {
    assert(&alloc1 == &alloc2);
}

template <class T>
void check_single_arg(T&& first, T&& second) {
    assert(first == second);
}

template <class Arg1, class Arg2>
void check_arg(Arg1&& arg1, Arg2&& arg2) {
    if constexpr (IsTuple<std::decay_t<Arg1>>::value) {
        check_tuples(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
    } else {
        check_single_arg(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
    }
}

template <class Tuple1, class Tuple2, std::size_t... Idx>
void check_tuples_impl(Tuple1&& expected_args, Tuple2&& actual_args, std::index_sequence<Idx...>) {
    (check_arg(std::get<Idx>(std::forward<Tuple1>(expected_args)),
               std::get<Idx>(std::forward<Tuple2>(actual_args))), ...);
}

template <class Tuple1, class Tuple2, typename Indices>
void check_tuples(Tuple1&& expected_args, Tuple2&& actual_args) {
    check_tuples_impl(std::forward<Tuple1>(expected_args), std::forward<Tuple2>(actual_args), Indices{});
}

template <class T, class ExpectedArgsTuple, class... Args>
void test_args(const ExpectedArgsTuple& expected_tuple, Args&&... args) {
    ASSERT_NOEXCEPT(std::uses_allocator_construction_args<T>(alloc, std::forward<Args>(args)...));
    auto construction_args = std::uses_allocator_construction_args<T>(alloc, std::forward<Args>(args)...);
    ASSERT_SAME_TYPE(decltype(construction_args), ExpectedArgsTuple);
    check_tuples(construction_args, expected_tuple);
}

template <class T, class... Args>
void test_object(UsesAllocatorType expected_construction, Args... args) {
    auto object = std::make_obj_using_allocator<T>(alloc, args...);
    ASSERT_SAME_TYPE(decltype(object), T);
    assert(expected_construction == object.constructor_called);

    using AllocatorTraits = std::allocator_traits<AllocatorType>;
    using TAlloc = AllocatorTraits::template rebind_alloc<T>;
    using TTraits = std::allocator_traits<TAlloc>;

    TAlloc t_alloc(alloc);
    T* object_ptr1 = TTraits::allocate(t_alloc, 1);
    auto object_ptr2 = std::uninitialized_construct_using_allocator(object_ptr1, alloc, args...);
    ASSERT_SAME_TYPE(decltype(object_ptr2), T*);
    assert(object_ptr1 == object_ptr2);
    assert(expected_construction == object_ptr1->constructor_called);
    std::destroy_at(object_ptr1);
    TTraits::deallocate(t_alloc, object_ptr1, 1);
}

template <class Pair, class... Args>
void test_pair_object(UsesAllocatorType expected_first, UsesAllocatorType expected_second, Args... args) {
    auto pair = std::make_obj_using_allocator<Pair>(alloc, args...);
    ASSERT_SAME_TYPE(decltype(pair), Pair);
    assert(expected_first == pair.first.constructor_called);
    assert(expected_second == pair.second.constructor_called);

    using AllocatorTraits = std::allocator_traits<AllocatorType>;
    using PairAlloc = AllocatorTraits::template rebind_alloc<Pair>;
    using PairTraits = std::allocator_traits<PairAlloc>;

    PairAlloc t_alloc(alloc);
    Pair* object_ptr1 = PairTraits::allocate(t_alloc, 1);
    auto object_ptr2 = std::uninitialized_construct_using_allocator(object_ptr1, alloc, args...);
    ASSERT_SAME_TYPE(decltype(object_ptr1), Pair*);
    assert(object_ptr1 == object_ptr2);
    assert(expected_first == object_ptr1->first.constructor_called);
    assert(expected_second == object_ptr1->second.constructor_called);
    std::destroy_at(object_ptr1);
    PairTraits::deallocate(t_alloc, object_ptr1, 1);
}

template <class Tuple>
auto get_leading_tuple(Tuple&& arguments) {
    return std::tuple_cat(std::tuple<std::allocator_arg_t,
                                     const std::allocator<int>&>{std::allocator_arg, alloc},
                          std::forward<Tuple>(arguments));
}

template <class Tuple>
auto get_trailing_tuple(Tuple&& arguments) {
    return std::tuple_cat(std::forward<Tuple>(arguments),
                          std::tuple<const std::allocator<int>&>{alloc});
}

template <class... Args>
void test_non_pair_basic(Args&&... args) {
    test_args<NotUses>(std::forward_as_tuple(std::forward<Args>(args)...),
                       std::forward<Args>(args)...);
    test_object<NotUses>(UsesAllocatorType::UA_None, std::forward<Args>(args)...);

    std::tuple expected_leading = get_leading_tuple(std::forward_as_tuple(std::forward<Args>(args)...));
    test_args<UsesLeading>(expected_leading, std::forward<Args>(args)...);
    test_args<UsesBoth>(expected_leading, std::forward<Args>(args)...);
    test_object<UsesLeading>(UsesAllocatorType::UA_AllocArg, std::forward<Args>(args)...);
    test_object<UsesLeading>(UsesAllocatorType::UA_AllocArg, std::forward<Args>(args)...);

    test_args<UsesTrailing>(get_trailing_tuple(std::forward_as_tuple(std::forward<Args>(args)...)),
                            std::forward<Args>(args)...);
    test_object<UsesTrailing>(UsesAllocatorType::UA_AllocLast, std::forward<Args>(args)...);
}

void test_non_pair() {
    int value = 1;

    test_non_pair_basic(value);
    test_non_pair_basic(std::as_const(value));
    test_non_pair_basic(1);
}

template <class T, class Tuple>
auto get_expected_args(Tuple&& tuple) {
    if constexpr (std::is_same_v<T, UsesLeading> ||
                  std::is_same_v<T, DefaultConstructibleUsesLeading> ||
                  std::is_same_v<T, UsesBoth> ||
                  std::is_same_v<T, DefaultConstructibleUsesBoth>) {
        return get_leading_tuple(std::forward<Tuple>(tuple));
    } else if constexpr (std::is_same_v<T, UsesTrailing> ||
                         std::is_same_v<T, DefaultConstructibleUsesTrailing>) {
        return get_trailing_tuple(std::forward<Tuple>(tuple));
    } else {
        static_assert(std::is_same_v<T, NotUses> ||
                      std::is_same_v<T, DefaultConstructibleNotUses>);
        return std::forward<Tuple>(tuple);
    }
}

template <class T>
UsesAllocatorType get_expected_state() {
    if constexpr (std::is_same_v<T, UsesLeading> ||
                  std::is_same_v<T, DefaultConstructibleUsesLeading> ||
                  std::is_same_v<T, UsesBoth> ||
                  std::is_same_v<T, DefaultConstructibleUsesBoth>) {
        return UsesAllocatorType::UA_AllocArg;
    } else if constexpr (std::is_same_v<T, UsesTrailing> ||
                         std::is_same_v<T, DefaultConstructibleUsesTrailing>) {
        return UsesAllocatorType::UA_AllocLast;
    } else {
        static_assert(std::is_same_v<T, NotUses> ||
                      std::is_same_v<T, DefaultConstructibleNotUses>);
        return UsesAllocatorType::UA_None;
    }
}

template <class Pair, class Tuple1, class Tuple2>
void test_pair_tuples(Tuple1&& tuple1, Tuple2&& tuple2) {
    using FirstType = typename Pair::first_type;
    using SecondType = typename Pair::second_type;

    std::tuple expected_tuple{std::piecewise_construct,
                              get_expected_args<FirstType>(std::forward<Tuple1>(tuple1)),
                              get_expected_args<SecondType>(std::forward<Tuple2>(tuple2))};
    test_args<Pair>(expected_tuple, std::piecewise_construct,
                    std::forward<Tuple1>(tuple1), std::forward<Tuple2>(tuple2));
    test_pair_object<Pair>(get_expected_state<FirstType>(),
                           get_expected_state<SecondType>(),
                           std::piecewise_construct, std::forward<Tuple1>(tuple1),
                           std::forward<Tuple2>(tuple2));
}

template <class Pair>
void test_pair_alloc() {
    using FirstType = typename Pair::first_type;
    using SecondType = typename Pair::second_type;

    std::tuple expected_tuple{std::piecewise_construct,
                              get_expected_args<FirstType>(std::tuple{}),
                              get_expected_args<SecondType>(std::tuple{})};
    test_args<Pair>(expected_tuple);
    test_pair_object<Pair>(get_expected_state<FirstType>(), get_expected_state<SecondType>());
}

template <class Pair, class U, class V>
void test_pair_uv(U&& u, V&& v) {
    using FirstType = typename Pair::first_type;
    using SecondType = typename Pair::second_type;

    std::tuple expected_tuple{std::piecewise_construct,
                              get_expected_args<FirstType>(std::forward_as_tuple(std::forward<U>(u))),
                              get_expected_args<SecondType>(std::forward_as_tuple(std::forward<V>(v)))};
    test_args<Pair>(expected_tuple, std::forward<U>(u), std::forward<V>(v));
    test_pair_object<Pair>(get_expected_state<FirstType>(), get_expected_state<SecondType>(),
                           std::forward<U>(u), std::forward<V>(v));
}

template <class Pair, class U, class V>
void test_pair_lpair(const std::pair<U, V>& pair) {
    using FirstType = typename Pair::first_type;
    using SecondType = typename Pair::second_type;

    std::tuple expected_tuple{std::piecewise_construct,
                              get_expected_args<FirstType>(std::forward_as_tuple(pair.first)),
                              get_expected_args<SecondType>(std::forward_as_tuple(pair.second))};
    test_args<Pair>(expected_tuple, pair);
    test_pair_object<Pair>(get_expected_state<FirstType>(), get_expected_state<SecondType>(),
                           pair);
}

template <class Pair, class U, class V>
void test_pair_rpair(std::pair<U, V>&& pair) {
    using FirstType = typename Pair::first_type;
    using SecondType = typename Pair::second_type;

    std::tuple expected_tuple{std::piecewise_construct,
                              get_expected_args<FirstType>(std::forward_as_tuple(std::move(pair.first))),
                              get_expected_args<SecondType>(std::forward_as_tuple(std::move(pair.second)))};
    test_args<Pair>(expected_tuple, std::move(pair));
    test_pair_object<Pair>(get_expected_state<FirstType>(), get_expected_state<SecondType>(),
                           std::move(pair));
}

template <class Pair>
struct CorrectPair;

template <template <class, std::size_t> class Base1,
          template <class, std::size_t> class Base2,
          class Alloc, std::size_t N1, std::size_t N2>
struct CorrectPair<std::pair<Base1<Alloc, N1>, Base2<Alloc, N2>>> {
    // In general tests the uses allocator types with non-zero arity are used
    // For correctness, uses allocator types with arity 0 should be used
    using type = std::pair<Base1<Alloc, 0>, Base2<Alloc, 0>>;
};

template <class Pair>
void test_pair_basic() {
    int value = 1;
    bool bvalue = false;
    test_pair_tuples<Pair>(std::forward_as_tuple(value),
                           std::forward_as_tuple(std::as_const(value)));
    test_pair_tuples<Pair>(std::forward_as_tuple(1),
                           std::forward_as_tuple(value));
    test_pair_tuples<Pair>(std::forward_as_tuple(std::as_const(value)),
                           std::forward_as_tuple(1));

    test_pair_alloc<typename CorrectPair<Pair>::type>();

    test_pair_uv<Pair>(value, std::as_const(bvalue));
    test_pair_uv<Pair>(1, bvalue);
    test_pair_uv<Pair>(std::as_const(value), false);

    std::pair lpair{value, bvalue};
    test_pair_lpair<Pair>(lpair);
    test_pair_rpair<Pair>(std::move(lpair));
}

template <class... Args> struct SecondInPair;

template <class FirstArg, class Arg, class... Args>
struct SecondInPair<FirstArg, Arg, Args...> {
    static void iterate() {
        using PairToTest = std::pair<FirstArg, Arg>;
        test_pair_basic<PairToTest>();
        SecondInPair<FirstArg, Args...>::iterate();
    }
};

template <class FirstArg, class Arg>
struct SecondInPair<FirstArg, Arg> {
    static void iterate() {
        using PairToTest = std::pair<FirstArg, Arg>;
        test_pair_basic<PairToTest>();
    }
};

template <class... Args> struct FirstInPair;

template <class Arg, class... Args1>
struct FirstInPair<Arg, Args1...> {
    template <class... Args2>
    static void iterate(std::tuple<Args2...> tpl) {
        SecondInPair<Arg, Args2...>::iterate();
        FirstInPair<Args1...>::iterate(tpl);
    }
};

template <class Arg>
struct FirstInPair<Arg> {
    template <class... Args2>
    static void iterate(std::tuple<Args2...>) {
        SecondInPair<Arg, Args2...>::iterate();
    }
};

template <class... Args>
void test_pair(std::tuple<Args...> tpl) {
    FirstInPair<Args...>::iterate(tpl);
}

// // TODO: test constexpr

int main(int, char**) {
    test_non_pair();
    test_pair(TupleOfAll{});
}
