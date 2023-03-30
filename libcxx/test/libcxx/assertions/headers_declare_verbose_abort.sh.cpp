//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that all public C++ headers define the verbose termination function, which
// is required for users to be able to include any public header and then override
// the function using a strong definition.

// The system-provided <uchar.h> seems to be broken on AIX, which trips up this test.
// XFAIL: LIBCXX-AIX-FIXME

// XFAIL: availability-verbose_abort-missing

/*
BEGIN-SCRIPT

for i, header in enumerate(public_headers):
    # Skip C compatibility headers.
    if header.endswith('.h'):
        continue

    vars = {
        'run': 'RUN',
        'i': i,
        'restrictions': ' && ' + header_restrictions[header] if header in header_restrictions else '',
        'header': header
    }

    print("""\
// {run}: %{{build}} -DTEST_{i}
#if defined(TEST_{i}){restrictions}
#   include <{header}>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif
""".format(**vars))

END-SCRIPT
*/

#include <__config>

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#   undef __DEPRECATED
#endif

int main(int, char**) { return 0; }

// DO NOT MANUALLY EDIT ANYTHING BETWEEN THE MARKERS BELOW
// GENERATED-MARKER
// RUN: %{build} -DTEST_0
#if defined(TEST_0)
#   include <algorithm>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_1
#if defined(TEST_1)
#   include <any>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_2
#if defined(TEST_2)
#   include <array>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_3
#if defined(TEST_3)
#   include <atomic>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_4
#if defined(TEST_4) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <barrier>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_5
#if defined(TEST_5)
#   include <bit>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_6
#if defined(TEST_6)
#   include <bitset>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_7
#if defined(TEST_7)
#   include <cassert>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_8
#if defined(TEST_8)
#   include <ccomplex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_9
#if defined(TEST_9)
#   include <cctype>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_10
#if defined(TEST_10)
#   include <cerrno>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_11
#if defined(TEST_11)
#   include <cfenv>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_12
#if defined(TEST_12)
#   include <cfloat>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_13
#if defined(TEST_13)
#   include <charconv>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_14
#if defined(TEST_14)
#   include <chrono>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_15
#if defined(TEST_15)
#   include <cinttypes>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_16
#if defined(TEST_16)
#   include <ciso646>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_17
#if defined(TEST_17)
#   include <climits>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_18
#if defined(TEST_18) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <clocale>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_19
#if defined(TEST_19)
#   include <cmath>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_20
#if defined(TEST_20) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <codecvt>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_21
#if defined(TEST_21)
#   include <compare>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_22
#if defined(TEST_22)
#   include <complex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_24
#if defined(TEST_24)
#   include <concepts>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_25
#if defined(TEST_25)
#   include <condition_variable>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_26
#if defined(TEST_26) && (defined(__cpp_impl_coroutine) && __cpp_impl_coroutine >= 201902L) || (defined(__cpp_coroutines) && __cpp_coroutines >= 201703L)
#   include <coroutine>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_27
#if defined(TEST_27)
#   include <csetjmp>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_28
#if defined(TEST_28)
#   include <csignal>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_29
#if defined(TEST_29)
#   include <cstdarg>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_30
#if defined(TEST_30)
#   include <cstdbool>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_31
#if defined(TEST_31)
#   include <cstddef>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_32
#if defined(TEST_32)
#   include <cstdint>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_33
#if defined(TEST_33)
#   include <cstdio>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_34
#if defined(TEST_34)
#   include <cstdlib>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_35
#if defined(TEST_35)
#   include <cstring>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_36
#if defined(TEST_36)
#   include <ctgmath>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_37
#if defined(TEST_37)
#   include <ctime>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_39
#if defined(TEST_39)
#   include <cuchar>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_40
#if defined(TEST_40) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <cwchar>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_41
#if defined(TEST_41) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <cwctype>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_42
#if defined(TEST_42)
#   include <deque>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_44
#if defined(TEST_44)
#   include <exception>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_45
#if defined(TEST_45)
#   include <execution>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_46
#if defined(TEST_46)
#   include <expected>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_48
#if defined(TEST_48) && !defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#   include <filesystem>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_50
#if defined(TEST_50)
#   include <format>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_51
#if defined(TEST_51)
#   include <forward_list>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_52
#if defined(TEST_52) && !defined(_LIBCPP_HAS_NO_LOCALIZATION) && !defined(_LIBCPP_HAS_NO_FSTREAM)
#   include <fstream>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_53
#if defined(TEST_53)
#   include <functional>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_54
#if defined(TEST_54) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <future>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_55
#if defined(TEST_55)
#   include <initializer_list>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_57
#if defined(TEST_57) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <iomanip>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_58
#if defined(TEST_58) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <ios>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_59
#if defined(TEST_59)
#   include <iosfwd>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_60
#if defined(TEST_60) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <iostream>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_61
#if defined(TEST_61) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <istream>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_62
#if defined(TEST_62)
#   include <iterator>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_63
#if defined(TEST_63) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <latch>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_64
#if defined(TEST_64)
#   include <limits>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_66
#if defined(TEST_66)
#   include <list>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_67
#if defined(TEST_67) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <locale>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_69
#if defined(TEST_69)
#   include <map>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_71
#if defined(TEST_71)
#   include <memory>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_72
#if defined(TEST_72)
#   include <memory_resource>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_73
#if defined(TEST_73) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <mutex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_74
#if defined(TEST_74)
#   include <new>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_75
#if defined(TEST_75)
#   include <numbers>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_76
#if defined(TEST_76)
#   include <numeric>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_77
#if defined(TEST_77)
#   include <optional>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_78
#if defined(TEST_78) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <ostream>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_79
#if defined(TEST_79)
#   include <queue>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_80
#if defined(TEST_80)
#   include <random>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_81
#if defined(TEST_81)
#   include <ranges>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_82
#if defined(TEST_82)
#   include <ratio>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_83
#if defined(TEST_83) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <regex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_84
#if defined(TEST_84)
#   include <scoped_allocator>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_85
#if defined(TEST_85) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <semaphore>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_86
#if defined(TEST_86)
#   include <set>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_88
#if defined(TEST_88) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <shared_mutex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_89
#if defined(TEST_89)
#   include <source_location>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_90
#if defined(TEST_90)
#   include <span>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_91
#if defined(TEST_91) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <sstream>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_92
#if defined(TEST_92)
#   include <stack>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_96
#if defined(TEST_96)
#   include <stdexcept>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_100
#if defined(TEST_100) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <streambuf>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_101
#if defined(TEST_101)
#   include <string>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_103
#if defined(TEST_103)
#   include <string_view>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_104
#if defined(TEST_104) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <strstream>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_105
#if defined(TEST_105)
#   include <system_error>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_107
#if defined(TEST_107) && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <thread>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_108
#if defined(TEST_108)
#   include <tuple>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_109
#if defined(TEST_109)
#   include <type_traits>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_110
#if defined(TEST_110)
#   include <typeindex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_111
#if defined(TEST_111)
#   include <typeinfo>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_113
#if defined(TEST_113)
#   include <unordered_map>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_114
#if defined(TEST_114)
#   include <unordered_set>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_115
#if defined(TEST_115)
#   include <utility>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_116
#if defined(TEST_116)
#   include <valarray>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_117
#if defined(TEST_117)
#   include <variant>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_118
#if defined(TEST_118)
#   include <vector>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_119
#if defined(TEST_119)
#   include <version>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_122
#if defined(TEST_122) && __cplusplus >= 201103L
#   include <experimental/deque>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_123
#if defined(TEST_123) && __cplusplus >= 201103L
#   include <experimental/forward_list>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_124
#if defined(TEST_124) && __cplusplus >= 201103L
#   include <experimental/iterator>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_125
#if defined(TEST_125) && __cplusplus >= 201103L
#   include <experimental/list>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_126
#if defined(TEST_126) && __cplusplus >= 201103L
#   include <experimental/map>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_127
#if defined(TEST_127) && __cplusplus >= 201103L
#   include <experimental/memory_resource>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_128
#if defined(TEST_128) && __cplusplus >= 201103L
#   include <experimental/propagate_const>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_129
#if defined(TEST_129) && !defined(_LIBCPP_HAS_NO_LOCALIZATION) && __cplusplus >= 201103L
#   include <experimental/regex>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_130
#if defined(TEST_130) && __cplusplus >= 201103L
#   include <experimental/set>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_131
#if defined(TEST_131) && __cplusplus >= 201103L
#   include <experimental/simd>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_132
#if defined(TEST_132) && __cplusplus >= 201103L
#   include <experimental/string>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_133
#if defined(TEST_133) && __cplusplus >= 201103L
#   include <experimental/type_traits>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_134
#if defined(TEST_134) && __cplusplus >= 201103L
#   include <experimental/unordered_map>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_135
#if defined(TEST_135) && __cplusplus >= 201103L
#   include <experimental/unordered_set>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_136
#if defined(TEST_136) && __cplusplus >= 201103L
#   include <experimental/utility>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_137
#if defined(TEST_137) && __cplusplus >= 201103L
#   include <experimental/vector>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_138
#if defined(TEST_138)
#   include <ext/hash_map>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// RUN: %{build} -DTEST_139
#if defined(TEST_139)
#   include <ext/hash_set>
    using HandlerType = decltype(std::__libcpp_verbose_abort);
#endif

// GENERATED-MARKER
