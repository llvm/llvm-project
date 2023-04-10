//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we don't remove transitive includes of public C++ headers in the library accidentally.
// When we remove a transitive public include, clients tend to break because they don't always
// properly include what they use. Note that we don't check which system (C) headers are
// included transitively, because that is too unstable across platforms, and hence difficult
// to test for.
//
// This is not meant to block libc++ from removing unused transitive includes
// forever, however we do try to group removals for a couple of releases
// to avoid breaking users at every release.

// This test doesn't support being run when some headers are not available, since we
// would need to add significant complexity to make that work.
// UNSUPPORTED: no-localization, no-threads, no-wide-characters, no-filesystem, libcpp-has-no-incomplete-format

// When built with modules, this test doesn't work because --trace-includes doesn't
// report the stack of includes correctly.
// UNSUPPORTED: modules-build

// This test uses --trace-includes, which is not supported by GCC.
// UNSUPPORTED: gcc

// This test is not supported when we remove the transitive includes provided for backwards
// compatibility. When we bulk-remove them, we'll adjust the includes that are expected by
// this test instead.
// UNSUPPORTED: transitive-includes-disabled

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#    undef __DEPRECATED
#endif

/*
BEGIN-SCRIPT

import re

# To re-generate the list of expected headers, temporarily set this to True, re-generate
# the file and run this test.
# Note that this needs to be done for all supported language versions of libc++:
# for std in c++03 c++11 c++14 c++17 c++20 c++2b; do <build>/bin/llvm-lit --param std=$std ${path_to_this_file}; done
regenerate_expected_results = False
# Used because the sequence of tokens RUN : can't appear anywhere or it'll confuse Lit.
RUN = "RUN"

print(f"// {RUN}: mkdir %t")

for i, header in enumerate(public_headers):
  if header.endswith('.h'): # Skip C compatibility or detail headers
    continue

  normalized_header = re.sub('/', '_', header)
  print(f"// {RUN}: %{{cxx}} %s %{{flags}} %{{compile_flags}} --trace-includes -fshow-skipped-includes --preprocess -DTEST_{i} 2> %t/header.{normalized_header}")
  print(f"#if defined(TEST_{i})")
  print(f"#include <{header}>")
  print("#endif")

if regenerate_expected_results:
  print(f"// {RUN}: %{{python}} %S/transitive_includes_to_csv.py %t > %S/transitive_includes/%{{cxx_std}}.csv")
else:
  print(f"// {RUN}: %{{python}} %S/transitive_includes_to_csv.py %t > %t/transitive_includes.csv")
  print(f"// {RUN}: diff -w %S/transitive_includes/%{{cxx_std}}.csv %t/transitive_includes.csv")

END-SCRIPT
*/

// DO NOT MANUALLY EDIT ANYTHING BETWEEN THE MARKERS BELOW
// GENERATED-MARKER
// RUN: mkdir %t
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_0 2> %t/header.algorithm
#if defined(TEST_0)
#include <algorithm>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_1 2> %t/header.any
#if defined(TEST_1)
#include <any>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_2 2> %t/header.array
#if defined(TEST_2)
#include <array>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_3 2> %t/header.atomic
#if defined(TEST_3)
#include <atomic>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_4 2> %t/header.barrier
#if defined(TEST_4)
#include <barrier>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_5 2> %t/header.bit
#if defined(TEST_5)
#include <bit>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_6 2> %t/header.bitset
#if defined(TEST_6)
#include <bitset>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_7 2> %t/header.cassert
#if defined(TEST_7)
#include <cassert>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_8 2> %t/header.ccomplex
#if defined(TEST_8)
#include <ccomplex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_9 2> %t/header.cctype
#if defined(TEST_9)
#include <cctype>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_10 2> %t/header.cerrno
#if defined(TEST_10)
#include <cerrno>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_11 2> %t/header.cfenv
#if defined(TEST_11)
#include <cfenv>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_12 2> %t/header.cfloat
#if defined(TEST_12)
#include <cfloat>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_13 2> %t/header.charconv
#if defined(TEST_13)
#include <charconv>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_14 2> %t/header.chrono
#if defined(TEST_14)
#include <chrono>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_15 2> %t/header.cinttypes
#if defined(TEST_15)
#include <cinttypes>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_16 2> %t/header.ciso646
#if defined(TEST_16)
#include <ciso646>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_17 2> %t/header.climits
#if defined(TEST_17)
#include <climits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_18 2> %t/header.clocale
#if defined(TEST_18)
#include <clocale>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_19 2> %t/header.cmath
#if defined(TEST_19)
#include <cmath>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_20 2> %t/header.codecvt
#if defined(TEST_20)
#include <codecvt>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_21 2> %t/header.compare
#if defined(TEST_21)
#include <compare>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_22 2> %t/header.complex
#if defined(TEST_22)
#include <complex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_24 2> %t/header.concepts
#if defined(TEST_24)
#include <concepts>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_25 2> %t/header.condition_variable
#if defined(TEST_25)
#include <condition_variable>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_26 2> %t/header.coroutine
#if defined(TEST_26)
#include <coroutine>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_27 2> %t/header.csetjmp
#if defined(TEST_27)
#include <csetjmp>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_28 2> %t/header.csignal
#if defined(TEST_28)
#include <csignal>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_29 2> %t/header.cstdarg
#if defined(TEST_29)
#include <cstdarg>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_30 2> %t/header.cstdbool
#if defined(TEST_30)
#include <cstdbool>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_31 2> %t/header.cstddef
#if defined(TEST_31)
#include <cstddef>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_32 2> %t/header.cstdint
#if defined(TEST_32)
#include <cstdint>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_33 2> %t/header.cstdio
#if defined(TEST_33)
#include <cstdio>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_34 2> %t/header.cstdlib
#if defined(TEST_34)
#include <cstdlib>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_35 2> %t/header.cstring
#if defined(TEST_35)
#include <cstring>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_36 2> %t/header.ctgmath
#if defined(TEST_36)
#include <ctgmath>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_37 2> %t/header.ctime
#if defined(TEST_37)
#include <ctime>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_39 2> %t/header.cuchar
#if defined(TEST_39)
#include <cuchar>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_40 2> %t/header.cwchar
#if defined(TEST_40)
#include <cwchar>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_41 2> %t/header.cwctype
#if defined(TEST_41)
#include <cwctype>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_42 2> %t/header.deque
#if defined(TEST_42)
#include <deque>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_44 2> %t/header.exception
#if defined(TEST_44)
#include <exception>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_45 2> %t/header.execution
#if defined(TEST_45)
#include <execution>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_46 2> %t/header.expected
#if defined(TEST_46)
#include <expected>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_48 2> %t/header.filesystem
#if defined(TEST_48)
#include <filesystem>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_50 2> %t/header.format
#if defined(TEST_50)
#include <format>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_51 2> %t/header.forward_list
#if defined(TEST_51)
#include <forward_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_52 2> %t/header.fstream
#if defined(TEST_52)
#include <fstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_53 2> %t/header.functional
#if defined(TEST_53)
#include <functional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_54 2> %t/header.future
#if defined(TEST_54)
#include <future>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_55 2> %t/header.initializer_list
#if defined(TEST_55)
#include <initializer_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_57 2> %t/header.iomanip
#if defined(TEST_57)
#include <iomanip>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_58 2> %t/header.ios
#if defined(TEST_58)
#include <ios>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_59 2> %t/header.iosfwd
#if defined(TEST_59)
#include <iosfwd>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_60 2> %t/header.iostream
#if defined(TEST_60)
#include <iostream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_61 2> %t/header.istream
#if defined(TEST_61)
#include <istream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_62 2> %t/header.iterator
#if defined(TEST_62)
#include <iterator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_63 2> %t/header.latch
#if defined(TEST_63)
#include <latch>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_64 2> %t/header.limits
#if defined(TEST_64)
#include <limits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_66 2> %t/header.list
#if defined(TEST_66)
#include <list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_67 2> %t/header.locale
#if defined(TEST_67)
#include <locale>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_69 2> %t/header.map
#if defined(TEST_69)
#include <map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_71 2> %t/header.memory
#if defined(TEST_71)
#include <memory>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_72 2> %t/header.memory_resource
#if defined(TEST_72)
#include <memory_resource>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_73 2> %t/header.mutex
#if defined(TEST_73)
#include <mutex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_74 2> %t/header.new
#if defined(TEST_74)
#include <new>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_75 2> %t/header.numbers
#if defined(TEST_75)
#include <numbers>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_76 2> %t/header.numeric
#if defined(TEST_76)
#include <numeric>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_77 2> %t/header.optional
#if defined(TEST_77)
#include <optional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_78 2> %t/header.ostream
#if defined(TEST_78)
#include <ostream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_79 2> %t/header.queue
#if defined(TEST_79)
#include <queue>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_80 2> %t/header.random
#if defined(TEST_80)
#include <random>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_81 2> %t/header.ranges
#if defined(TEST_81)
#include <ranges>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_82 2> %t/header.ratio
#if defined(TEST_82)
#include <ratio>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_83 2> %t/header.regex
#if defined(TEST_83)
#include <regex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_84 2> %t/header.scoped_allocator
#if defined(TEST_84)
#include <scoped_allocator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_85 2> %t/header.semaphore
#if defined(TEST_85)
#include <semaphore>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_86 2> %t/header.set
#if defined(TEST_86)
#include <set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_88 2> %t/header.shared_mutex
#if defined(TEST_88)
#include <shared_mutex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_89 2> %t/header.source_location
#if defined(TEST_89)
#include <source_location>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_90 2> %t/header.span
#if defined(TEST_90)
#include <span>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_91 2> %t/header.sstream
#if defined(TEST_91)
#include <sstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_92 2> %t/header.stack
#if defined(TEST_92)
#include <stack>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_96 2> %t/header.stdexcept
#if defined(TEST_96)
#include <stdexcept>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_100 2> %t/header.streambuf
#if defined(TEST_100)
#include <streambuf>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_101 2> %t/header.string
#if defined(TEST_101)
#include <string>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_103 2> %t/header.string_view
#if defined(TEST_103)
#include <string_view>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_104 2> %t/header.strstream
#if defined(TEST_104)
#include <strstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_105 2> %t/header.system_error
#if defined(TEST_105)
#include <system_error>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_107 2> %t/header.thread
#if defined(TEST_107)
#include <thread>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_108 2> %t/header.tuple
#if defined(TEST_108)
#include <tuple>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_109 2> %t/header.type_traits
#if defined(TEST_109)
#include <type_traits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_110 2> %t/header.typeindex
#if defined(TEST_110)
#include <typeindex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_111 2> %t/header.typeinfo
#if defined(TEST_111)
#include <typeinfo>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_113 2> %t/header.unordered_map
#if defined(TEST_113)
#include <unordered_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_114 2> %t/header.unordered_set
#if defined(TEST_114)
#include <unordered_set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_115 2> %t/header.utility
#if defined(TEST_115)
#include <utility>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_116 2> %t/header.valarray
#if defined(TEST_116)
#include <valarray>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_117 2> %t/header.variant
#if defined(TEST_117)
#include <variant>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_118 2> %t/header.vector
#if defined(TEST_118)
#include <vector>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_119 2> %t/header.version
#if defined(TEST_119)
#include <version>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_122 2> %t/header.experimental_deque
#if defined(TEST_122)
#include <experimental/deque>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_123 2> %t/header.experimental_forward_list
#if defined(TEST_123)
#include <experimental/forward_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_124 2> %t/header.experimental_iterator
#if defined(TEST_124)
#include <experimental/iterator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_125 2> %t/header.experimental_list
#if defined(TEST_125)
#include <experimental/list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_126 2> %t/header.experimental_map
#if defined(TEST_126)
#include <experimental/map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_127 2> %t/header.experimental_memory_resource
#if defined(TEST_127)
#include <experimental/memory_resource>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_128 2> %t/header.experimental_propagate_const
#if defined(TEST_128)
#include <experimental/propagate_const>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_129 2> %t/header.experimental_regex
#if defined(TEST_129)
#include <experimental/regex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_130 2> %t/header.experimental_set
#if defined(TEST_130)
#include <experimental/set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_131 2> %t/header.experimental_simd
#if defined(TEST_131)
#include <experimental/simd>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_132 2> %t/header.experimental_string
#if defined(TEST_132)
#include <experimental/string>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_133 2> %t/header.experimental_type_traits
#if defined(TEST_133)
#include <experimental/type_traits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_134 2> %t/header.experimental_unordered_map
#if defined(TEST_134)
#include <experimental/unordered_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_135 2> %t/header.experimental_unordered_set
#if defined(TEST_135)
#include <experimental/unordered_set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_136 2> %t/header.experimental_utility
#if defined(TEST_136)
#include <experimental/utility>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_137 2> %t/header.experimental_vector
#if defined(TEST_137)
#include <experimental/vector>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_138 2> %t/header.ext_hash_map
#if defined(TEST_138)
#include <ext/hash_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fshow-skipped-includes --preprocess -DTEST_139 2> %t/header.ext_hash_set
#if defined(TEST_139)
#include <ext/hash_set>
#endif
// RUN: %{python} %S/transitive_includes_to_csv.py %t > %t/transitive_includes.csv
// RUN: diff -w %S/transitive_includes/%{cxx_std}.csv %t/transitive_includes.csv
// GENERATED-MARKER
