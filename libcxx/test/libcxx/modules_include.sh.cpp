//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we can include each header in a TU while using modules.
// This is important notably because the LLDB data formatters use
// libc++ headers with modules enabled.

// The system-provided <uchar.h> seems to be broken on AIX
// XFAIL: LIBCXX-AIX-FIXME

// GCC doesn't support -fcxx-modules
// UNSUPPORTED: gcc

// The Windows headers don't appear to be compatible with modules
// UNSUPPORTED: windows

// The Android headers don't appear to be compatible with modules yet
// XFAIL: LIBCXX-ANDROID-FIXME

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#    undef __DEPRECATED
#endif

#include <__config>

/*
BEGIN-SCRIPT

for i, header in enumerate(public_headers):
  print("// {}: %{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_{}".format('RUN', i))
  if header in header_restrictions:
    print("#if defined(TEST_{}) && {}".format(i, header_restrictions[header]))
  else:
    print("#if defined(TEST_{})".format(i))
  print("#include <{}>".format(header))
  print("#endif")

END-SCRIPT
*/

// RUN: rm -rf %t
// RUN: mkdir %t

// DO NOT MANUALLY EDIT ANYTHING BETWEEN THE MARKERS BELOW
// GENERATED-MARKER
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_0
#if defined(TEST_0)
#include <algorithm>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_1
#if defined(TEST_1)
#include <any>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_2
#if defined(TEST_2)
#include <array>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_3
#if defined(TEST_3)
#include <atomic>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_4
#if defined(TEST_4) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <barrier>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_5
#if defined(TEST_5)
#include <bit>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_6
#if defined(TEST_6)
#include <bitset>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_7
#if defined(TEST_7)
#include <cassert>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_8
#if defined(TEST_8)
#include <ccomplex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_9
#if defined(TEST_9)
#include <cctype>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_10
#if defined(TEST_10)
#include <cerrno>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_11
#if defined(TEST_11)
#include <cfenv>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_12
#if defined(TEST_12)
#include <cfloat>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_13
#if defined(TEST_13)
#include <charconv>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_14
#if defined(TEST_14)
#include <chrono>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_15
#if defined(TEST_15)
#include <cinttypes>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_16
#if defined(TEST_16)
#include <ciso646>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_17
#if defined(TEST_17)
#include <climits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_18
#if defined(TEST_18) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <clocale>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_19
#if defined(TEST_19)
#include <cmath>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_20
#if defined(TEST_20) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <codecvt>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_21
#if defined(TEST_21)
#include <compare>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_22
#if defined(TEST_22)
#include <complex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_23
#if defined(TEST_23)
#include <complex.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_24
#if defined(TEST_24)
#include <concepts>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_25
#if defined(TEST_25)
#include <condition_variable>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_26
#if defined(TEST_26)
#include <coroutine>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_27
#if defined(TEST_27)
#include <csetjmp>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_28
#if defined(TEST_28)
#include <csignal>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_29
#if defined(TEST_29)
#include <cstdarg>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_30
#if defined(TEST_30)
#include <cstdbool>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_31
#if defined(TEST_31)
#include <cstddef>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_32
#if defined(TEST_32)
#include <cstdint>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_33
#if defined(TEST_33)
#include <cstdio>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_34
#if defined(TEST_34)
#include <cstdlib>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_35
#if defined(TEST_35)
#include <cstring>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_36
#if defined(TEST_36)
#include <ctgmath>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_37
#if defined(TEST_37)
#include <ctime>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_38
#if defined(TEST_38)
#include <ctype.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_39
#if defined(TEST_39)
#include <cuchar>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_40
#if defined(TEST_40) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <cwchar>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_41
#if defined(TEST_41) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <cwctype>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_42
#if defined(TEST_42)
#include <deque>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_43
#if defined(TEST_43)
#include <errno.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_44
#if defined(TEST_44)
#include <exception>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_45
#if defined(TEST_45)
#include <execution>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_46
#if defined(TEST_46)
#include <expected>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_47
#if defined(TEST_47)
#include <fenv.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_48
#if defined(TEST_48) && !defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#include <filesystem>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_49
#if defined(TEST_49)
#include <float.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_50
#if defined(TEST_50)
#include <format>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_51
#if defined(TEST_51)
#include <forward_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_52
#if defined(TEST_52) && !defined(_LIBCPP_HAS_NO_LOCALIZATION) && !defined(_LIBCPP_HAS_NO_FSTREAM)
#include <fstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_53
#if defined(TEST_53)
#include <functional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_54
#if defined(TEST_54) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <future>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_55
#if defined(TEST_55)
#include <initializer_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_56
#if defined(TEST_56)
#include <inttypes.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_57
#if defined(TEST_57) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <iomanip>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_58
#if defined(TEST_58) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <ios>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_59
#if defined(TEST_59)
#include <iosfwd>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_60
#if defined(TEST_60) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <iostream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_61
#if defined(TEST_61) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <istream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_62
#if defined(TEST_62)
#include <iterator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_63
#if defined(TEST_63) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <latch>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_64
#if defined(TEST_64)
#include <limits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_65
#if defined(TEST_65)
#include <limits.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_66
#if defined(TEST_66)
#include <list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_67
#if defined(TEST_67) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <locale>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_68
#if defined(TEST_68) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <locale.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_69
#if defined(TEST_69)
#include <map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_70
#if defined(TEST_70)
#include <math.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_71
#if defined(TEST_71)
#include <memory>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_72
#if defined(TEST_72)
#include <memory_resource>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_73
#if defined(TEST_73) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <mutex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_74
#if defined(TEST_74)
#include <new>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_75
#if defined(TEST_75)
#include <numbers>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_76
#if defined(TEST_76)
#include <numeric>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_77
#if defined(TEST_77)
#include <optional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_78
#if defined(TEST_78) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <ostream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_79
#if defined(TEST_79)
#include <queue>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_80
#if defined(TEST_80)
#include <random>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_81
#if defined(TEST_81)
#include <ranges>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_82
#if defined(TEST_82)
#include <ratio>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_83
#if defined(TEST_83) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <regex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_84
#if defined(TEST_84)
#include <scoped_allocator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_85
#if defined(TEST_85) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <semaphore>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_86
#if defined(TEST_86)
#include <set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_87
#if defined(TEST_87)
#include <setjmp.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_88
#if defined(TEST_88) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <shared_mutex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_89
#if defined(TEST_89)
#include <source_location>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_90
#if defined(TEST_90)
#include <span>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_91
#if defined(TEST_91) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <sstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_92
#if defined(TEST_92)
#include <stack>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_93
#if defined(TEST_93) && __cplusplus > 202002L && !defined(_LIBCPP_HAS_NO_THREADS)
#include <stdatomic.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_94
#if defined(TEST_94)
#include <stdbool.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_95
#if defined(TEST_95)
#include <stddef.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_96
#if defined(TEST_96)
#include <stdexcept>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_97
#if defined(TEST_97)
#include <stdint.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_98
#if defined(TEST_98)
#include <stdio.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_99
#if defined(TEST_99)
#include <stdlib.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_100
#if defined(TEST_100) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <streambuf>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_101
#if defined(TEST_101)
#include <string>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_102
#if defined(TEST_102)
#include <string.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_103
#if defined(TEST_103)
#include <string_view>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_104
#if defined(TEST_104) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <strstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_105
#if defined(TEST_105)
#include <system_error>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_106
#if defined(TEST_106)
#include <tgmath.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_107
#if defined(TEST_107) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <thread>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_108
#if defined(TEST_108)
#include <tuple>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_109
#if defined(TEST_109)
#include <type_traits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_110
#if defined(TEST_110)
#include <typeindex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_111
#if defined(TEST_111)
#include <typeinfo>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_112
#if defined(TEST_112)
#include <uchar.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_113
#if defined(TEST_113)
#include <unordered_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_114
#if defined(TEST_114)
#include <unordered_set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_115
#if defined(TEST_115)
#include <utility>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_116
#if defined(TEST_116)
#include <valarray>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_117
#if defined(TEST_117)
#include <variant>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_118
#if defined(TEST_118)
#include <vector>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_119
#if defined(TEST_119)
#include <version>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_120
#if defined(TEST_120) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <wchar.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_121
#if defined(TEST_121) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <wctype.h>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_122
#if defined(TEST_122) && __cplusplus >= 201103L
#include <experimental/algorithm>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_123
#if defined(TEST_123) && __cplusplus >= 201103L && !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_COROUTINES)
#include <experimental/coroutine>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_124
#if defined(TEST_124) && __cplusplus >= 201103L
#include <experimental/deque>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_125
#if defined(TEST_125) && __cplusplus >= 201103L
#include <experimental/forward_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_126
#if defined(TEST_126) && __cplusplus >= 201103L
#include <experimental/functional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_127
#if defined(TEST_127) && __cplusplus >= 201103L
#include <experimental/iterator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_128
#if defined(TEST_128) && __cplusplus >= 201103L
#include <experimental/list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_129
#if defined(TEST_129) && __cplusplus >= 201103L
#include <experimental/map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_130
#if defined(TEST_130) && __cplusplus >= 201103L
#include <experimental/memory_resource>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_131
#if defined(TEST_131) && __cplusplus >= 201103L
#include <experimental/propagate_const>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_132
#if defined(TEST_132) && !defined(_LIBCPP_HAS_NO_LOCALIZATION) && __cplusplus >= 201103L
#include <experimental/regex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_133
#if defined(TEST_133) && __cplusplus >= 201103L
#include <experimental/set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_134
#if defined(TEST_134) && __cplusplus >= 201103L
#include <experimental/simd>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_135
#if defined(TEST_135) && __cplusplus >= 201103L
#include <experimental/string>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_136
#if defined(TEST_136) && __cplusplus >= 201103L
#include <experimental/type_traits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_137
#if defined(TEST_137) && __cplusplus >= 201103L
#include <experimental/unordered_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_138
#if defined(TEST_138) && __cplusplus >= 201103L
#include <experimental/unordered_set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_139
#if defined(TEST_139) && __cplusplus >= 201103L
#include <experimental/utility>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_140
#if defined(TEST_140) && __cplusplus >= 201103L
#include <experimental/vector>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_141
#if defined(TEST_141)
#include <ext/hash_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_142
#if defined(TEST_142)
#include <ext/hash_set>
#endif
// GENERATED-MARKER
