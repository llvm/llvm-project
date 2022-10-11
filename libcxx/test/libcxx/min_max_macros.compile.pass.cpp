//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that headers are not tripped up by the surrounding code defining the
// min() and max() macros.

// The system-provided <uchar.h> seems to be broken on AIX
// XFAIL: LIBCXX-AIX-FIXME

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#    undef __DEPRECATED
#endif

#define TEST_MACROS() static_assert(min() == true && max() == true, "")
#define min() true
#define max() true

/*
BEGIN-SCRIPT

for header in public_headers:
  print("{}#{}include <{}>\nTEST_MACROS();{}".format(
    '#if ' + header_restrictions[header] + '\n' if header in header_restrictions else '',
    3 * ' ' if header in header_restrictions else '',
    header,
    '\n#endif' if header in header_restrictions else ''
  ))

END-SCRIPT
*/

// DO NOT MANUALLY EDIT ANYTHING BETWEEN THE MARKERS BELOW
// GENERATED-MARKER
#include <algorithm>
TEST_MACROS();
#include <any>
TEST_MACROS();
#include <array>
TEST_MACROS();
#include <atomic>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <barrier>
TEST_MACROS();
#endif
#include <bit>
TEST_MACROS();
#include <bitset>
TEST_MACROS();
#include <cassert>
TEST_MACROS();
#include <ccomplex>
TEST_MACROS();
#include <cctype>
TEST_MACROS();
#include <cerrno>
TEST_MACROS();
#include <cfenv>
TEST_MACROS();
#include <cfloat>
TEST_MACROS();
#include <charconv>
TEST_MACROS();
#include <chrono>
TEST_MACROS();
#include <cinttypes>
TEST_MACROS();
#include <ciso646>
TEST_MACROS();
#include <climits>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <clocale>
TEST_MACROS();
#endif
#include <cmath>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <codecvt>
TEST_MACROS();
#endif
#include <compare>
TEST_MACROS();
#include <complex>
TEST_MACROS();
#include <complex.h>
TEST_MACROS();
#include <concepts>
TEST_MACROS();
#include <condition_variable>
TEST_MACROS();
#include <coroutine>
TEST_MACROS();
#include <csetjmp>
TEST_MACROS();
#include <csignal>
TEST_MACROS();
#include <cstdarg>
TEST_MACROS();
#include <cstdbool>
TEST_MACROS();
#include <cstddef>
TEST_MACROS();
#include <cstdint>
TEST_MACROS();
#include <cstdio>
TEST_MACROS();
#include <cstdlib>
TEST_MACROS();
#include <cstring>
TEST_MACROS();
#include <ctgmath>
TEST_MACROS();
#include <ctime>
TEST_MACROS();
#include <ctype.h>
TEST_MACROS();
#include <cuchar>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <cwchar>
TEST_MACROS();
#endif
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <cwctype>
TEST_MACROS();
#endif
#include <deque>
TEST_MACROS();
#include <errno.h>
TEST_MACROS();
#include <exception>
TEST_MACROS();
#include <execution>
TEST_MACROS();
#include <fenv.h>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#   include <filesystem>
TEST_MACROS();
#endif
#include <float.h>
TEST_MACROS();
#include <format>
TEST_MACROS();
#include <forward_list>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <fstream>
TEST_MACROS();
#endif
#include <functional>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <future>
TEST_MACROS();
#endif
#include <initializer_list>
TEST_MACROS();
#include <inttypes.h>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <iomanip>
TEST_MACROS();
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <ios>
TEST_MACROS();
#endif
#include <iosfwd>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <iostream>
TEST_MACROS();
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <istream>
TEST_MACROS();
#endif
#include <iterator>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <latch>
TEST_MACROS();
#endif
#include <limits>
TEST_MACROS();
#include <limits.h>
TEST_MACROS();
#include <list>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <locale>
TEST_MACROS();
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <locale.h>
TEST_MACROS();
#endif
#include <map>
TEST_MACROS();
#include <math.h>
TEST_MACROS();
#include <memory>
TEST_MACROS();
#include <memory_resource>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <mutex>
TEST_MACROS();
#endif
#include <new>
TEST_MACROS();
#include <numbers>
TEST_MACROS();
#include <numeric>
TEST_MACROS();
#include <optional>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <ostream>
TEST_MACROS();
#endif
#include <queue>
TEST_MACROS();
#include <random>
TEST_MACROS();
#include <ranges>
TEST_MACROS();
#include <ratio>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <regex>
TEST_MACROS();
#endif
#include <scoped_allocator>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <semaphore>
TEST_MACROS();
#endif
#include <set>
TEST_MACROS();
#include <setjmp.h>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <shared_mutex>
TEST_MACROS();
#endif
#include <span>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <sstream>
TEST_MACROS();
#endif
#include <stack>
TEST_MACROS();
#if __cplusplus > 202002L && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <stdatomic.h>
TEST_MACROS();
#endif
#include <stdbool.h>
TEST_MACROS();
#include <stddef.h>
TEST_MACROS();
#include <stdexcept>
TEST_MACROS();
#include <stdint.h>
TEST_MACROS();
#include <stdio.h>
TEST_MACROS();
#include <stdlib.h>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <streambuf>
TEST_MACROS();
#endif
#include <string>
TEST_MACROS();
#include <string.h>
TEST_MACROS();
#include <string_view>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <strstream>
TEST_MACROS();
#endif
#include <system_error>
TEST_MACROS();
#include <tgmath.h>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <thread>
TEST_MACROS();
#endif
#include <tuple>
TEST_MACROS();
#include <type_traits>
TEST_MACROS();
#include <typeindex>
TEST_MACROS();
#include <typeinfo>
TEST_MACROS();
#include <uchar.h>
TEST_MACROS();
#include <unordered_map>
TEST_MACROS();
#include <unordered_set>
TEST_MACROS();
#include <utility>
TEST_MACROS();
#include <valarray>
TEST_MACROS();
#include <variant>
TEST_MACROS();
#include <vector>
TEST_MACROS();
#include <version>
TEST_MACROS();
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <wchar.h>
TEST_MACROS();
#endif
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <wctype.h>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/algorithm>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L && !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_COROUTINES)
#   include <experimental/coroutine>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/deque>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/forward_list>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/functional>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/iterator>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/list>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/map>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/memory_resource>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/propagate_const>
TEST_MACROS();
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION) && __cplusplus >= 201103L
#   include <experimental/regex>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/set>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/simd>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/string>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/type_traits>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/unordered_map>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/unordered_set>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/utility>
TEST_MACROS();
#endif
#if __cplusplus >= 201103L
#   include <experimental/vector>
TEST_MACROS();
#endif
#include <ext/hash_map>
TEST_MACROS();
#include <ext/hash_set>
TEST_MACROS();
// GENERATED-MARKER
