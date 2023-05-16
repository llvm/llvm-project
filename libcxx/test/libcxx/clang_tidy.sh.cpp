//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-clang-tidy

// The GCC compiler flags are not always compatible with clang-tidy.
// UNSUPPORTED: gcc

// TODO: run clang-tidy with modules enabled once they are supported
// RUN: %{clang-tidy} %s --warnings-as-errors=* -header-filter=.* --checks='-*,libcpp-*' --load=%{test-tools}/clang_tidy_checks/libcxx-tidy.plugin -- %{compile_flags} -fno-modules
// RUN: %{clang-tidy} %s --warnings-as-errors=* -header-filter=.* --config-file=%S/../../.clang-tidy -- -Wweak-vtables %{compile_flags} -fno-modules

/*
BEGIN-SCRIPT

for header in public_headers:
  print("{}#{}include <{}>{}".format(
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
#include <any>
#include <array>
#include <atomic>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <barrier>
#endif
#include <bit>
#include <bitset>
#include <cassert>
#include <ccomplex>
#include <cctype>
#include <cerrno>
#include <cfenv>
#include <cfloat>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <ciso646>
#include <climits>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <clocale>
#endif
#include <cmath>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <codecvt>
#endif
#include <compare>
#include <complex>
#include <complex.h>
#include <concepts>
#include <condition_variable>
#if (defined(__cpp_impl_coroutine) && __cpp_impl_coroutine >= 201902L) || (defined(__cpp_coroutines) && __cpp_coroutines >= 201703L)
#   include <coroutine>
#endif
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctgmath>
#include <ctime>
#include <ctype.h>
#include <cuchar>
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <cwchar>
#endif
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <cwctype>
#endif
#include <deque>
#include <errno.h>
#include <exception>
#include <execution>
#include <expected>
#include <fenv.h>
#if !defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#   include <filesystem>
#endif
#include <float.h>
#include <format>
#include <forward_list>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION) && !defined(_LIBCPP_HAS_NO_FSTREAM)
#   include <fstream>
#endif
#include <functional>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <future>
#endif
#include <initializer_list>
#include <inttypes.h>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <iomanip>
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <ios>
#endif
#include <iosfwd>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <iostream>
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <istream>
#endif
#include <iterator>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <latch>
#endif
#include <limits>
#include <limits.h>
#include <list>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <locale>
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <locale.h>
#endif
#include <map>
#include <math.h>
#include <memory>
#include <memory_resource>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <mutex>
#endif
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <ostream>
#endif
#include <queue>
#include <random>
#include <ranges>
#include <ratio>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <regex>
#endif
#include <scoped_allocator>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <semaphore>
#endif
#include <set>
#include <setjmp.h>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <shared_mutex>
#endif
#include <source_location>
#include <span>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <sstream>
#endif
#include <stack>
#if __cplusplus > 202002L && !defined(_LIBCPP_HAS_NO_THREADS)
#   include <stdatomic.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <streambuf>
#endif
#include <string>
#include <string.h>
#include <string_view>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#   include <strstream>
#endif
#include <system_error>
#include <tgmath.h>
#if !defined(_LIBCPP_HAS_NO_THREADS)
#   include <thread>
#endif
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <uchar.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <variant>
#include <vector>
#include <version>
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <wchar.h>
#endif
#if !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#   include <wctype.h>
#endif
#if __cplusplus >= 201103L
#   include <experimental/deque>
#endif
#if __cplusplus >= 201103L
#   include <experimental/forward_list>
#endif
#if __cplusplus >= 201103L
#   include <experimental/iterator>
#endif
#if __cplusplus >= 201103L
#   include <experimental/list>
#endif
#if __cplusplus >= 201103L
#   include <experimental/map>
#endif
#if __cplusplus >= 201103L
#   include <experimental/memory_resource>
#endif
#if __cplusplus >= 201103L
#   include <experimental/propagate_const>
#endif
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION) && __cplusplus >= 201103L
#   include <experimental/regex>
#endif
#if __cplusplus >= 201103L
#   include <experimental/set>
#endif
#if __cplusplus >= 201103L
#   include <experimental/simd>
#endif
#if __cplusplus >= 201103L
#   include <experimental/string>
#endif
#if __cplusplus >= 201103L
#   include <experimental/type_traits>
#endif
#if __cplusplus >= 201103L
#   include <experimental/unordered_map>
#endif
#if __cplusplus >= 201103L
#   include <experimental/unordered_set>
#endif
#if __cplusplus >= 201103L
#   include <experimental/utility>
#endif
#if __cplusplus >= 201103L
#   include <experimental/vector>
#endif
// GENERATED-MARKER
