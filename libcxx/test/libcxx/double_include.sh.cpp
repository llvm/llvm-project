//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we can include each header in two TU's and link them together.

// RUN: %{cxx} -c %s -o %t.first.o %{flags} %{compile_flags}
// RUN: %{cxx} -c %s -o %t.second.o -DWITH_MAIN %{flags} %{compile_flags}
// RUN: %{cxx} -o %t.exe %t.first.o %t.second.o %{flags} %{link_flags}
// RUN: %{run}

// The system-provided <uchar.h> seems to be broken on AIX
// XFAIL: LIBCXX-AIX-FIXME

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#    undef __DEPRECATED
#endif

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
#include <coroutine>
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
#include <fenv.h>
#if !defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#   include <filesystem>
#endif
#include <float.h>
#include <format>
#include <forward_list>
#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
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
#   include <experimental/algorithm>
#endif
#if __cplusplus >= 201103L && !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_COROUTINES)
#   include <experimental/coroutine>
#endif
#if __cplusplus >= 201103L
#   include <experimental/deque>
#endif
#if __cplusplus >= 201103L
#   include <experimental/forward_list>
#endif
#if __cplusplus >= 201103L
#   include <experimental/functional>
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
#include <ext/hash_map>
#include <ext/hash_set>
// GENERATED-MARKER

#if defined(WITH_MAIN)
int main(int, char**) { return 0; }
#endif
