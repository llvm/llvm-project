// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that headers are not tripped up by the surrounding code defining the
// min() and max() macros.

// GCC 5 has incomplete support for C++17, so some headers fail when included.
// UNSUPPORTED: gcc-5 && c++17

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#undef __DEPRECATED
#endif

#define TEST_MACROS() static_assert(min() == true && max() == true, "")
#define min() true
#define max() true

// Top level headers
#include <algorithm>
TEST_MACROS();
#include <any>
TEST_MACROS();
#include <array>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <atomic>
TEST_MACROS();
#endif
#ifndef _LIBCPP_HAS_NO_THREADS
#include <barrier>
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
#include <cmath>
TEST_MACROS();
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
#include <cwchar>
TEST_MACROS();
#include <cwctype>
TEST_MACROS();
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
#include <filesystem>
TEST_MACROS();
#include <float.h>
TEST_MACROS();
#include <forward_list>
TEST_MACROS();
#include <functional>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <future>
TEST_MACROS();
#endif
#include <initializer_list>
TEST_MACROS();
#include <inttypes.h>
TEST_MACROS();
#include <iosfwd>
TEST_MACROS();
#include <iterator>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <latch>
TEST_MACROS();
#endif
#include <limits>
TEST_MACROS();
#include <limits.h>
TEST_MACROS();
#include <list>
TEST_MACROS();
#include <map>
TEST_MACROS();
#include <math.h>
TEST_MACROS();
#include <memory>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <mutex>
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
#include <queue>
TEST_MACROS();
#include <random>
TEST_MACROS();
#include <ratio>
TEST_MACROS();
#include <scoped_allocator>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <semaphore>
TEST_MACROS();
#endif
#include <set>
TEST_MACROS();
#include <setjmp.h>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <shared_mutex>
TEST_MACROS();
#endif
#include <span>
TEST_MACROS();
#include <stack>
TEST_MACROS();
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
#include <string>
TEST_MACROS();
#include <string.h>
TEST_MACROS();
#include <string_view>
TEST_MACROS();
#include <system_error>
TEST_MACROS();
#include <tgmath.h>
TEST_MACROS();
#ifndef _LIBCPP_HAS_NO_THREADS
#include <thread>
TEST_MACROS();
#endif
#include <tuple>
TEST_MACROS();
#include <typeindex>
TEST_MACROS();
#include <typeinfo>
TEST_MACROS();
#include <type_traits>
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
#include <wchar.h>
TEST_MACROS();
#include <wctype.h>
TEST_MACROS();

#ifndef _LIBCPP_HAS_NO_LOCALIZATION
#   include <clocale>
    TEST_MACROS();
#   include <codecvt>
    TEST_MACROS();
#   include <fstream>
    TEST_MACROS();
#   include <iomanip>
    TEST_MACROS();
#   include <ios>
    TEST_MACROS();
#   include <iostream>
    TEST_MACROS();
#   include <istream>
    TEST_MACROS();
#   include <locale>
    TEST_MACROS();
#   include <locale.h>
    TEST_MACROS();
#   include <ostream>
    TEST_MACROS();
#   include <regex>
    TEST_MACROS();
#   include <sstream>
    TEST_MACROS();
#   include <streambuf>
    TEST_MACROS();
#   include <strstream>
    TEST_MACROS();
#   if __cplusplus >= 201103L
#       include <experimental/regex>
        TEST_MACROS();
#   endif
#endif

// experimental headers
#if __cplusplus >= 201103L
#include <experimental/algorithm>
TEST_MACROS();
#if defined(__cpp_coroutines)
#include <experimental/coroutine>
TEST_MACROS();
#endif
#include <experimental/deque>
TEST_MACROS();
#include <experimental/filesystem>
TEST_MACROS();
#include <experimental/forward_list>
TEST_MACROS();
#include <experimental/functional>
TEST_MACROS();
#include <experimental/iterator>
TEST_MACROS();
#include <experimental/list>
TEST_MACROS();
#include <experimental/map>
TEST_MACROS();
#include <experimental/memory_resource>
TEST_MACROS();
#include <experimental/propagate_const>
TEST_MACROS();
#include <experimental/set>
TEST_MACROS();
#include <experimental/simd>
TEST_MACROS();
#include <experimental/string>
TEST_MACROS();
#include <experimental/type_traits>
TEST_MACROS();
#include <experimental/unordered_map>
TEST_MACROS();
#include <experimental/unordered_set>
TEST_MACROS();
#include <experimental/utility>
TEST_MACROS();
#include <experimental/vector>
TEST_MACROS();
#endif // __cplusplus >= 201103L

// extended headers
#include <ext/hash_map>
TEST_MACROS();
#include <ext/hash_set>
TEST_MACROS();
