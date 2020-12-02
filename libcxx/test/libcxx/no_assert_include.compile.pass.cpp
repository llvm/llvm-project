// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that none of the standard C++ headers implicitly include cassert or
// assert.h (because assert() is implemented as a macro).

// GCC 5 has incomplete support for C++17, so some headers fail when included.
// UNSUPPORTED: gcc-5 && c++17

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
#if defined(__DEPRECATED)
#undef __DEPRECATED
#endif

// Top level headers
#include <algorithm>
#include <any>
#include <array>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <atomic>
#endif
#ifndef _LIBCPP_HAS_NO_THREADS
#include <barrier>
#endif
#include <bit>
#include <bitset>
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
#include <cmath>
#include <compare>
#include <complex>
#include <complex.h>
#include <concepts>
#include <condition_variable>
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
#include <cwchar>
#include <cwctype>
#include <deque>
#include <errno.h>
#include <exception>
#include <execution>
#include <fenv.h>
#include <filesystem>
#include <float.h>
#include <forward_list>
#include <functional>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <future>
#endif
#include <initializer_list>
#include <inttypes.h>
#include <iosfwd>
#include <iterator>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <latch>
#endif
#include <limits>
#include <limits.h>
#include <list>
#include <map>
#include <math.h>
#include <memory>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <mutex>
#endif
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <ratio>
#include <scoped_allocator>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <semaphore>
#endif
#include <set>
#include <setjmp.h>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <shared_mutex>
#endif
#include <span>
#include <stack>
#include <stdbool.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <string_view>
#include <system_error>
#include <tgmath.h>
#ifndef _LIBCPP_HAS_NO_THREADS
#include <thread>
#endif
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <variant>
#include <vector>
#include <version>
#include <wchar.h>
#include <wctype.h>

#ifndef _LIBCPP_HAS_NO_LOCALIZATION
#   include <clocale>
#   include <codecvt>
#   include <fstream>
#   include <iomanip>
#   include <ios>
#   include <iostream>
#   include <istream>
#   include <locale>
#   include <locale.h>
#   include <ostream>
#   include <regex>
#   include <sstream>
#   include <streambuf>
#   include <strstream>
#   if __cplusplus >= 201103L
#       include <experimental/regex>
#   endif
#endif

// experimental headers
#if __cplusplus >= 201103L
#include <experimental/algorithm>
#if defined(__cpp_coroutines)
#include <experimental/coroutine>
#endif
#include <experimental/deque>
#include <experimental/filesystem>
#include <experimental/forward_list>
#include <experimental/functional>
#include <experimental/iterator>
#include <experimental/list>
#include <experimental/map>
#include <experimental/memory_resource>
#include <experimental/propagate_const>
#include <experimental/set>
#include <experimental/simd>
#include <experimental/string>
#include <experimental/type_traits>
#include <experimental/unordered_map>
#include <experimental/unordered_set>
#include <experimental/utility>
#include <experimental/vector>
#endif // __cplusplus >= 201103L

// extended headers
#include <ext/hash_map>
#include <ext/hash_set>

#ifdef assert
#error "Do not include cassert or assert.h in standard header files"
#endif
