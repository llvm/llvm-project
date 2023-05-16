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

// GCC doesn't support -fcxx-modules
// UNSUPPORTED: gcc

// The Windows headers don't appear to be compatible with modules
// UNSUPPORTED: windows
// UNSUPPORTED: buildhost=windows

// The Android headers don't appear to be compatible with modules yet
// XFAIL: LIBCXX-ANDROID-FIXME

#include <__config>

/*
BEGIN-SCRIPT

for i, header in enumerate(public_headers):
  print("// {}: echo '%{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_{} &' >> %t.sh".format('RUN', i))
  print("// {}: echo 'TEST_{}=$!' >> %t.sh".format('RUN', i))
  if i >= 16:
    print("// {}: echo \"wait $TEST_{}\" >> %t.sh".format('RUN', i - 16))
  if header in header_restrictions:
    print("#if defined(TEST_{}) && {}".format(i, header_restrictions[header]))
  else:
    print("#if defined(TEST_{})".format(i))
  print("#include <{}>".format(header))
  print("#endif")

for i in range(len(public_headers))[-16:]:
  print("// {}: echo \"wait $TEST_{}\" >> %t.sh".format('RUN', i))

print("// {}: bash %t.sh".format('RUN'))

END-SCRIPT
*/

// RUN: echo "" > %t.sh
// RUN: rm -rf %t
// RUN: mkdir %t

// DO NOT MANUALLY EDIT ANYTHING BETWEEN THE MARKERS BELOW
// GENERATED-MARKER
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_0 &' >> %t.sh
// RUN: echo 'TEST_0=$!' >> %t.sh
#if defined(TEST_0)
#include <algorithm>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_1 &' >> %t.sh
// RUN: echo 'TEST_1=$!' >> %t.sh
#if defined(TEST_1)
#include <any>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_2 &' >> %t.sh
// RUN: echo 'TEST_2=$!' >> %t.sh
#if defined(TEST_2)
#include <array>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_3 &' >> %t.sh
// RUN: echo 'TEST_3=$!' >> %t.sh
#if defined(TEST_3)
#include <atomic>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_4 &' >> %t.sh
// RUN: echo 'TEST_4=$!' >> %t.sh
#if defined(TEST_4) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <barrier>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_5 &' >> %t.sh
// RUN: echo 'TEST_5=$!' >> %t.sh
#if defined(TEST_5)
#include <bit>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_6 &' >> %t.sh
// RUN: echo 'TEST_6=$!' >> %t.sh
#if defined(TEST_6)
#include <bitset>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_7 &' >> %t.sh
// RUN: echo 'TEST_7=$!' >> %t.sh
#if defined(TEST_7)
#include <cassert>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_8 &' >> %t.sh
// RUN: echo 'TEST_8=$!' >> %t.sh
#if defined(TEST_8)
#include <ccomplex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_9 &' >> %t.sh
// RUN: echo 'TEST_9=$!' >> %t.sh
#if defined(TEST_9)
#include <cctype>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_10 &' >> %t.sh
// RUN: echo 'TEST_10=$!' >> %t.sh
#if defined(TEST_10)
#include <cerrno>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_11 &' >> %t.sh
// RUN: echo 'TEST_11=$!' >> %t.sh
#if defined(TEST_11)
#include <cfenv>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_12 &' >> %t.sh
// RUN: echo 'TEST_12=$!' >> %t.sh
#if defined(TEST_12)
#include <cfloat>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_13 &' >> %t.sh
// RUN: echo 'TEST_13=$!' >> %t.sh
#if defined(TEST_13)
#include <charconv>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_14 &' >> %t.sh
// RUN: echo 'TEST_14=$!' >> %t.sh
#if defined(TEST_14)
#include <chrono>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_15 &' >> %t.sh
// RUN: echo 'TEST_15=$!' >> %t.sh
#if defined(TEST_15)
#include <cinttypes>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_16 &' >> %t.sh
// RUN: echo 'TEST_16=$!' >> %t.sh
// RUN: echo "wait $TEST_0" >> %t.sh
#if defined(TEST_16)
#include <ciso646>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_17 &' >> %t.sh
// RUN: echo 'TEST_17=$!' >> %t.sh
// RUN: echo "wait $TEST_1" >> %t.sh
#if defined(TEST_17)
#include <climits>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_18 &' >> %t.sh
// RUN: echo 'TEST_18=$!' >> %t.sh
// RUN: echo "wait $TEST_2" >> %t.sh
#if defined(TEST_18) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <clocale>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_19 &' >> %t.sh
// RUN: echo 'TEST_19=$!' >> %t.sh
// RUN: echo "wait $TEST_3" >> %t.sh
#if defined(TEST_19)
#include <cmath>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_20 &' >> %t.sh
// RUN: echo 'TEST_20=$!' >> %t.sh
// RUN: echo "wait $TEST_4" >> %t.sh
#if defined(TEST_20) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <codecvt>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_21 &' >> %t.sh
// RUN: echo 'TEST_21=$!' >> %t.sh
// RUN: echo "wait $TEST_5" >> %t.sh
#if defined(TEST_21)
#include <compare>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_22 &' >> %t.sh
// RUN: echo 'TEST_22=$!' >> %t.sh
// RUN: echo "wait $TEST_6" >> %t.sh
#if defined(TEST_22)
#include <complex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_23 &' >> %t.sh
// RUN: echo 'TEST_23=$!' >> %t.sh
// RUN: echo "wait $TEST_7" >> %t.sh
#if defined(TEST_23)
#include <complex.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_24 &' >> %t.sh
// RUN: echo 'TEST_24=$!' >> %t.sh
// RUN: echo "wait $TEST_8" >> %t.sh
#if defined(TEST_24)
#include <concepts>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_25 &' >> %t.sh
// RUN: echo 'TEST_25=$!' >> %t.sh
// RUN: echo "wait $TEST_9" >> %t.sh
#if defined(TEST_25)
#include <condition_variable>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_26 &' >> %t.sh
// RUN: echo 'TEST_26=$!' >> %t.sh
// RUN: echo "wait $TEST_10" >> %t.sh
#if defined(TEST_26) && (defined(__cpp_impl_coroutine) && __cpp_impl_coroutine >= 201902L) || (defined(__cpp_coroutines) && __cpp_coroutines >= 201703L)
#include <coroutine>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_27 &' >> %t.sh
// RUN: echo 'TEST_27=$!' >> %t.sh
// RUN: echo "wait $TEST_11" >> %t.sh
#if defined(TEST_27)
#include <csetjmp>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_28 &' >> %t.sh
// RUN: echo 'TEST_28=$!' >> %t.sh
// RUN: echo "wait $TEST_12" >> %t.sh
#if defined(TEST_28)
#include <csignal>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_29 &' >> %t.sh
// RUN: echo 'TEST_29=$!' >> %t.sh
// RUN: echo "wait $TEST_13" >> %t.sh
#if defined(TEST_29)
#include <cstdarg>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_30 &' >> %t.sh
// RUN: echo 'TEST_30=$!' >> %t.sh
// RUN: echo "wait $TEST_14" >> %t.sh
#if defined(TEST_30)
#include <cstdbool>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_31 &' >> %t.sh
// RUN: echo 'TEST_31=$!' >> %t.sh
// RUN: echo "wait $TEST_15" >> %t.sh
#if defined(TEST_31)
#include <cstddef>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_32 &' >> %t.sh
// RUN: echo 'TEST_32=$!' >> %t.sh
// RUN: echo "wait $TEST_16" >> %t.sh
#if defined(TEST_32)
#include <cstdint>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_33 &' >> %t.sh
// RUN: echo 'TEST_33=$!' >> %t.sh
// RUN: echo "wait $TEST_17" >> %t.sh
#if defined(TEST_33)
#include <cstdio>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_34 &' >> %t.sh
// RUN: echo 'TEST_34=$!' >> %t.sh
// RUN: echo "wait $TEST_18" >> %t.sh
#if defined(TEST_34)
#include <cstdlib>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_35 &' >> %t.sh
// RUN: echo 'TEST_35=$!' >> %t.sh
// RUN: echo "wait $TEST_19" >> %t.sh
#if defined(TEST_35)
#include <cstring>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_36 &' >> %t.sh
// RUN: echo 'TEST_36=$!' >> %t.sh
// RUN: echo "wait $TEST_20" >> %t.sh
#if defined(TEST_36)
#include <ctgmath>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_37 &' >> %t.sh
// RUN: echo 'TEST_37=$!' >> %t.sh
// RUN: echo "wait $TEST_21" >> %t.sh
#if defined(TEST_37)
#include <ctime>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_38 &' >> %t.sh
// RUN: echo 'TEST_38=$!' >> %t.sh
// RUN: echo "wait $TEST_22" >> %t.sh
#if defined(TEST_38)
#include <ctype.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_39 &' >> %t.sh
// RUN: echo 'TEST_39=$!' >> %t.sh
// RUN: echo "wait $TEST_23" >> %t.sh
#if defined(TEST_39)
#include <cuchar>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_40 &' >> %t.sh
// RUN: echo 'TEST_40=$!' >> %t.sh
// RUN: echo "wait $TEST_24" >> %t.sh
#if defined(TEST_40) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <cwchar>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_41 &' >> %t.sh
// RUN: echo 'TEST_41=$!' >> %t.sh
// RUN: echo "wait $TEST_25" >> %t.sh
#if defined(TEST_41) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <cwctype>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_42 &' >> %t.sh
// RUN: echo 'TEST_42=$!' >> %t.sh
// RUN: echo "wait $TEST_26" >> %t.sh
#if defined(TEST_42)
#include <deque>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_43 &' >> %t.sh
// RUN: echo 'TEST_43=$!' >> %t.sh
// RUN: echo "wait $TEST_27" >> %t.sh
#if defined(TEST_43)
#include <errno.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_44 &' >> %t.sh
// RUN: echo 'TEST_44=$!' >> %t.sh
// RUN: echo "wait $TEST_28" >> %t.sh
#if defined(TEST_44)
#include <exception>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_45 &' >> %t.sh
// RUN: echo 'TEST_45=$!' >> %t.sh
// RUN: echo "wait $TEST_29" >> %t.sh
#if defined(TEST_45)
#include <execution>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_46 &' >> %t.sh
// RUN: echo 'TEST_46=$!' >> %t.sh
// RUN: echo "wait $TEST_30" >> %t.sh
#if defined(TEST_46)
#include <expected>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_47 &' >> %t.sh
// RUN: echo 'TEST_47=$!' >> %t.sh
// RUN: echo "wait $TEST_31" >> %t.sh
#if defined(TEST_47)
#include <fenv.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_48 &' >> %t.sh
// RUN: echo 'TEST_48=$!' >> %t.sh
// RUN: echo "wait $TEST_32" >> %t.sh
#if defined(TEST_48) && !defined(_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
#include <filesystem>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_49 &' >> %t.sh
// RUN: echo 'TEST_49=$!' >> %t.sh
// RUN: echo "wait $TEST_33" >> %t.sh
#if defined(TEST_49)
#include <float.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_50 &' >> %t.sh
// RUN: echo 'TEST_50=$!' >> %t.sh
// RUN: echo "wait $TEST_34" >> %t.sh
#if defined(TEST_50)
#include <format>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_51 &' >> %t.sh
// RUN: echo 'TEST_51=$!' >> %t.sh
// RUN: echo "wait $TEST_35" >> %t.sh
#if defined(TEST_51)
#include <forward_list>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_52 &' >> %t.sh
// RUN: echo 'TEST_52=$!' >> %t.sh
// RUN: echo "wait $TEST_36" >> %t.sh
#if defined(TEST_52) && !defined(_LIBCPP_HAS_NO_LOCALIZATION) && !defined(_LIBCPP_HAS_NO_FSTREAM)
#include <fstream>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_53 &' >> %t.sh
// RUN: echo 'TEST_53=$!' >> %t.sh
// RUN: echo "wait $TEST_37" >> %t.sh
#if defined(TEST_53)
#include <functional>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_54 &' >> %t.sh
// RUN: echo 'TEST_54=$!' >> %t.sh
// RUN: echo "wait $TEST_38" >> %t.sh
#if defined(TEST_54) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <future>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_55 &' >> %t.sh
// RUN: echo 'TEST_55=$!' >> %t.sh
// RUN: echo "wait $TEST_39" >> %t.sh
#if defined(TEST_55)
#include <initializer_list>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_56 &' >> %t.sh
// RUN: echo 'TEST_56=$!' >> %t.sh
// RUN: echo "wait $TEST_40" >> %t.sh
#if defined(TEST_56)
#include <inttypes.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_57 &' >> %t.sh
// RUN: echo 'TEST_57=$!' >> %t.sh
// RUN: echo "wait $TEST_41" >> %t.sh
#if defined(TEST_57) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <iomanip>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_58 &' >> %t.sh
// RUN: echo 'TEST_58=$!' >> %t.sh
// RUN: echo "wait $TEST_42" >> %t.sh
#if defined(TEST_58) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <ios>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_59 &' >> %t.sh
// RUN: echo 'TEST_59=$!' >> %t.sh
// RUN: echo "wait $TEST_43" >> %t.sh
#if defined(TEST_59)
#include <iosfwd>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_60 &' >> %t.sh
// RUN: echo 'TEST_60=$!' >> %t.sh
// RUN: echo "wait $TEST_44" >> %t.sh
#if defined(TEST_60) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <iostream>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_61 &' >> %t.sh
// RUN: echo 'TEST_61=$!' >> %t.sh
// RUN: echo "wait $TEST_45" >> %t.sh
#if defined(TEST_61) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <istream>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_62 &' >> %t.sh
// RUN: echo 'TEST_62=$!' >> %t.sh
// RUN: echo "wait $TEST_46" >> %t.sh
#if defined(TEST_62)
#include <iterator>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_63 &' >> %t.sh
// RUN: echo 'TEST_63=$!' >> %t.sh
// RUN: echo "wait $TEST_47" >> %t.sh
#if defined(TEST_63) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <latch>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_64 &' >> %t.sh
// RUN: echo 'TEST_64=$!' >> %t.sh
// RUN: echo "wait $TEST_48" >> %t.sh
#if defined(TEST_64)
#include <limits>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_65 &' >> %t.sh
// RUN: echo 'TEST_65=$!' >> %t.sh
// RUN: echo "wait $TEST_49" >> %t.sh
#if defined(TEST_65)
#include <limits.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_66 &' >> %t.sh
// RUN: echo 'TEST_66=$!' >> %t.sh
// RUN: echo "wait $TEST_50" >> %t.sh
#if defined(TEST_66)
#include <list>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_67 &' >> %t.sh
// RUN: echo 'TEST_67=$!' >> %t.sh
// RUN: echo "wait $TEST_51" >> %t.sh
#if defined(TEST_67) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <locale>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_68 &' >> %t.sh
// RUN: echo 'TEST_68=$!' >> %t.sh
// RUN: echo "wait $TEST_52" >> %t.sh
#if defined(TEST_68) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <locale.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_69 &' >> %t.sh
// RUN: echo 'TEST_69=$!' >> %t.sh
// RUN: echo "wait $TEST_53" >> %t.sh
#if defined(TEST_69)
#include <map>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_70 &' >> %t.sh
// RUN: echo 'TEST_70=$!' >> %t.sh
// RUN: echo "wait $TEST_54" >> %t.sh
#if defined(TEST_70)
#include <math.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_71 &' >> %t.sh
// RUN: echo 'TEST_71=$!' >> %t.sh
// RUN: echo "wait $TEST_55" >> %t.sh
#if defined(TEST_71)
#include <memory>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_72 &' >> %t.sh
// RUN: echo 'TEST_72=$!' >> %t.sh
// RUN: echo "wait $TEST_56" >> %t.sh
#if defined(TEST_72)
#include <memory_resource>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_73 &' >> %t.sh
// RUN: echo 'TEST_73=$!' >> %t.sh
// RUN: echo "wait $TEST_57" >> %t.sh
#if defined(TEST_73) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <mutex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_74 &' >> %t.sh
// RUN: echo 'TEST_74=$!' >> %t.sh
// RUN: echo "wait $TEST_58" >> %t.sh
#if defined(TEST_74)
#include <new>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_75 &' >> %t.sh
// RUN: echo 'TEST_75=$!' >> %t.sh
// RUN: echo "wait $TEST_59" >> %t.sh
#if defined(TEST_75)
#include <numbers>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_76 &' >> %t.sh
// RUN: echo 'TEST_76=$!' >> %t.sh
// RUN: echo "wait $TEST_60" >> %t.sh
#if defined(TEST_76)
#include <numeric>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_77 &' >> %t.sh
// RUN: echo 'TEST_77=$!' >> %t.sh
// RUN: echo "wait $TEST_61" >> %t.sh
#if defined(TEST_77)
#include <optional>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_78 &' >> %t.sh
// RUN: echo 'TEST_78=$!' >> %t.sh
// RUN: echo "wait $TEST_62" >> %t.sh
#if defined(TEST_78) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <ostream>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_79 &' >> %t.sh
// RUN: echo 'TEST_79=$!' >> %t.sh
// RUN: echo "wait $TEST_63" >> %t.sh
#if defined(TEST_79)
#include <queue>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_80 &' >> %t.sh
// RUN: echo 'TEST_80=$!' >> %t.sh
// RUN: echo "wait $TEST_64" >> %t.sh
#if defined(TEST_80)
#include <random>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_81 &' >> %t.sh
// RUN: echo 'TEST_81=$!' >> %t.sh
// RUN: echo "wait $TEST_65" >> %t.sh
#if defined(TEST_81)
#include <ranges>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_82 &' >> %t.sh
// RUN: echo 'TEST_82=$!' >> %t.sh
// RUN: echo "wait $TEST_66" >> %t.sh
#if defined(TEST_82)
#include <ratio>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_83 &' >> %t.sh
// RUN: echo 'TEST_83=$!' >> %t.sh
// RUN: echo "wait $TEST_67" >> %t.sh
#if defined(TEST_83) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <regex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_84 &' >> %t.sh
// RUN: echo 'TEST_84=$!' >> %t.sh
// RUN: echo "wait $TEST_68" >> %t.sh
#if defined(TEST_84)
#include <scoped_allocator>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_85 &' >> %t.sh
// RUN: echo 'TEST_85=$!' >> %t.sh
// RUN: echo "wait $TEST_69" >> %t.sh
#if defined(TEST_85) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <semaphore>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_86 &' >> %t.sh
// RUN: echo 'TEST_86=$!' >> %t.sh
// RUN: echo "wait $TEST_70" >> %t.sh
#if defined(TEST_86)
#include <set>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_87 &' >> %t.sh
// RUN: echo 'TEST_87=$!' >> %t.sh
// RUN: echo "wait $TEST_71" >> %t.sh
#if defined(TEST_87)
#include <setjmp.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_88 &' >> %t.sh
// RUN: echo 'TEST_88=$!' >> %t.sh
// RUN: echo "wait $TEST_72" >> %t.sh
#if defined(TEST_88) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <shared_mutex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_89 &' >> %t.sh
// RUN: echo 'TEST_89=$!' >> %t.sh
// RUN: echo "wait $TEST_73" >> %t.sh
#if defined(TEST_89)
#include <source_location>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_90 &' >> %t.sh
// RUN: echo 'TEST_90=$!' >> %t.sh
// RUN: echo "wait $TEST_74" >> %t.sh
#if defined(TEST_90)
#include <span>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_91 &' >> %t.sh
// RUN: echo 'TEST_91=$!' >> %t.sh
// RUN: echo "wait $TEST_75" >> %t.sh
#if defined(TEST_91) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <sstream>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_92 &' >> %t.sh
// RUN: echo 'TEST_92=$!' >> %t.sh
// RUN: echo "wait $TEST_76" >> %t.sh
#if defined(TEST_92)
#include <stack>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_93 &' >> %t.sh
// RUN: echo 'TEST_93=$!' >> %t.sh
// RUN: echo "wait $TEST_77" >> %t.sh
#if defined(TEST_93) && __cplusplus > 202002L && !defined(_LIBCPP_HAS_NO_THREADS)
#include <stdatomic.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_94 &' >> %t.sh
// RUN: echo 'TEST_94=$!' >> %t.sh
// RUN: echo "wait $TEST_78" >> %t.sh
#if defined(TEST_94)
#include <stdbool.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_95 &' >> %t.sh
// RUN: echo 'TEST_95=$!' >> %t.sh
// RUN: echo "wait $TEST_79" >> %t.sh
#if defined(TEST_95)
#include <stddef.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_96 &' >> %t.sh
// RUN: echo 'TEST_96=$!' >> %t.sh
// RUN: echo "wait $TEST_80" >> %t.sh
#if defined(TEST_96)
#include <stdexcept>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_97 &' >> %t.sh
// RUN: echo 'TEST_97=$!' >> %t.sh
// RUN: echo "wait $TEST_81" >> %t.sh
#if defined(TEST_97)
#include <stdint.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_98 &' >> %t.sh
// RUN: echo 'TEST_98=$!' >> %t.sh
// RUN: echo "wait $TEST_82" >> %t.sh
#if defined(TEST_98)
#include <stdio.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_99 &' >> %t.sh
// RUN: echo 'TEST_99=$!' >> %t.sh
// RUN: echo "wait $TEST_83" >> %t.sh
#if defined(TEST_99)
#include <stdlib.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_100 &' >> %t.sh
// RUN: echo 'TEST_100=$!' >> %t.sh
// RUN: echo "wait $TEST_84" >> %t.sh
#if defined(TEST_100) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <streambuf>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_101 &' >> %t.sh
// RUN: echo 'TEST_101=$!' >> %t.sh
// RUN: echo "wait $TEST_85" >> %t.sh
#if defined(TEST_101)
#include <string>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_102 &' >> %t.sh
// RUN: echo 'TEST_102=$!' >> %t.sh
// RUN: echo "wait $TEST_86" >> %t.sh
#if defined(TEST_102)
#include <string.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_103 &' >> %t.sh
// RUN: echo 'TEST_103=$!' >> %t.sh
// RUN: echo "wait $TEST_87" >> %t.sh
#if defined(TEST_103)
#include <string_view>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_104 &' >> %t.sh
// RUN: echo 'TEST_104=$!' >> %t.sh
// RUN: echo "wait $TEST_88" >> %t.sh
#if defined(TEST_104) && !defined(_LIBCPP_HAS_NO_LOCALIZATION)
#include <strstream>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_105 &' >> %t.sh
// RUN: echo 'TEST_105=$!' >> %t.sh
// RUN: echo "wait $TEST_89" >> %t.sh
#if defined(TEST_105)
#include <system_error>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_106 &' >> %t.sh
// RUN: echo 'TEST_106=$!' >> %t.sh
// RUN: echo "wait $TEST_90" >> %t.sh
#if defined(TEST_106)
#include <tgmath.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_107 &' >> %t.sh
// RUN: echo 'TEST_107=$!' >> %t.sh
// RUN: echo "wait $TEST_91" >> %t.sh
#if defined(TEST_107) && !defined(_LIBCPP_HAS_NO_THREADS)
#include <thread>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_108 &' >> %t.sh
// RUN: echo 'TEST_108=$!' >> %t.sh
// RUN: echo "wait $TEST_92" >> %t.sh
#if defined(TEST_108)
#include <tuple>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_109 &' >> %t.sh
// RUN: echo 'TEST_109=$!' >> %t.sh
// RUN: echo "wait $TEST_93" >> %t.sh
#if defined(TEST_109)
#include <type_traits>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_110 &' >> %t.sh
// RUN: echo 'TEST_110=$!' >> %t.sh
// RUN: echo "wait $TEST_94" >> %t.sh
#if defined(TEST_110)
#include <typeindex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_111 &' >> %t.sh
// RUN: echo 'TEST_111=$!' >> %t.sh
// RUN: echo "wait $TEST_95" >> %t.sh
#if defined(TEST_111)
#include <typeinfo>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_112 &' >> %t.sh
// RUN: echo 'TEST_112=$!' >> %t.sh
// RUN: echo "wait $TEST_96" >> %t.sh
#if defined(TEST_112)
#include <uchar.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_113 &' >> %t.sh
// RUN: echo 'TEST_113=$!' >> %t.sh
// RUN: echo "wait $TEST_97" >> %t.sh
#if defined(TEST_113)
#include <unordered_map>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_114 &' >> %t.sh
// RUN: echo 'TEST_114=$!' >> %t.sh
// RUN: echo "wait $TEST_98" >> %t.sh
#if defined(TEST_114)
#include <unordered_set>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_115 &' >> %t.sh
// RUN: echo 'TEST_115=$!' >> %t.sh
// RUN: echo "wait $TEST_99" >> %t.sh
#if defined(TEST_115)
#include <utility>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_116 &' >> %t.sh
// RUN: echo 'TEST_116=$!' >> %t.sh
// RUN: echo "wait $TEST_100" >> %t.sh
#if defined(TEST_116)
#include <valarray>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_117 &' >> %t.sh
// RUN: echo 'TEST_117=$!' >> %t.sh
// RUN: echo "wait $TEST_101" >> %t.sh
#if defined(TEST_117)
#include <variant>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_118 &' >> %t.sh
// RUN: echo 'TEST_118=$!' >> %t.sh
// RUN: echo "wait $TEST_102" >> %t.sh
#if defined(TEST_118)
#include <vector>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_119 &' >> %t.sh
// RUN: echo 'TEST_119=$!' >> %t.sh
// RUN: echo "wait $TEST_103" >> %t.sh
#if defined(TEST_119)
#include <version>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_120 &' >> %t.sh
// RUN: echo 'TEST_120=$!' >> %t.sh
// RUN: echo "wait $TEST_104" >> %t.sh
#if defined(TEST_120) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <wchar.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_121 &' >> %t.sh
// RUN: echo 'TEST_121=$!' >> %t.sh
// RUN: echo "wait $TEST_105" >> %t.sh
#if defined(TEST_121) && !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS)
#include <wctype.h>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_122 &' >> %t.sh
// RUN: echo 'TEST_122=$!' >> %t.sh
// RUN: echo "wait $TEST_106" >> %t.sh
#if defined(TEST_122) && __cplusplus >= 201103L
#include <experimental/deque>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_123 &' >> %t.sh
// RUN: echo 'TEST_123=$!' >> %t.sh
// RUN: echo "wait $TEST_107" >> %t.sh
#if defined(TEST_123) && __cplusplus >= 201103L
#include <experimental/forward_list>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_124 &' >> %t.sh
// RUN: echo 'TEST_124=$!' >> %t.sh
// RUN: echo "wait $TEST_108" >> %t.sh
#if defined(TEST_124) && __cplusplus >= 201103L
#include <experimental/iterator>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_125 &' >> %t.sh
// RUN: echo 'TEST_125=$!' >> %t.sh
// RUN: echo "wait $TEST_109" >> %t.sh
#if defined(TEST_125) && __cplusplus >= 201103L
#include <experimental/list>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_126 &' >> %t.sh
// RUN: echo 'TEST_126=$!' >> %t.sh
// RUN: echo "wait $TEST_110" >> %t.sh
#if defined(TEST_126) && __cplusplus >= 201103L
#include <experimental/map>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_127 &' >> %t.sh
// RUN: echo 'TEST_127=$!' >> %t.sh
// RUN: echo "wait $TEST_111" >> %t.sh
#if defined(TEST_127) && __cplusplus >= 201103L
#include <experimental/memory_resource>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_128 &' >> %t.sh
// RUN: echo 'TEST_128=$!' >> %t.sh
// RUN: echo "wait $TEST_112" >> %t.sh
#if defined(TEST_128) && __cplusplus >= 201103L
#include <experimental/propagate_const>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_129 &' >> %t.sh
// RUN: echo 'TEST_129=$!' >> %t.sh
// RUN: echo "wait $TEST_113" >> %t.sh
#if defined(TEST_129) && !defined(_LIBCPP_HAS_NO_LOCALIZATION) && __cplusplus >= 201103L
#include <experimental/regex>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_130 &' >> %t.sh
// RUN: echo 'TEST_130=$!' >> %t.sh
// RUN: echo "wait $TEST_114" >> %t.sh
#if defined(TEST_130) && __cplusplus >= 201103L
#include <experimental/set>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_131 &' >> %t.sh
// RUN: echo 'TEST_131=$!' >> %t.sh
// RUN: echo "wait $TEST_115" >> %t.sh
#if defined(TEST_131) && __cplusplus >= 201103L
#include <experimental/simd>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_132 &' >> %t.sh
// RUN: echo 'TEST_132=$!' >> %t.sh
// RUN: echo "wait $TEST_116" >> %t.sh
#if defined(TEST_132) && __cplusplus >= 201103L
#include <experimental/string>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_133 &' >> %t.sh
// RUN: echo 'TEST_133=$!' >> %t.sh
// RUN: echo "wait $TEST_117" >> %t.sh
#if defined(TEST_133) && __cplusplus >= 201103L
#include <experimental/type_traits>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_134 &' >> %t.sh
// RUN: echo 'TEST_134=$!' >> %t.sh
// RUN: echo "wait $TEST_118" >> %t.sh
#if defined(TEST_134) && __cplusplus >= 201103L
#include <experimental/unordered_map>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_135 &' >> %t.sh
// RUN: echo 'TEST_135=$!' >> %t.sh
// RUN: echo "wait $TEST_119" >> %t.sh
#if defined(TEST_135) && __cplusplus >= 201103L
#include <experimental/unordered_set>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_136 &' >> %t.sh
// RUN: echo 'TEST_136=$!' >> %t.sh
// RUN: echo "wait $TEST_120" >> %t.sh
#if defined(TEST_136) && __cplusplus >= 201103L
#include <experimental/utility>
#endif
// RUN: echo '%{cxx} %s %{flags} %{compile_flags} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only -DTEST_137 &' >> %t.sh
// RUN: echo 'TEST_137=$!' >> %t.sh
// RUN: echo "wait $TEST_121" >> %t.sh
#if defined(TEST_137) && __cplusplus >= 201103L
#include <experimental/vector>
#endif
// RUN: echo "wait $TEST_122" >> %t.sh
// RUN: echo "wait $TEST_123" >> %t.sh
// RUN: echo "wait $TEST_124" >> %t.sh
// RUN: echo "wait $TEST_125" >> %t.sh
// RUN: echo "wait $TEST_126" >> %t.sh
// RUN: echo "wait $TEST_127" >> %t.sh
// RUN: echo "wait $TEST_128" >> %t.sh
// RUN: echo "wait $TEST_129" >> %t.sh
// RUN: echo "wait $TEST_130" >> %t.sh
// RUN: echo "wait $TEST_131" >> %t.sh
// RUN: echo "wait $TEST_132" >> %t.sh
// RUN: echo "wait $TEST_133" >> %t.sh
// RUN: echo "wait $TEST_134" >> %t.sh
// RUN: echo "wait $TEST_135" >> %t.sh
// RUN: echo "wait $TEST_136" >> %t.sh
// RUN: echo "wait $TEST_137" >> %t.sh
// RUN: bash %t.sh
// GENERATED-MARKER
