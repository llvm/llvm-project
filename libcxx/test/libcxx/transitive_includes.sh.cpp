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

// This test doesn't work on AIX, but it should. Needs investigation.
// XFAIL: buildhost=aix

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

if regenerate_expected_results:
  print(f"// {RUN}: rm -rf %S/transitive_includes/%{{cxx_std}}")
  print(f"// {RUN}: mkdir %S/transitive_includes/%{{cxx_std}}")

for i, header in enumerate(public_headers):
  if header.endswith('.h'): # Skip C compatibility headers
    continue

  normalized_header = re.sub('/', '_', header)
  trace_includes = "%{{cxx}} %s %{{flags}} %{{compile_flags}} --trace-includes -fsyntax-only -DTEST_{} 2>&1".format(i)

  if regenerate_expected_results:
    print(f"// {RUN}: {trace_includes} | %{{python}} %S/transitive_includes.sanitize.py > %S/transitive_includes/%{{cxx_std}}/expected.{normalized_header}")
  else:
    print(f"// {RUN}: {trace_includes} | %{{python}} %S/transitive_includes.sanitize.py > %t.actual.{normalized_header}")
    print(f"// {RUN}: diff -w %S/transitive_includes/%{{cxx_std}}/expected.{normalized_header} %t.actual.{normalized_header}")

  print(f"#if defined(TEST_{i})")
  print(f"#include <{header}>")
  print("#endif")

END-SCRIPT
*/

// DO NOT MANUALLY EDIT ANYTHING BETWEEN THE MARKERS BELOW
// GENERATED-MARKER
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_0 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.algorithm
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.algorithm %t.actual.algorithm
#if defined(TEST_0)
#include <algorithm>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_1 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.any
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.any %t.actual.any
#if defined(TEST_1)
#include <any>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_2 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.array
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.array %t.actual.array
#if defined(TEST_2)
#include <array>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_3 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.atomic
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.atomic %t.actual.atomic
#if defined(TEST_3)
#include <atomic>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_4 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.barrier
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.barrier %t.actual.barrier
#if defined(TEST_4)
#include <barrier>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_5 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.bit
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.bit %t.actual.bit
#if defined(TEST_5)
#include <bit>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_6 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.bitset
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.bitset %t.actual.bitset
#if defined(TEST_6)
#include <bitset>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_7 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cassert
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cassert %t.actual.cassert
#if defined(TEST_7)
#include <cassert>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_8 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ccomplex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ccomplex %t.actual.ccomplex
#if defined(TEST_8)
#include <ccomplex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_9 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cctype
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cctype %t.actual.cctype
#if defined(TEST_9)
#include <cctype>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_10 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cerrno
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cerrno %t.actual.cerrno
#if defined(TEST_10)
#include <cerrno>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_11 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cfenv
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cfenv %t.actual.cfenv
#if defined(TEST_11)
#include <cfenv>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_12 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cfloat
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cfloat %t.actual.cfloat
#if defined(TEST_12)
#include <cfloat>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_13 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.charconv
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.charconv %t.actual.charconv
#if defined(TEST_13)
#include <charconv>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_14 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.chrono
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.chrono %t.actual.chrono
#if defined(TEST_14)
#include <chrono>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_15 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cinttypes
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cinttypes %t.actual.cinttypes
#if defined(TEST_15)
#include <cinttypes>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_16 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ciso646
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ciso646 %t.actual.ciso646
#if defined(TEST_16)
#include <ciso646>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_17 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.climits
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.climits %t.actual.climits
#if defined(TEST_17)
#include <climits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_18 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.clocale
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.clocale %t.actual.clocale
#if defined(TEST_18)
#include <clocale>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_19 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cmath
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cmath %t.actual.cmath
#if defined(TEST_19)
#include <cmath>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_20 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.codecvt
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.codecvt %t.actual.codecvt
#if defined(TEST_20)
#include <codecvt>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_21 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.compare
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.compare %t.actual.compare
#if defined(TEST_21)
#include <compare>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_22 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.complex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.complex %t.actual.complex
#if defined(TEST_22)
#include <complex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_24 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.concepts
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.concepts %t.actual.concepts
#if defined(TEST_24)
#include <concepts>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_25 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.condition_variable
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.condition_variable %t.actual.condition_variable
#if defined(TEST_25)
#include <condition_variable>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_26 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.coroutine
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.coroutine %t.actual.coroutine
#if defined(TEST_26)
#include <coroutine>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_27 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.csetjmp
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.csetjmp %t.actual.csetjmp
#if defined(TEST_27)
#include <csetjmp>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_28 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.csignal
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.csignal %t.actual.csignal
#if defined(TEST_28)
#include <csignal>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_29 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstdarg
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstdarg %t.actual.cstdarg
#if defined(TEST_29)
#include <cstdarg>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_30 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstdbool
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstdbool %t.actual.cstdbool
#if defined(TEST_30)
#include <cstdbool>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_31 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstddef
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstddef %t.actual.cstddef
#if defined(TEST_31)
#include <cstddef>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_32 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstdint
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstdint %t.actual.cstdint
#if defined(TEST_32)
#include <cstdint>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_33 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstdio
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstdio %t.actual.cstdio
#if defined(TEST_33)
#include <cstdio>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_34 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstdlib
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstdlib %t.actual.cstdlib
#if defined(TEST_34)
#include <cstdlib>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_35 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cstring
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cstring %t.actual.cstring
#if defined(TEST_35)
#include <cstring>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_36 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ctgmath
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ctgmath %t.actual.ctgmath
#if defined(TEST_36)
#include <ctgmath>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_37 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ctime
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ctime %t.actual.ctime
#if defined(TEST_37)
#include <ctime>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_39 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cuchar
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cuchar %t.actual.cuchar
#if defined(TEST_39)
#include <cuchar>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_40 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cwchar
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cwchar %t.actual.cwchar
#if defined(TEST_40)
#include <cwchar>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_41 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.cwctype
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.cwctype %t.actual.cwctype
#if defined(TEST_41)
#include <cwctype>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_42 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.deque
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.deque %t.actual.deque
#if defined(TEST_42)
#include <deque>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_44 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.exception
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.exception %t.actual.exception
#if defined(TEST_44)
#include <exception>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_45 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.execution
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.execution %t.actual.execution
#if defined(TEST_45)
#include <execution>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_47 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.filesystem
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.filesystem %t.actual.filesystem
#if defined(TEST_47)
#include <filesystem>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_49 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.format
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.format %t.actual.format
#if defined(TEST_49)
#include <format>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_50 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.forward_list
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.forward_list %t.actual.forward_list
#if defined(TEST_50)
#include <forward_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_51 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.fstream
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.fstream %t.actual.fstream
#if defined(TEST_51)
#include <fstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_52 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.functional
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.functional %t.actual.functional
#if defined(TEST_52)
#include <functional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_53 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.future
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.future %t.actual.future
#if defined(TEST_53)
#include <future>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_54 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.initializer_list
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.initializer_list %t.actual.initializer_list
#if defined(TEST_54)
#include <initializer_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_56 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.iomanip
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.iomanip %t.actual.iomanip
#if defined(TEST_56)
#include <iomanip>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_57 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ios
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ios %t.actual.ios
#if defined(TEST_57)
#include <ios>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_58 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.iosfwd
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.iosfwd %t.actual.iosfwd
#if defined(TEST_58)
#include <iosfwd>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_59 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.iostream
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.iostream %t.actual.iostream
#if defined(TEST_59)
#include <iostream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_60 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.istream
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.istream %t.actual.istream
#if defined(TEST_60)
#include <istream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_61 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.iterator
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.iterator %t.actual.iterator
#if defined(TEST_61)
#include <iterator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_62 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.latch
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.latch %t.actual.latch
#if defined(TEST_62)
#include <latch>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_63 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.limits
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.limits %t.actual.limits
#if defined(TEST_63)
#include <limits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_65 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.list
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.list %t.actual.list
#if defined(TEST_65)
#include <list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_66 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.locale
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.locale %t.actual.locale
#if defined(TEST_66)
#include <locale>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_68 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.map
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.map %t.actual.map
#if defined(TEST_68)
#include <map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_70 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.memory
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.memory %t.actual.memory
#if defined(TEST_70)
#include <memory>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_71 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.mutex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.mutex %t.actual.mutex
#if defined(TEST_71)
#include <mutex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_72 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.new
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.new %t.actual.new
#if defined(TEST_72)
#include <new>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_73 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.numbers
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.numbers %t.actual.numbers
#if defined(TEST_73)
#include <numbers>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_74 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.numeric
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.numeric %t.actual.numeric
#if defined(TEST_74)
#include <numeric>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_75 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.optional
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.optional %t.actual.optional
#if defined(TEST_75)
#include <optional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_76 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ostream
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ostream %t.actual.ostream
#if defined(TEST_76)
#include <ostream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_77 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.queue
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.queue %t.actual.queue
#if defined(TEST_77)
#include <queue>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_78 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.random
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.random %t.actual.random
#if defined(TEST_78)
#include <random>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_79 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ranges
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ranges %t.actual.ranges
#if defined(TEST_79)
#include <ranges>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_80 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ratio
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ratio %t.actual.ratio
#if defined(TEST_80)
#include <ratio>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_81 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.regex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.regex %t.actual.regex
#if defined(TEST_81)
#include <regex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_82 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.scoped_allocator
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.scoped_allocator %t.actual.scoped_allocator
#if defined(TEST_82)
#include <scoped_allocator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_83 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.semaphore
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.semaphore %t.actual.semaphore
#if defined(TEST_83)
#include <semaphore>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_84 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.set
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.set %t.actual.set
#if defined(TEST_84)
#include <set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_86 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.shared_mutex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.shared_mutex %t.actual.shared_mutex
#if defined(TEST_86)
#include <shared_mutex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_87 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.span
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.span %t.actual.span
#if defined(TEST_87)
#include <span>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_88 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.sstream
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.sstream %t.actual.sstream
#if defined(TEST_88)
#include <sstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_89 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.stack
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.stack %t.actual.stack
#if defined(TEST_89)
#include <stack>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_93 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.stdexcept
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.stdexcept %t.actual.stdexcept
#if defined(TEST_93)
#include <stdexcept>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_97 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.streambuf
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.streambuf %t.actual.streambuf
#if defined(TEST_97)
#include <streambuf>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_98 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.string
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.string %t.actual.string
#if defined(TEST_98)
#include <string>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_100 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.string_view
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.string_view %t.actual.string_view
#if defined(TEST_100)
#include <string_view>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_101 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.strstream
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.strstream %t.actual.strstream
#if defined(TEST_101)
#include <strstream>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_102 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.system_error
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.system_error %t.actual.system_error
#if defined(TEST_102)
#include <system_error>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_104 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.thread
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.thread %t.actual.thread
#if defined(TEST_104)
#include <thread>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_105 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.tuple
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.tuple %t.actual.tuple
#if defined(TEST_105)
#include <tuple>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_106 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.type_traits
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.type_traits %t.actual.type_traits
#if defined(TEST_106)
#include <type_traits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_107 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.typeindex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.typeindex %t.actual.typeindex
#if defined(TEST_107)
#include <typeindex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_108 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.typeinfo
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.typeinfo %t.actual.typeinfo
#if defined(TEST_108)
#include <typeinfo>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_110 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.unordered_map
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.unordered_map %t.actual.unordered_map
#if defined(TEST_110)
#include <unordered_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_111 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.unordered_set
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.unordered_set %t.actual.unordered_set
#if defined(TEST_111)
#include <unordered_set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_112 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.utility
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.utility %t.actual.utility
#if defined(TEST_112)
#include <utility>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_113 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.valarray
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.valarray %t.actual.valarray
#if defined(TEST_113)
#include <valarray>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_114 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.variant
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.variant %t.actual.variant
#if defined(TEST_114)
#include <variant>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_115 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.vector
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.vector %t.actual.vector
#if defined(TEST_115)
#include <vector>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_116 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.version
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.version %t.actual.version
#if defined(TEST_116)
#include <version>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_119 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_algorithm
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_algorithm %t.actual.experimental_algorithm
#if defined(TEST_119)
#include <experimental/algorithm>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_120 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_coroutine
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_coroutine %t.actual.experimental_coroutine
#if defined(TEST_120)
#include <experimental/coroutine>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_121 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_deque
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_deque %t.actual.experimental_deque
#if defined(TEST_121)
#include <experimental/deque>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_122 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_forward_list
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_forward_list %t.actual.experimental_forward_list
#if defined(TEST_122)
#include <experimental/forward_list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_123 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_functional
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_functional %t.actual.experimental_functional
#if defined(TEST_123)
#include <experimental/functional>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_124 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_iterator
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_iterator %t.actual.experimental_iterator
#if defined(TEST_124)
#include <experimental/iterator>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_125 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_list
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_list %t.actual.experimental_list
#if defined(TEST_125)
#include <experimental/list>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_126 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_map
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_map %t.actual.experimental_map
#if defined(TEST_126)
#include <experimental/map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_127 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_memory_resource
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_memory_resource %t.actual.experimental_memory_resource
#if defined(TEST_127)
#include <experimental/memory_resource>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_128 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_propagate_const
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_propagate_const %t.actual.experimental_propagate_const
#if defined(TEST_128)
#include <experimental/propagate_const>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_129 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_regex
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_regex %t.actual.experimental_regex
#if defined(TEST_129)
#include <experimental/regex>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_130 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_set
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_set %t.actual.experimental_set
#if defined(TEST_130)
#include <experimental/set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_131 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_simd
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_simd %t.actual.experimental_simd
#if defined(TEST_131)
#include <experimental/simd>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_132 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_string
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_string %t.actual.experimental_string
#if defined(TEST_132)
#include <experimental/string>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_133 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_type_traits
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_type_traits %t.actual.experimental_type_traits
#if defined(TEST_133)
#include <experimental/type_traits>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_134 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_unordered_map
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_unordered_map %t.actual.experimental_unordered_map
#if defined(TEST_134)
#include <experimental/unordered_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_135 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_unordered_set
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_unordered_set %t.actual.experimental_unordered_set
#if defined(TEST_135)
#include <experimental/unordered_set>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_136 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_utility
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_utility %t.actual.experimental_utility
#if defined(TEST_136)
#include <experimental/utility>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_137 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.experimental_vector
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.experimental_vector %t.actual.experimental_vector
#if defined(TEST_137)
#include <experimental/vector>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_138 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ext_hash_map
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ext_hash_map %t.actual.ext_hash_map
#if defined(TEST_138)
#include <ext/hash_map>
#endif
// RUN: %{cxx} %s %{flags} %{compile_flags} --trace-includes -fsyntax-only -DTEST_139 2>&1 | %{python} %S/transitive_includes.sanitize.py > %t.actual.ext_hash_set
// RUN: diff -w %S/transitive_includes/%{cxx_std}/expected.ext_hash_set %t.actual.ext_hash_set
#if defined(TEST_139)
#include <ext/hash_set>
#endif
// GENERATED-MARKER
