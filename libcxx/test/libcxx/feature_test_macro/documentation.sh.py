# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# RUN: %{python} %s %{libcxx-dir}/utils %{libcxx-dir}/test/libcxx/feature_test_macro/test_data.json

import sys
import unittest

UTILS = sys.argv[1]
TEST_DATA = sys.argv[2]
del sys.argv[1:3]

sys.path.append(UTILS)
from generate_feature_test_macro_components import FeatureTestMacros


class Test(unittest.TestCase):
    def setUp(self):
        self.ftm = FeatureTestMacros(TEST_DATA, ["charconv"])
        self.maxDiff = None  # This causes the diff to be printed when the test fails

    def test_implementation(self):
        expected = """\
.. _FeatureTestMacroTable:

==========================
Feature Test Macro Support
==========================

.. contents::
   :local:

Overview
========

This page documents libc++'s implementation status of the Standard library
feature test macros. This page does not list all details, that information can
be found at the `isoccp
<https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations#library-feature-test-macros>`__.

.. _feature-status:

Status
======

.. list-table:: Current Status
  :widths: auto
  :header-rows: 1
  :align: left

  * - Macro Name
    - Libc++ Value
    - Standard Value
    - 
    - Paper
  * - | **C++17**
    - | 
    - | 
    - | 
    - | 
  * - | ``__cpp_lib_any``
    - | 201606L
    - | 201606L
    - | ✅
    - | 
  * - | ``__cpp_lib_clamp``
    - | 201603L
    - | 201603L
    - | ✅
    - | 
  * - | ``__cpp_lib_parallel_algorithm``
    - | 201603L
    - | 201603L
    - | ✅
    - | 
  * - | ``__cpp_lib_to_chars``
    - | *unimplemented*
    - | 201611L
    - | ❌
    - | 
  * - | ``__cpp_lib_variant``
    - | 202102L
    - | 202102L
    - | ✅
    - | ``std::visit`` for classes derived from ``std::variant``
  * - | ``__cpp_lib_zz_missing_FTM_in_older_standard``
    - | *unimplemented*
    - | 2017L
    - | ❌
    - | Some FTM missing a paper in an older Standard mode, which should result in the FTM never being defined.
  * - | **C++20**
    - | 
    - | 
    - | 
    - | 
  * - | ``__cpp_lib_barrier``
    - | 201907L
    - | 201907L
    - | ✅
    - | 
  * - | ``__cpp_lib_format``
      | 
      | 
      | 
      | 
    - | *unimplemented*
      | 
      | 
      | 
      | 
    - | 201907L
      | 
      | 202106L
      | 202110L
      | 
    - | ✅
      | ❌
      | ✅
      | ❌
      | ✅
    - | `P0645R10 <https://wg21.link/P0645R10>`__ Text Formatting
      | `P1361R2 <https://wg21.link/P1361R2>`__ Integration of chrono with text formatting
      | `P2216R3 <https://wg21.link/P2216R3>`__ std::format improvements
      | `P2372R3 <https://wg21.link/P2372R3>`__ Fixing locale handling in chrono formatters
      | `P2418R2 <https://wg21.link/P2418R2>`__ FAdd support for std::generator-like types to std::format
  * - | ``__cpp_lib_variant``
    - | *unimplemented*
    - | 202106L
    - | ❌
    - | Fully constexpr ``std::variant``
  * - | ``__cpp_lib_zz_missing_FTM_in_older_standard``
    - | *unimplemented*
    - | 2020L
    - | ✅
    - | 
  * - | **C++23**
    - | 
    - | 
    - | 
    - | 
  * - | ``__cpp_lib_format``
    - | *unimplemented*
    - | 202207L
    - | ❌
    - | `P2419R2 <https://wg21.link/P2419R2>`__ Clarify handling of encodings in localized formatting of chrono types
  * - | **C++26**
    - | 
    - | 
    - | 
    - | 
  * - | ``__cpp_lib_barrier``
    - | 299900L
    - | 299900L
    - | ✅
    - | 
  * - | ``__cpp_lib_format``
      | 
    - | *unimplemented*
      | 
    - | 202306L
      | 202311L
    - | ✅
      | ✅
    - | `P2637R3 <https://wg21.link/P2637R3>`__ Member Visit
      | `P2918R2 <https://wg21.link/P2918R2>`__ Runtime format strings II
  * - | ``__cpp_lib_variant``
    - | *unimplemented*
    - | 202306L
    - | ✅
    - | Member visit
  * - | ``__cpp_lib_zz_missing_FTM_in_older_standard``
    - | *unimplemented*
    - | 2026L
    - | ✅
    - | 

"""
        self.assertEqual(self.ftm.documentation, expected)


if __name__ == "__main__":
    unittest.main()
