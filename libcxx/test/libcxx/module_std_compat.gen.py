# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Test that all named declarations with external linkage match the
# exported declarations in their associated module partition.
# Then it tests the sum of the exported declarations in the module
# partitions matches the export of the std.compat module.

# Note the test of the std.compat module requires all partitions to be tested
# first. Since lit tests have no dependencies, this means the test needs
# to be one monolitic test. Since the test doesn't take very long it's
# not a huge issue.

# RUN: %{python} %s %{libcxx-dir}/utils

import sys

sys.path.append(sys.argv[1])
from libcxx.header_information import module_c_headers
from libcxx.test.modules import module_test_generator

generator = module_test_generator(
    "%t",
    "%{module-dir}",
    "%{clang-tidy}",
    "%{test-tools-dir}/clang_tidy_checks/libcxx-tidy.plugin",
    "%{cxx}",
    "%{flags} %{compile_flags}",
    "std.compat",
)


print("//--- module_std_compat.sh.cpp")
generator.write_test(
    "std.compat",
    module_c_headers,
)
