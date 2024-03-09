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
# partitions matches the export of the std module.

# Note the test of the std module requires all partitions to be tested
# first. Since lit tests have no dependencies, this means the test needs
# to be one monolitic test. Since the test doesn't take very long it's
# not a huge issue.

# WARNING: Disabled at the bottom. Fix this test and remove the UNSUPPORTED line
# TODO: Re-enable this test once we understand why it keeps timing out.

# RUN: %{python} %s %{libcxx-dir}/utils
# END.

import sys

sys.path.append(sys.argv[1])
from libcxx.test.modules import module_test_generator

generator = module_test_generator(
    "%t",
    "%{module-dir}",
    "%{clang-tidy}",
    "%{test-tools-dir}/clang_tidy_checks/libcxx-tidy.plugin",
    "%{cxx}",
    "%{flags} %{compile_flags}",
    "std",
)


print("//--- module_std.sh.cpp")
print('// UNSUPPORTED: clang')
generator.write_test("std")
