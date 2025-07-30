# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# REQUIRES: has-clang-tidy

# RUN: %{python} %{libcxx-dir}/../clang-tools-extra/clang-tidy/tool/run-clang-tidy.py -clang-tidy-binary %{clang-tidy} -warnings-as-errors "*" -source-filter=".*libcxx/src.*" -quiet -p %{bin-dir}/..
