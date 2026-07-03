# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from . import availability, compiler, gdb, libcxx_macros, localization, misc, platform

# Lit features are evaluated in order. Some features depend on other features, so
# we are careful to define them in the correct order. For example, several features
# require the compiler detection to have been performed.
DEFAULT_FEATURES = []
DEFAULT_FEATURES += compiler.features
DEFAULT_FEATURES += libcxx_macros.features
DEFAULT_FEATURES += platform.features
DEFAULT_FEATURES += localization.features
DEFAULT_FEATURES += gdb.features
DEFAULT_FEATURES += misc.features
DEFAULT_FEATURES += availability.features
