# -*- Python -*-
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# All the Lit configuration is handled in the site config -- this file is only
# left as a canary to catch invocations of Lit that do not go through llvm-lit.
#
# Invocations that go through llvm-lit will automatically use the right Lit
# site configuration inside the build directory.

lit_config.fatal(
    "You seem to be running Lit directly -- you should be running Lit through "
    "<build>/bin/llvm-lit, which will ensure that the right Lit configuration "
    "file is used."
)
