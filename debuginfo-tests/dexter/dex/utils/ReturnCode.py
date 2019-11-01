# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum


class ReturnCode(Enum):
   """Used to indicate whole program success status."""

   OK = 0
   _ERROR = 1        # Unhandled exceptions result in exit(1) by default.
                     # Usage of _ERROR is discouraged:
                     # If the program cannot run, raise an exception.
                     # If the program runs successfully but the result is
                     # "failure" based on the inputs, return FAIL
   FAIL = 2
