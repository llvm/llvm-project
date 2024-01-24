# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import lldb

# https://lldb.llvm.org/use/python-reference.html#running-a-python-script-when-a-breakpoint-gets-hit

def breakpoint_function_wrapper(frame, bp_loc, internal_dict):
   # Your code goes here
