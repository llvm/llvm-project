#!/usr/bin/python3
# Expand test inputs for fromfp functions into text to edit into libm-test.inc.
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

# Take test inputs on stdin, in format:
#
# i <value>:width [int-value]
#
# for integer inputs, or
#
# t <value> <pos> <z> <a>
#
# for noninteger inputs, where <pos> is "a" for fractional part
# between 0 and 0.5, "be" for 0.5 with even integer part, "bo" for 0.5
# with odd integer part and "c" for between 0.5 and 1; <z> is the
# value truncated towards zero, <a> is the value rounded away from
# zero, both being in the form <value>:<width>.  Width values are for
# the smallest type that can hold the value; for positive values, this
# is an unsigned type.
#
# Command-line argument is function to generate tests for.  Any input
# lines not of the above form are just passed through unchanged.
#
# Note that the output of this script forms the largest part of the
# tests for the fromfp functions, but not the whole of those tests.

import sys

func = sys.argv[1]

invalid_res = 'IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM'
exact_res = 'NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED'
if func == 'fromfpx' or func == 'ufromfpx':
    inexact_res = 'INEXACT_EXCEPTION|ERRNO_UNCHANGED'
else:
    inexact_res = exact_res
unsigned = func.startswith('ufromfp')
rm_list = ['FP_INT_UPWARD', 'FP_INT_DOWNWARD', 'FP_INT_TOWARDZERO',
           'FP_INT_TONEARESTFROMZERO', 'FP_INT_TONEAREST']
rm_away_pos = {'FP_INT_UPWARD': 'a',
               'FP_INT_DOWNWARD': 'z',
               'FP_INT_TOWARDZERO': 'z',
               'FP_INT_TONEARESTFROMZERO': 'be',
               'FP_INT_TONEAREST': 'bo'}
rm_away_neg = {'FP_INT_UPWARD': 'z',
               'FP_INT_DOWNWARD': 'a',
               'FP_INT_TOWARDZERO': 'z',
               'FP_INT_TONEARESTFROMZERO': 'be',
               'FP_INT_TONEAREST': 'bo'}
if unsigned:
    test_macro = 'TEST_fiu_U'
else:
    test_macro = 'TEST_fiu_M'

for line in sys.stdin:
    if line.startswith('i'):
        data = line.split()
        val_width = data[1]
        val, width = val_width.split(':')
        negative = val.startswith('-')
        if unsigned and negative:
            continue
        width = int(width)
        if not unsigned and not negative:
            width += 1
        width_list = [0, 1]
        if width > 2:
            width_list.append(width - 1)
        if width > 1 and width <= 64:
            width_list.append(width)
        if width < 64:
            width_list.append(width + 1)
        if width < 63:
            width_list.append(64)
        width_list = [(w, str(w)) for w in width_list]
        width_list.append((64, 'UINT_MAX'))
        for rm in rm_list:
            for we in width_list:
                w, ws = we
                if w < width:
                    print('    %s (%s, %s, %s, %s, %s),' %
                          (test_macro, func, val, rm, ws, invalid_res))
                else:
                    print('    %s (%s, %s, %s, %s, %s, %s),' %
                          (test_macro, func, val, rm, ws, val, exact_res))
    elif line.startswith('t'):
        data = line.split()
        val = data[1]
        pos = data[2]
        z, z_width = data[3].split(':')
        z_width = int(z_width)
        a, a_width = data[4].split(':')
        a_width = int(a_width)
        if unsigned and z.startswith('-'):
            continue
        negative = val.startswith('-')
        if negative:
            rm_away = rm_away_neg
        else:
            rm_away = rm_away_pos
        for rm in rm_list:
            if pos >= rm_away[rm]:
                res, width = a, a_width
            else:
                res, width = z, z_width
            if not unsigned and not negative and res != '0':
                width += 1
            width_list = [0, 1]
            if width > 2:
                width_list.append(width - 1)
            if width > 1 and width <= 64:
                width_list.append(width)
            if width < 64:
                width_list.append(width + 1)
            if width < 63:
                width_list.append(64)
            width_list = [(w, str(w)) for w in width_list]
            width_list.append((64, 'UINT_MAX'))
            for we in width_list:
                w, ws = we
                if w < width or (unsigned and res.startswith('-')):
                    print('    %s (%s, %s, %s, %s, %s),' %
                          (test_macro, func, val, rm, ws, invalid_res))
                else:
                    print('    %s (%s, %s, %s, %s, %s, %s),' %
                          (test_macro, func, val, rm, ws, res, inexact_res))
    else:
        print(line.rstrip())
