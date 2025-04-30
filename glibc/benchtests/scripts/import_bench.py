#!/usr/bin/python
# Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
"""Functions to import benchmark data and process it"""

import json
try:
    import jsonschema as validator
except ImportError:
    print('Could not find jsonschema module.')
    raise


def mean(lst):
    """Compute and return mean of numbers in a list

    The numpy average function has horrible performance, so implement our
    own mean function.

    Args:
        lst: The list of numbers to average.
    Return:
        The mean of members in the list.
    """
    return sum(lst) / len(lst)


def split_list(bench, func, var):
    """ Split the list into a smaller set of more distinct points

    Group together points such that the difference between the smallest
    point and the mean is less than 1/3rd of the mean.  This means that
    the mean is at most 1.5x the smallest member of that group.

    mean - xmin < mean / 3
    i.e. 2 * mean / 3 < xmin
    i.e. mean < 3 * xmin / 2

    For an evenly distributed group, the largest member will be less than
    twice the smallest member of the group.
    Derivation:

    An evenly distributed series would be xmin, xmin + d, xmin + 2d...

    mean = (2 * n * xmin + n * (n - 1) * d) / 2 * n
    and max element is xmin + (n - 1) * d

    Now, mean < 3 * xmin / 2

    3 * xmin > 2 * mean
    3 * xmin > (2 * n * xmin + n * (n - 1) * d) / n
    3 * n * xmin > 2 * n * xmin + n * (n - 1) * d
    n * xmin > n * (n - 1) * d
    xmin > (n - 1) * d
    2 * xmin > xmin + (n-1) * d
    2 * xmin > xmax

    Hence, proved.

    Similarly, it is trivial to prove that for a similar aggregation by using
    the maximum element, the maximum element in the group must be at most 4/3
    times the mean.

    Args:
        bench: The benchmark object
        func: The function name
        var: The function variant name
    """
    means = []
    lst = bench['functions'][func][var]['timings']
    last = len(lst) - 1
    while lst:
        for i in range(last + 1):
            avg = mean(lst[i:])
            if avg > 0.75 * lst[last]:
                means.insert(0, avg)
                lst = lst[:i]
                last = i - 1
                break
    bench['functions'][func][var]['timings'] = means


def do_for_all_timings(bench, callback):
    """Call a function for all timing objects for each function and its
    variants.

    Args:
        bench: The benchmark object
        callback: The callback function
    """
    for func in bench['functions'].keys():
        for k in bench['functions'][func].keys():
            if 'timings' not in bench['functions'][func][k].keys():
                continue

            callback(bench, func, k)


def compress_timings(points):
    """Club points with close enough values into a single mean value

    See split_list for details on how the clubbing is done.

    Args:
        points: The set of points.
    """
    do_for_all_timings(points, split_list)


def parse_bench(filename, schema_filename):
    """Parse the input file

    Parse and validate the json file containing the benchmark outputs.  Return
    the resulting object.
    Args:
        filename: Name of the benchmark output file.
    Return:
        The bench dictionary.
    """
    with open(schema_filename, 'r') as schemafile:
        schema = json.load(schemafile)
        with open(filename, 'r') as benchfile:
            bench = json.load(benchfile)
            validator.validate(bench, schema)
            do_for_all_timings(bench, lambda b, f, v:
                    b['functions'][f][v]['timings'].sort())
            return bench
