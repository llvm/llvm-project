"""report.py - Utilities for reporting statistics about benchmark results
"""
import os

class BenchmarkColor(object):
    def __init__(self, name, code):
        self.name = name
        self.code = code

    def __repr__(self):
        return '%s%r' % (self.__class__.__name__,
                         (self.name, self.code))

    def __format__(self, format):
        return self.code

# Benchmark Colors Enumeration
BC_NONE = BenchmarkColor('NONE', '')
BC_MAGENTA = BenchmarkColor('MAGENTA', '\033[95m')
BC_CYAN = BenchmarkColor('CYAN', '\033[96m')
BC_OKBLUE = BenchmarkColor('OKBLUE', '\033[94m')
BC_HEADER = BenchmarkColor('HEADER', '\033[92m')
BC_WARNING = BenchmarkColor('WARNING', '\033[93m')
BC_WHITE = BenchmarkColor('WHITE', '\033[97m')
BC_FAIL = BenchmarkColor('FAIL', '\033[91m')
BC_ENDC = BenchmarkColor('ENDC', '\033[0m')
BC_BOLD = BenchmarkColor('BOLD', '\033[1m')
BC_UNDERLINE = BenchmarkColor('UNDERLINE', '\033[4m')

def color_format(use_color, fmt_str, *args, **kwargs):
    """
    Return the result of 'fmt_str.format(*args, **kwargs)' after transforming
    'args' and 'kwargs' according to the value of 'use_color'. If 'use_color'
    is False then all color codes in 'args' and 'kwargs' are replaced with
    the empty string.
    """
    assert use_color is True or use_color is False
    if not use_color:
        args = [arg if not isinstance(arg, BenchmarkColor) else BC_NONE
                for arg in args]
        kwargs = {key: arg if not isinstance(arg, BenchmarkColor) else BC_NONE
                  for key, arg in kwargs.items()}
    return fmt_str.format(*args, **kwargs)


def find_longest_name(benchmark_list):
    """
    Return the length of the longest benchmark name in a given list of
    benchmark JSON objects
    """
    longest_name = 1
    for bc in benchmark_list:
        if len(bc['name']) > longest_name:
            longest_name = len(bc['name'])
    return longest_name


def calculate_change(old_val, new_val):
    """
    Return a float representing the decimal change between old_val and new_val.
    """
    if old_val == 0 and new_val == 0:
        return 0.0
    if old_val == 0:
        return float(new_val - old_val) / (float(old_val + new_val) / 2)
    return float(new_val - old_val) / abs(old_val)


def generate_difference_report(json1, json2, use_color=True):
    """
    Calculate and report the difference between each test of two benchmarks
    runs specified as 'json1' and 'json2'.
    """
    first_col_width = find_longest_name(json1['benchmarks']) + 5
    def find_test(name):
        for b in json2['benchmarks']:
            if b['name'] == name:
                return b
        return None
    first_line = "{:<{}s}     Time           CPU           Old           New".format(
        'Benchmark', first_col_width)
    output_strs = [first_line, '-' * len(first_line)]
    for bn in json1['benchmarks']:
        other_bench = find_test(bn['name'])
        if not other_bench:
            continue

        def get_color(res):
            if res > 0.05:
                return BC_FAIL
            elif res > -0.07:
                return BC_WHITE
            else:
                return BC_CYAN
        fmt_str = "{}{:<{}s}{endc}{}{:+9.2f}{endc}{}{:+14.2f}{endc}{:14d}{:14d}"
        tres = calculate_change(bn['real_time'], other_bench['real_time'])
        cpures = calculate_change(bn['cpu_time'], other_bench['cpu_time'])
        output_strs += [color_format(use_color, fmt_str,
            BC_HEADER, bn['name'], first_col_width,
            get_color(tres), tres, get_color(cpures), cpures,
            bn['cpu_time'], other_bench['cpu_time'],
            endc=BC_ENDC)]
    return output_strs

###############################################################################
# Unit tests

import unittest

class TestReportDifference(unittest.TestCase):
    def load_results(self):
        import json
        testInputs = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Inputs')
        testOutput1 = os.path.join(testInputs, 'test1_run1.json')
        testOutput2 = os.path.join(testInputs, 'test1_run2.json')
        with open(testOutput1, 'r') as f:
            json1 = json.load(f)
        with open(testOutput2, 'r') as f:
            json2 = json.load(f)
        return json1, json2

    def test_basic(self):
        expect_lines = [
            ['BM_SameTimes', '+0.00', '+0.00', '10', '10'],
            ['BM_2xFaster', '-0.50', '-0.50', '50', '25'],
            ['BM_2xSlower', '+1.00', '+1.00', '50', '100'],
            ['BM_10PercentFaster', '-0.10', '-0.10', '100', '90'],
            ['BM_10PercentSlower', '+0.10', '+0.10', '100', '110'],
            ['BM_100xSlower', '+99.00', '+99.00', '100', '10000'],
            ['BM_100xFaster', '-0.99', '-0.99', '10000', '100'],
        ]
        json1, json2 = self.load_results()
        output_lines_with_header = generate_difference_report(json1, json2, use_color=False)
        output_lines = output_lines_with_header[2:]
        print("\n".join(output_lines_with_header))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in xrange(0, len(output_lines)):
            parts = [x for x in output_lines[i].split(' ') if x]
            self.assertEqual(len(parts), 5)
            self.assertEqual(parts, expect_lines[i])


if __name__ == '__main__':
    unittest.main()
