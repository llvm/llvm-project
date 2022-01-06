"""report.py - Utilities for reporting statistics about benchmark results
"""

import unittest
import os
import re
import copy
import random

from scipy.stats import mannwhitneyu


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
BC_OKGREEN = BenchmarkColor('OKGREEN', '\033[32m')
BC_HEADER = BenchmarkColor('HEADER', '\033[92m')
BC_WARNING = BenchmarkColor('WARNING', '\033[93m')
BC_WHITE = BenchmarkColor('WHITE', '\033[97m')
BC_FAIL = BenchmarkColor('FAIL', '\033[91m')
BC_ENDC = BenchmarkColor('ENDC', '\033[0m')
BC_BOLD = BenchmarkColor('BOLD', '\033[1m')
BC_UNDERLINE = BenchmarkColor('UNDERLINE', '\033[4m')

UTEST_MIN_REPETITIONS = 2
UTEST_OPTIMAL_REPETITIONS = 9  # Lowest reasonable number, More is better.
UTEST_COL_NAME = "_pvalue"


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


def filter_benchmark(json_orig, family, replacement=""):
    """
    Apply a filter to the json, and only leave the 'family' of benchmarks.
    """
    regex = re.compile(family)
    filtered = {}
    filtered['benchmarks'] = []
    for be in json_orig['benchmarks']:
        if not regex.search(be['name']):
            continue
        filteredbench = copy.deepcopy(be)  # Do NOT modify the old name!
        filteredbench['name'] = regex.sub(replacement, filteredbench['name'])
        filtered['benchmarks'].append(filteredbench)
    return filtered


def get_unique_benchmark_names(json):
    """
    While *keeping* the order, give all the unique 'names' used for benchmarks.
    """
    seen = set()
    uniqued = [x['name'] for x in json['benchmarks']
               if x['name'] not in seen and
               (seen.add(x['name']) or True)]
    return uniqued


def intersect(list1, list2):
    """
    Given two lists, get a new list consisting of the elements only contained
    in *both of the input lists*, while preserving the ordering.
    """
    return [x for x in list1 if x in list2]


def is_potentially_comparable_benchmark(x):
    return ('time_unit' in x and 'real_time' in x and 'cpu_time' in x)


def partition_benchmarks(json1, json2):
    """
    While preserving the ordering, find benchmarks with the same names in
    both of the inputs, and group them.
    (i.e. partition/filter into groups with common name)
    """
    json1_unique_names = get_unique_benchmark_names(json1)
    json2_unique_names = get_unique_benchmark_names(json2)
    names = intersect(json1_unique_names, json2_unique_names)
    partitions = []
    for name in names:
        time_unit = None
        # Pick the time unit from the first entry of the lhs benchmark.
        # We should be careful not to crash with unexpected input.
        for x in json1['benchmarks']:
            if (x['name'] == name and is_potentially_comparable_benchmark(x)):
                time_unit = x['time_unit']
                break
        if time_unit is None:
            continue
        # Filter by name and time unit.
        # All the repetitions are assumed to be comparable.
        lhs = [x for x in json1['benchmarks'] if x['name'] == name and
               x['time_unit'] == time_unit]
        rhs = [x for x in json2['benchmarks'] if x['name'] == name and
               x['time_unit'] == time_unit]
        partitions.append([lhs, rhs])
    return partitions


def extract_field(partition, field_name):
    # The count of elements may be different. We want *all* of them.
    lhs = [x[field_name] for x in partition[0]]
    rhs = [x[field_name] for x in partition[1]]
    return [lhs, rhs]


def calc_utest(timings_cpu, timings_time):
    min_rep_cnt = min(len(timings_time[0]),
                      len(timings_time[1]),
                      len(timings_cpu[0]),
                      len(timings_cpu[1]))

    # Does *everything* has at least UTEST_MIN_REPETITIONS repetitions?
    if min_rep_cnt < UTEST_MIN_REPETITIONS:
        return False, None, None

    time_pvalue = mannwhitneyu(
        timings_time[0], timings_time[1], alternative='two-sided').pvalue
    cpu_pvalue = mannwhitneyu(
        timings_cpu[0], timings_cpu[1], alternative='two-sided').pvalue

    return (min_rep_cnt >= UTEST_OPTIMAL_REPETITIONS), cpu_pvalue, time_pvalue

def print_utest(bc_name, utest, utest_alpha, first_col_width, use_color=True):
    def get_utest_color(pval):
        return BC_FAIL if pval >= utest_alpha else BC_OKGREEN

    # Check if we failed miserably with minimum required repetitions for utest
    if not utest['have_optimal_repetitions'] and utest['cpu_pvalue'] is None and utest['time_pvalue'] is None:
        return []

    dsc = "U Test, Repetitions: {} vs {}".format(
        utest['nr_of_repetitions'], utest['nr_of_repetitions_other'])
    dsc_color = BC_OKGREEN

    # We still got some results to show but issue a warning about it.
    if not utest['have_optimal_repetitions']:
        dsc_color = BC_WARNING
        dsc += ". WARNING: Results unreliable! {}+ repetitions recommended.".format(
            UTEST_OPTIMAL_REPETITIONS)

    special_str = "{}{:<{}s}{endc}{}{:16.4f}{endc}{}{:16.4f}{endc}{}      {}"

    return [color_format(use_color,
                         special_str,
                         BC_HEADER,
                         "{}{}".format(bc_name, UTEST_COL_NAME),
                         first_col_width,
                         get_utest_color(
                             utest['time_pvalue']), utest['time_pvalue'],
                         get_utest_color(
                             utest['cpu_pvalue']), utest['cpu_pvalue'],
                         dsc_color, dsc,
                         endc=BC_ENDC)]


def get_difference_report(
        json1,
        json2,
        utest=False):
    """
    Calculate and report the difference between each test of two benchmarks
    runs specified as 'json1' and 'json2'. Output is another json containing
    relevant details for each test run.
    """
    assert utest is True or utest is False

    diff_report = []
    partitions = partition_benchmarks(json1, json2)
    for partition in partitions:
        benchmark_name = partition[0][0]['name']
        time_unit = partition[0][0]['time_unit']
        measurements = []
        utest_results = {}
        # Careful, we may have different repetition count.
        for i in range(min(len(partition[0]), len(partition[1]))):
            bn = partition[0][i]
            other_bench = partition[1][i]
            measurements.append({
                'real_time': bn['real_time'],
                'cpu_time': bn['cpu_time'],
                'real_time_other': other_bench['real_time'],
                'cpu_time_other': other_bench['cpu_time'],
                'time': calculate_change(bn['real_time'], other_bench['real_time']),
                'cpu': calculate_change(bn['cpu_time'], other_bench['cpu_time'])
            })

        # After processing the whole partition, if requested, do the U test.
        if utest:
            timings_cpu = extract_field(partition, 'cpu_time')
            timings_time = extract_field(partition, 'real_time')
            have_optimal_repetitions, cpu_pvalue, time_pvalue = calc_utest(timings_cpu, timings_time)
            if cpu_pvalue and time_pvalue:
                utest_results = {
                    'have_optimal_repetitions': have_optimal_repetitions,
                    'cpu_pvalue': cpu_pvalue,
                    'time_pvalue': time_pvalue,
                    'nr_of_repetitions': len(timings_cpu[0]),
                    'nr_of_repetitions_other': len(timings_cpu[1])
                }

        # Store only if we had any measurements for given benchmark.
        # E.g. partition_benchmarks will filter out the benchmarks having
        # time units which are not compatible with other time units in the
        # benchmark suite.
        if measurements:
            run_type = partition[0][0]['run_type'] if 'run_type' in partition[0][0] else ''
            aggregate_name = partition[0][0]['aggregate_name'] if run_type == 'aggregate' and 'aggregate_name' in partition[0][0] else ''
            diff_report.append({
                'name': benchmark_name,
                'measurements': measurements,
                'time_unit': time_unit,
                'run_type': run_type,
                'aggregate_name': aggregate_name,
                'utest': utest_results
            })

    return diff_report


def print_difference_report(
        json_diff_report,
        include_aggregates_only=False,
        utest=False,
        utest_alpha=0.05,
        use_color=True):
    """
    Calculate and report the difference between each test of two benchmarks
    runs specified as 'json1' and 'json2'.
    """
    assert utest is True or utest is False

    def get_color(res):
        if res > 0.05:
            return BC_FAIL
        elif res > -0.07:
            return BC_WHITE
        else:
            return BC_CYAN

    first_col_width = find_longest_name(json_diff_report)
    first_col_width = max(
        first_col_width,
        len('Benchmark'))
    first_col_width += len(UTEST_COL_NAME)
    first_line = "{:<{}s}Time             CPU      Time Old      Time New       CPU Old       CPU New".format(
        'Benchmark', 12 + first_col_width)
    output_strs = [first_line, '-' * len(first_line)]

    fmt_str = "{}{:<{}s}{endc}{}{:+16.4f}{endc}{}{:+16.4f}{endc}{:14.0f}{:14.0f}{endc}{:14.0f}{:14.0f}"
    for benchmark in json_diff_report:
        # *If* we were asked to only include aggregates,
        # and if it is non-aggregate, then don't print it.
        if not include_aggregates_only or not 'run_type' in benchmark or benchmark['run_type'] == 'aggregate':
            for measurement in benchmark['measurements']:
                output_strs += [color_format(use_color,
                                            fmt_str,
                                            BC_HEADER,
                                            benchmark['name'],
                                            first_col_width,
                                            get_color(measurement['time']),
                                            measurement['time'],
                                            get_color(measurement['cpu']),
                                            measurement['cpu'],
                                            measurement['real_time'],
                                            measurement['real_time_other'],
                                            measurement['cpu_time'],
                                            measurement['cpu_time_other'],
                                            endc=BC_ENDC)]

        # After processing the measurements, if requested and
        # if applicable (e.g. u-test exists for given benchmark),
        # print the U test.
        if utest and benchmark['utest']:
            output_strs += print_utest(benchmark['name'],
                                       benchmark['utest'],
                                       utest_alpha=utest_alpha,
                                       first_col_width=first_col_width,
                                       use_color=use_color)

    return output_strs


###############################################################################
# Unit tests


class TestGetUniqueBenchmarkNames(unittest.TestCase):
    def load_results(self):
        import json
        testInputs = os.path.join(
            os.path.dirname(
                os.path.realpath(__file__)),
            'Inputs')
        testOutput = os.path.join(testInputs, 'test3_run0.json')
        with open(testOutput, 'r') as f:
            json = json.load(f)
        return json

    def test_basic(self):
        expect_lines = [
            'BM_One',
            'BM_Two',
            'short',  # These two are not sorted
            'medium',  # These two are not sorted
        ]
        json = self.load_results()
        output_lines = get_unique_benchmark_names(json)
        print("\n")
        print("\n".join(output_lines))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in range(0, len(output_lines)):
            self.assertEqual(expect_lines[i], output_lines[i])


class TestReportDifference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load_results():
            import json
            testInputs = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                'Inputs')
            testOutput1 = os.path.join(testInputs, 'test1_run1.json')
            testOutput2 = os.path.join(testInputs, 'test1_run2.json')
            with open(testOutput1, 'r') as f:
                json1 = json.load(f)
            with open(testOutput2, 'r') as f:
                json2 = json.load(f)
            return json1, json2

        json1, json2 = load_results()
        cls.json_diff_report = get_difference_report(json1, json2)

    def test_json_diff_report_pretty_printing(self):
        expect_lines = [
            ['BM_SameTimes', '+0.0000', '+0.0000', '10', '10', '10', '10'],
            ['BM_2xFaster', '-0.5000', '-0.5000', '50', '25', '50', '25'],
            ['BM_2xSlower', '+1.0000', '+1.0000', '50', '100', '50', '100'],
            ['BM_1PercentFaster', '-0.0100', '-0.0100', '100', '99', '100', '99'],
            ['BM_1PercentSlower', '+0.0100', '+0.0100', '100', '101', '100', '101'],
            ['BM_10PercentFaster', '-0.1000', '-0.1000', '100', '90', '100', '90'],
            ['BM_10PercentSlower', '+0.1000', '+0.1000', '100', '110', '100', '110'],
            ['BM_100xSlower', '+99.0000', '+99.0000',
                '100', '10000', '100', '10000'],
            ['BM_100xFaster', '-0.9900', '-0.9900',
                '10000', '100', '10000', '100'],
            ['BM_10PercentCPUToTime', '+0.1000',
                '-0.1000', '100', '110', '100', '90'],
            ['BM_ThirdFaster', '-0.3333', '-0.3334', '100', '67', '100', '67'],
            ['BM_NotBadTimeUnit', '-0.9000', '+0.2000', '0', '0', '0', '1'],
        ]
        output_lines_with_header = print_difference_report(
            self.json_diff_report, use_color=False)
        output_lines = output_lines_with_header[2:]
        print("\n")
        print("\n".join(output_lines_with_header))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in range(0, len(output_lines)):
            parts = [x for x in output_lines[i].split(' ') if x]
            self.assertEqual(len(parts), 7)
            self.assertEqual(expect_lines[i], parts)

    def test_json_diff_report_output(self):
        expected_output = [
            {
                'name': 'BM_SameTimes',
                'measurements': [{'time': 0.0000, 'cpu': 0.0000, 'real_time': 10, 'real_time_other': 10, 'cpu_time': 10, 'cpu_time_other': 10}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_2xFaster',
                'measurements': [{'time': -0.5000, 'cpu': -0.5000, 'real_time': 50, 'real_time_other': 25, 'cpu_time': 50, 'cpu_time_other': 25}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_2xSlower',
                'measurements': [{'time': 1.0000, 'cpu': 1.0000, 'real_time': 50, 'real_time_other': 100, 'cpu_time': 50, 'cpu_time_other': 100}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_1PercentFaster',
                'measurements': [{'time': -0.0100, 'cpu': -0.0100, 'real_time': 100, 'real_time_other': 98.9999999, 'cpu_time': 100, 'cpu_time_other': 98.9999999}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_1PercentSlower',
                'measurements': [{'time': 0.0100, 'cpu': 0.0100, 'real_time': 100, 'real_time_other': 101, 'cpu_time': 100, 'cpu_time_other': 101}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_10PercentFaster',
                'measurements': [{'time': -0.1000, 'cpu': -0.1000, 'real_time': 100, 'real_time_other': 90, 'cpu_time': 100, 'cpu_time_other': 90}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_10PercentSlower',
                'measurements': [{'time': 0.1000, 'cpu': 0.1000, 'real_time': 100, 'real_time_other': 110, 'cpu_time': 100, 'cpu_time_other': 110}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_100xSlower',
                'measurements': [{'time': 99.0000, 'cpu': 99.0000, 'real_time': 100, 'real_time_other': 10000, 'cpu_time': 100, 'cpu_time_other': 10000}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_100xFaster',
                'measurements': [{'time': -0.9900, 'cpu': -0.9900, 'real_time': 10000, 'real_time_other': 100, 'cpu_time': 10000, 'cpu_time_other': 100}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_10PercentCPUToTime',
                'measurements': [{'time': 0.1000, 'cpu': -0.1000, 'real_time': 100, 'real_time_other': 110, 'cpu_time': 100, 'cpu_time_other': 90}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_ThirdFaster',
                'measurements': [{'time': -0.3333, 'cpu': -0.3334, 'real_time': 100, 'real_time_other': 67, 'cpu_time': 100, 'cpu_time_other': 67}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': 'BM_NotBadTimeUnit',
                'measurements': [{'time': -0.9000, 'cpu': 0.2000, 'real_time': 0.4, 'real_time_other': 0.04, 'cpu_time': 0.5, 'cpu_time_other': 0.6}],
                'time_unit': 's',
                'utest': {}
            },
        ]
        self.assertEqual(len(self.json_diff_report), len(expected_output))
        for out, expected in zip(
                self.json_diff_report, expected_output):
            self.assertEqual(out['name'], expected['name'])
            self.assertEqual(out['time_unit'], expected['time_unit'])
            assert_utest(self, out, expected)
            assert_measurements(self, out, expected)


class TestReportDifferenceBetweenFamilies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load_result():
            import json
            testInputs = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                'Inputs')
            testOutput = os.path.join(testInputs, 'test2_run.json')
            with open(testOutput, 'r') as f:
                json = json.load(f)
            return json

        json = load_result()
        json1 = filter_benchmark(json, "BM_Z.ro", ".")
        json2 = filter_benchmark(json, "BM_O.e", ".")
        cls.json_diff_report = get_difference_report(json1, json2)

    def test_json_diff_report_pretty_printing(self):
        expect_lines = [
            ['.', '-0.5000', '-0.5000', '10', '5', '10', '5'],
            ['./4', '-0.5000', '-0.5000', '40', '20', '40', '20'],
            ['Prefix/.', '-0.5000', '-0.5000', '20', '10', '20', '10'],
            ['Prefix/./3', '-0.5000', '-0.5000', '30', '15', '30', '15'],
        ]
        output_lines_with_header = print_difference_report(
            self.json_diff_report, use_color=False)
        output_lines = output_lines_with_header[2:]
        print("\n")
        print("\n".join(output_lines_with_header))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in range(0, len(output_lines)):
            parts = [x for x in output_lines[i].split(' ') if x]
            self.assertEqual(len(parts), 7)
            self.assertEqual(expect_lines[i], parts)

    def test_json_diff_report(self):
        expected_output = [
            {
                'name': u'.',
                'measurements': [{'time': -0.5, 'cpu': -0.5, 'real_time': 10, 'real_time_other': 5, 'cpu_time': 10, 'cpu_time_other': 5}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': u'./4',
                'measurements': [{'time': -0.5, 'cpu': -0.5, 'real_time': 40, 'real_time_other': 20, 'cpu_time': 40, 'cpu_time_other': 20}],
                'time_unit': 'ns',
                'utest': {},
            },
            {
                'name': u'Prefix/.',
                'measurements': [{'time': -0.5, 'cpu': -0.5, 'real_time': 20, 'real_time_other': 10, 'cpu_time': 20, 'cpu_time_other': 10}],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': u'Prefix/./3',
                'measurements': [{'time': -0.5, 'cpu': -0.5, 'real_time': 30, 'real_time_other': 15, 'cpu_time': 30, 'cpu_time_other': 15}],
                'time_unit': 'ns',
                'utest': {}
            }
        ]
        self.assertEqual(len(self.json_diff_report), len(expected_output))
        for out, expected in zip(
                self.json_diff_report, expected_output):
            self.assertEqual(out['name'], expected['name'])
            self.assertEqual(out['time_unit'], expected['time_unit'])
            assert_utest(self, out, expected)
            assert_measurements(self, out, expected)


class TestReportDifferenceWithUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load_results():
            import json
            testInputs = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                'Inputs')
            testOutput1 = os.path.join(testInputs, 'test3_run0.json')
            testOutput2 = os.path.join(testInputs, 'test3_run1.json')
            with open(testOutput1, 'r') as f:
                json1 = json.load(f)
            with open(testOutput2, 'r') as f:
                json2 = json.load(f)
            return json1, json2

        json1, json2 = load_results()
        cls.json_diff_report = get_difference_report(
            json1, json2, utest=True)

    def test_json_diff_report_pretty_printing(self):
        expect_lines = [
            ['BM_One', '-0.1000', '+0.1000', '10', '9', '100', '110'],
            ['BM_Two', '+0.1111', '-0.0111', '9', '10', '90', '89'],
            ['BM_Two', '-0.1250', '-0.1628', '8', '7', '86', '72'],
            ['BM_Two_pvalue',
             '0.6985',
             '0.6985',
             'U',
             'Test,',
             'Repetitions:',
             '2',
             'vs',
             '2.',
             'WARNING:',
             'Results',
             'unreliable!',
             '9+',
             'repetitions',
             'recommended.'],
            ['short', '-0.1250', '-0.0625', '8', '7', '80', '75'],
            ['short', '-0.4325', '-0.1351', '8', '5', '77', '67'],
            ['short_pvalue',
             '0.7671',
             '0.1489',
             'U',
             'Test,',
             'Repetitions:',
             '2',
             'vs',
             '3.',
             'WARNING:',
             'Results',
             'unreliable!',
             '9+',
             'repetitions',
             'recommended.'],
            ['medium', '-0.3750', '-0.3375', '8', '5', '80', '53'],
        ]
        output_lines_with_header = print_difference_report(
            self.json_diff_report, utest=True, utest_alpha=0.05, use_color=False)
        output_lines = output_lines_with_header[2:]
        print("\n")
        print("\n".join(output_lines_with_header))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in range(0, len(output_lines)):
            parts = [x for x in output_lines[i].split(' ') if x]
            self.assertEqual(expect_lines[i], parts)

    def test_json_diff_report_pretty_printing_aggregates_only(self):
        expect_lines = [
            ['BM_One', '-0.1000', '+0.1000', '10', '9', '100', '110'],
            ['BM_Two_pvalue',
             '0.6985',
             '0.6985',
             'U',
             'Test,',
             'Repetitions:',
             '2',
             'vs',
             '2.',
             'WARNING:',
             'Results',
             'unreliable!',
             '9+',
             'repetitions',
             'recommended.'],
            ['short', '-0.1250', '-0.0625', '8', '7', '80', '75'],
            ['short', '-0.4325', '-0.1351', '8', '5', '77', '67'],
            ['short_pvalue',
             '0.7671',
             '0.1489',
             'U',
             'Test,',
             'Repetitions:',
             '2',
             'vs',
             '3.',
             'WARNING:',
             'Results',
             'unreliable!',
             '9+',
             'repetitions',
             'recommended.'],
        ]
        output_lines_with_header = print_difference_report(
            self.json_diff_report, include_aggregates_only=True, utest=True, utest_alpha=0.05, use_color=False)
        output_lines = output_lines_with_header[2:]
        print("\n")
        print("\n".join(output_lines_with_header))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in range(0, len(output_lines)):
            parts = [x for x in output_lines[i].split(' ') if x]
            self.assertEqual(expect_lines[i], parts)

    def test_json_diff_report(self):
        expected_output = [
            {
                'name': u'BM_One',
                'measurements': [
                    {'time': -0.1,
                     'cpu': 0.1,
                     'real_time': 10,
                     'real_time_other': 9,
                     'cpu_time': 100,
                     'cpu_time_other': 110}
                ],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': u'BM_Two',
                'measurements': [
                    {'time': 0.1111111111111111,
                     'cpu': -0.011111111111111112,
                     'real_time': 9,
                     'real_time_other': 10,
                     'cpu_time': 90,
                     'cpu_time_other': 89},
                    {'time': -0.125, 'cpu': -0.16279069767441862, 'real_time': 8,
                        'real_time_other': 7, 'cpu_time': 86, 'cpu_time_other': 72}
                ],
                'time_unit': 'ns',
                'utest': {
                    'have_optimal_repetitions': False, 'cpu_pvalue': 0.6985353583033387, 'time_pvalue': 0.6985353583033387
                }
            },
            {
                'name': u'short',
                'measurements': [
                    {'time': -0.125,
                     'cpu': -0.0625,
                     'real_time': 8,
                     'real_time_other': 7,
                     'cpu_time': 80,
                     'cpu_time_other': 75},
                    {'time': -0.4325,
                     'cpu': -0.13506493506493514,
                     'real_time': 8,
                     'real_time_other': 4.54,
                     'cpu_time': 77,
                     'cpu_time_other': 66.6}
                ],
                'time_unit': 'ns',
                'utest': {
                    'have_optimal_repetitions': False, 'cpu_pvalue': 0.14891467317876572, 'time_pvalue': 0.7670968684102772
                }
            },
            {
                'name': u'medium',
                'measurements': [
                    {'time': -0.375,
                     'cpu': -0.3375,
                     'real_time': 8,
                     'real_time_other': 5,
                     'cpu_time': 80,
                     'cpu_time_other': 53}
                ],
                'time_unit': 'ns',
                'utest': {}
            }
        ]
        self.assertEqual(len(self.json_diff_report), len(expected_output))
        for out, expected in zip(
                self.json_diff_report, expected_output):
            self.assertEqual(out['name'], expected['name'])
            self.assertEqual(out['time_unit'], expected['time_unit'])
            assert_utest(self, out, expected)
            assert_measurements(self, out, expected)


class TestReportDifferenceWithUTestWhileDisplayingAggregatesOnly(
        unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load_results():
            import json
            testInputs = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                'Inputs')
            testOutput1 = os.path.join(testInputs, 'test3_run0.json')
            testOutput2 = os.path.join(testInputs, 'test3_run1.json')
            with open(testOutput1, 'r') as f:
                json1 = json.load(f)
            with open(testOutput2, 'r') as f:
                json2 = json.load(f)
            return json1, json2

        json1, json2 = load_results()
        cls.json_diff_report = get_difference_report(
            json1, json2, utest=True)

    def test_json_diff_report_pretty_printing(self):
        expect_lines = [
            ['BM_One', '-0.1000', '+0.1000', '10', '9', '100', '110'],
            ['BM_Two', '+0.1111', '-0.0111', '9', '10', '90', '89'],
            ['BM_Two', '-0.1250', '-0.1628', '8', '7', '86', '72'],
            ['BM_Two_pvalue',
             '0.6985',
             '0.6985',
             'U',
             'Test,',
             'Repetitions:',
             '2',
             'vs',
             '2.',
             'WARNING:',
             'Results',
             'unreliable!',
             '9+',
             'repetitions',
             'recommended.'],
            ['short', '-0.1250', '-0.0625', '8', '7', '80', '75'],
            ['short', '-0.4325', '-0.1351', '8', '5', '77', '67'],
            ['short_pvalue',
             '0.7671',
             '0.1489',
             'U',
             'Test,',
             'Repetitions:',
             '2',
             'vs',
             '3.',
             'WARNING:',
             'Results',
             'unreliable!',
             '9+',
             'repetitions',
             'recommended.'],
             ['medium', '-0.3750', '-0.3375', '8', '5', '80', '53']
        ]
        output_lines_with_header = print_difference_report(
            self.json_diff_report,
            utest=True, utest_alpha=0.05, use_color=False)
        output_lines = output_lines_with_header[2:]
        print("\n")
        print("\n".join(output_lines_with_header))
        self.assertEqual(len(output_lines), len(expect_lines))
        for i in range(0, len(output_lines)):
            parts = [x for x in output_lines[i].split(' ') if x]
            self.assertEqual(expect_lines[i], parts)

    def test_json_diff_report(self):
        expected_output = [
            {
                'name': u'BM_One',
                'measurements': [
                    {'time': -0.1,
                     'cpu': 0.1,
                     'real_time': 10,
                     'real_time_other': 9,
                     'cpu_time': 100,
                     'cpu_time_other': 110}
                ],
                'time_unit': 'ns',
                'utest': {}
            },
            {
                'name': u'BM_Two',
                'measurements': [
                    {'time': 0.1111111111111111,
                     'cpu': -0.011111111111111112,
                     'real_time': 9,
                     'real_time_other': 10,
                     'cpu_time': 90,
                     'cpu_time_other': 89},
                    {'time': -0.125, 'cpu': -0.16279069767441862, 'real_time': 8,
                        'real_time_other': 7, 'cpu_time': 86, 'cpu_time_other': 72}
                ],
                'time_unit': 'ns',
                'utest': {
                    'have_optimal_repetitions': False, 'cpu_pvalue': 0.6985353583033387, 'time_pvalue': 0.6985353583033387
                }
            },
            {
                'name': u'short',
                'measurements': [
                    {'time': -0.125,
                     'cpu': -0.0625,
                     'real_time': 8,
                     'real_time_other': 7,
                     'cpu_time': 80,
                     'cpu_time_other': 75},
                    {'time': -0.4325,
                     'cpu': -0.13506493506493514,
                     'real_time': 8,
                     'real_time_other': 4.54,
                     'cpu_time': 77,
                     'cpu_time_other': 66.6}
                ],
                'time_unit': 'ns',
                'utest': {
                    'have_optimal_repetitions': False, 'cpu_pvalue': 0.14891467317876572, 'time_pvalue': 0.7670968684102772
                }
            },
            {
                'name': u'medium',
                'measurements': [
                    {'real_time_other': 5,
                     'cpu_time': 80,
                     'time': -0.375,
                     'real_time': 8,
                     'cpu_time_other': 53,
                     'cpu': -0.3375
                    }
                ],
                'utest': {},
                'time_unit': u'ns',
                'aggregate_name': ''
            }
        ]
        self.assertEqual(len(self.json_diff_report), len(expected_output))
        for out, expected in zip(
                self.json_diff_report, expected_output):
            self.assertEqual(out['name'], expected['name'])
            self.assertEqual(out['time_unit'], expected['time_unit'])
            assert_utest(self, out, expected)
            assert_measurements(self, out, expected)


class TestReportSorting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load_result():
            import json
            testInputs = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                'Inputs')
            testOutput = os.path.join(testInputs, 'test4_run.json')
            with open(testOutput, 'r') as f:
                json = json.load(f)
            return json

        cls.json = load_result()

    def test_json_diff_report_pretty_printing(self):
        import util

        expected_names = [
            "99 family 0 instance 0 repetition 0",
            "98 family 0 instance 0 repetition 1",
            "97 family 0 instance 0 aggregate",
            "96 family 0 instance 1 repetition 0",
            "95 family 0 instance 1 repetition 1",
            "94 family 0 instance 1 aggregate",
            "93 family 1 instance 0 repetition 0",
            "92 family 1 instance 0 repetition 1",
            "91 family 1 instance 0 aggregate",
            "90 family 1 instance 1 repetition 0",
            "89 family 1 instance 1 repetition 1",
            "88 family 1 instance 1 aggregate"
        ]

        for n in range(len(self.json['benchmarks']) ** 2):
            random.shuffle(self.json['benchmarks'])
            sorted_benchmarks = util.sort_benchmark_results(self.json)[
                'benchmarks']
            self.assertEqual(len(expected_names), len(sorted_benchmarks))
            for out, expected in zip(sorted_benchmarks, expected_names):
                self.assertEqual(out['name'], expected)


def assert_utest(unittest_instance, lhs, rhs):
    if lhs['utest']:
        unittest_instance.assertAlmostEqual(
            lhs['utest']['cpu_pvalue'],
            rhs['utest']['cpu_pvalue'])
        unittest_instance.assertAlmostEqual(
            lhs['utest']['time_pvalue'],
            rhs['utest']['time_pvalue'])
        unittest_instance.assertEqual(
            lhs['utest']['have_optimal_repetitions'],
            rhs['utest']['have_optimal_repetitions'])
    else:
        # lhs is empty. assert if rhs is not.
        unittest_instance.assertEqual(lhs['utest'], rhs['utest'])


def assert_measurements(unittest_instance, lhs, rhs):
    for m1, m2 in zip(lhs['measurements'], rhs['measurements']):
        unittest_instance.assertEqual(m1['real_time'], m2['real_time'])
        unittest_instance.assertEqual(m1['cpu_time'], m2['cpu_time'])
        # m1['time'] and m1['cpu'] hold values which are being calculated,
        # and therefore we must use almost-equal pattern.
        unittest_instance.assertAlmostEqual(m1['time'], m2['time'], places=4)
        unittest_instance.assertAlmostEqual(m1['cpu'], m2['cpu'], places=4)


if __name__ == '__main__':
    unittest.main()

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# kate: tab-width: 4; replace-tabs on; indent-width 4; tab-indents: off;
# kate: indent-mode python; remove-trailing-spaces modified;
