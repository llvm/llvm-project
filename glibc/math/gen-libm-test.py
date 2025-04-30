#!/usr/bin/python3
# Generate tests for libm functions.
# Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

import argparse
from collections import defaultdict
import os
import re


# Sorted list of all float types in ulps files.
ALL_FLOATS = ('double', 'float', 'float128', 'ldouble')

# Map float types in ulps files to C-like prefix for macros.
ALL_FLOATS_PFX = {'double': 'DBL',
                  'ldouble': 'LDBL',
                  'float': 'FLT',
                  'float128': 'FLT128'}

# Float types in the order used in the generated ulps tables in the
# manual.
ALL_FLOATS_MANUAL = ('float', 'double', 'ldouble', 'float128')

# Map float types in ulps files to C function suffix.
ALL_FLOATS_SUFFIX = {'double': '',
                     'ldouble': 'l',
                     'float': 'f',
                     'float128': 'f128'}

# Number of arguments in structure (as opposed to arguments that are
# pointers to return values) for an argument descriptor.
DESCR_NUM_ARGS = {'f': 1, 'a': 1, 'j': 1, 'i': 1, 'u': 1, 'l': 1, 'L': 1,
                  'p': 0, 'F': 0, 'I': 0,
                  'c': 2}

# Number of results in structure for a result descriptor.
DESCR_NUM_RES = {'f': 1, 'i': 1, 'l': 1, 'L': 1, 'M': 1, 'U': 1, 'b': 1,
                 '1': 1,
                 'c': 2}

# Rounding modes, in the form in which they appear in
# auto-libm-test-out-* and the order in which expected results appear
# in structures and TEST_* calls.
ROUNDING_MODES = ('downward', 'tonearest', 'towardzero', 'upward')

# Map from special text in TEST_* calls for rounding-mode-specific
# results and flags, to those results for each mode.
ROUNDING_MAP = {
    'plus_oflow': ('max_value', 'plus_infty', 'max_value', 'plus_infty'),
    'minus_oflow': ('minus_infty', 'minus_infty', '-max_value', '-max_value'),
    'plus_uflow': ('plus_zero', 'plus_zero', 'plus_zero', 'min_subnorm_value'),
    'minus_uflow': ('-min_subnorm_value', 'minus_zero', 'minus_zero',
                    'minus_zero'),
    'ERRNO_PLUS_OFLOW': ('0', 'ERRNO_ERANGE', '0', 'ERRNO_ERANGE'),
    'ERRNO_MINUS_OFLOW': ('ERRNO_ERANGE', 'ERRNO_ERANGE', '0', '0'),
    'ERRNO_PLUS_UFLOW': ('ERRNO_ERANGE', 'ERRNO_ERANGE', 'ERRNO_ERANGE', '0'),
    'ERRNO_MINUS_UFLOW': ('0', 'ERRNO_ERANGE', 'ERRNO_ERANGE', 'ERRNO_ERANGE'),
    'XFAIL_ROUNDING_IBM128_LIBGCC': ('XFAIL_IBM128_LIBGCC', '0',
                                     'XFAIL_IBM128_LIBGCC',
                                     'XFAIL_IBM128_LIBGCC')
    }

# Map from raw test arguments to a nicer form to use when displaying
# test results.
BEAUTIFY_MAP = {'minus_zero': '-0',
                'plus_zero': '+0',
                '-0x0p+0f': '-0',
                '-0x0p+0': '-0',
                '-0x0p+0L': '-0',
                '0x0p+0f': '+0',
                '0x0p+0': '+0',
                '0x0p+0L': '+0',
                'minus_infty': '-inf',
                'plus_infty': 'inf',
                'qnan_value': 'qNaN',
                'snan_value': 'sNaN',
                'snan_value_ld': 'sNaN'}

# Flags in auto-libm-test-out that map directly to C flags.
FLAGS_SIMPLE = {'ignore-zero-inf-sign': 'IGNORE_ZERO_INF_SIGN',
                'xfail': 'XFAIL_TEST'}

# Exceptions in auto-libm-test-out, and their corresponding C flags
# for being required, OK or required to be absent.
EXC_EXPECTED = {'divbyzero': 'DIVBYZERO_EXCEPTION',
                'inexact': 'INEXACT_EXCEPTION',
                'invalid': 'INVALID_EXCEPTION',
                'overflow': 'OVERFLOW_EXCEPTION',
                'underflow': 'UNDERFLOW_EXCEPTION'}
EXC_OK = {'divbyzero': 'DIVBYZERO_EXCEPTION_OK',
          'inexact': '0',
          'invalid': 'INVALID_EXCEPTION_OK',
          'overflow': 'OVERFLOW_EXCEPTION_OK',
          'underflow': 'UNDERFLOW_EXCEPTION_OK'}
EXC_NO = {'divbyzero': '0',
          'inexact': 'NO_INEXACT_EXCEPTION',
          'invalid': '0',
          'overflow': '0',
          'underflow': '0'}


class Ulps(object):
    """Maximum expected errors of libm functions."""

    def __init__(self):
        """Initialize an Ulps object."""
        # normal[function][float_type] is the ulps value, and likewise
        # for real and imag.
        self.normal = defaultdict(lambda: defaultdict(lambda: 0))
        self.real = defaultdict(lambda: defaultdict(lambda: 0))
        self.imag = defaultdict(lambda: defaultdict(lambda: 0))
        # List of ulps kinds, in the order in which they appear in
        # sorted ulps files.
        self.ulps_kinds = (('Real part of ', self.real),
                           ('Imaginary part of ', self.imag),
                           ('', self.normal))
        self

    def read(self, ulps_file):
        """Read ulps from a file into an Ulps object."""
        self.ulps_file = ulps_file
        with open(ulps_file, 'r') as f:
            ulps_dict = None
            ulps_fn = None
            for line in f:
                # Ignore comments.
                if line.startswith('#'):
                    continue
                line = line.rstrip()
                # Ignore empty lines.
                if line == '':
                    continue
                m = re.match(r'([^:]*): (.*)\Z', line)
                if not m:
                    raise ValueError('bad ulps line: %s' % line)
                line_first = m.group(1)
                line_second = m.group(2)
                if line_first == 'Function':
                    fn = None
                    ulps_dict = None
                    for k_prefix, k_dict in self.ulps_kinds:
                        if line_second.startswith(k_prefix):
                            ulps_dict = k_dict
                            fn = line_second[len(k_prefix):]
                            break
                    if not fn.startswith('"') or not fn.endswith('":'):
                        raise ValueError('bad ulps line: %s' % line)
                    ulps_fn = fn[1:-2]
                else:
                    if line_first not in ALL_FLOATS:
                        raise ValueError('bad ulps line: %s' % line)
                    ulps_val = int(line_second)
                    if ulps_val > 0:
                        ulps_dict[ulps_fn][line_first] = max(
                            ulps_dict[ulps_fn][line_first],
                            ulps_val)

    def all_functions(self):
        """Return the set of functions with ulps and whether they are
        complex."""
        funcs = set()
        complex = {}
        for k_prefix, k_dict in self.ulps_kinds:
            for f in k_dict:
                funcs.add(f)
                complex[f] = True if k_prefix else False
        return funcs, complex

    def write(self, ulps_file):
        """Write ulps back out as a sorted ulps file."""
        # Output is sorted first by function name, then by (real,
        # imag, normal), then by float type.
        out_data = {}
        for order, (prefix, d) in enumerate(self.ulps_kinds):
            for fn in d.keys():
                fn_data = ['%s: %d' % (f, d[fn][f])
                           for f in sorted(d[fn].keys())]
                fn_text = 'Function: %s"%s":\n%s' % (prefix, fn,
                                                     '\n'.join(fn_data))
                out_data[(fn, order)] = fn_text
        out_list = [out_data[fn_order] for fn_order in sorted(out_data.keys())]
        out_text = ('# Begin of automatic generation\n\n'
                    '# Maximal error of functions:\n'
                    '%s\n\n'
                    '# end of automatic generation\n'
                    % '\n\n'.join(out_list))
        with open(ulps_file, 'w') as f:
            f.write(out_text)

    @staticmethod
    def ulps_table(name, ulps_dict):
        """Return text of a C table of ulps."""
        ulps_list = []
        for fn in sorted(ulps_dict.keys()):
            fn_ulps = [str(ulps_dict[fn][f]) for f in ALL_FLOATS]
            ulps_list.append('    { "%s", {%s} },' % (fn, ', '.join(fn_ulps)))
        ulps_text = ('static const struct ulp_data %s[] =\n'
                     '  {\n'
                     '%s\n'
                     '  };'
                     % (name, '\n'.join(ulps_list)))
        return ulps_text

    def write_header(self, ulps_header):
        """Write header file with ulps data."""
        header_text_1 = ('/* This file is automatically generated\n'
                         '   from %s with gen-libm-test.py.\n'
                         '   Don\'t change it - change instead the master '
                         'files.  */\n\n'
                         'struct ulp_data\n'
                         '{\n'
                         '  const char *name;\n'
                         '  FLOAT max_ulp[%d];\n'
                         '};'
                         % (self.ulps_file, len(ALL_FLOATS)))
        macro_list = []
        for i, f in enumerate(ALL_FLOATS):
            if f.startswith('i'):
                itxt = 'I_'
                f = f[1:]
            else:
                itxt = ''
            macro_list.append('#define ULP_%s%s %d'
                              % (itxt, ALL_FLOATS_PFX[f], i))
        header_text = ('%s\n\n'
                       '%s\n\n'
                       '/* Maximal error of functions.  */\n'
                       '%s\n'
                       '%s\n'
                       '%s\n'
                       % (header_text_1, '\n'.join(macro_list),
                          self.ulps_table('func_ulps', self.normal),
                          self.ulps_table('func_real_ulps', self.real),
                          self.ulps_table('func_imag_ulps', self.imag)))
        with open(ulps_header, 'w') as f:
            f.write(header_text)


def read_all_ulps(srcdir):
    """Read all platforms' libm-test-ulps files."""
    all_ulps = {}
    for dirpath, dirnames, filenames in os.walk(srcdir):
        if 'libm-test-ulps' in filenames:
            with open(os.path.join(dirpath, 'libm-test-ulps-name')) as f:
                name = f.read().rstrip()
            all_ulps[name] = Ulps()
            all_ulps[name].read(os.path.join(dirpath, 'libm-test-ulps'))
    return all_ulps


def read_auto_tests(test_file):
    """Read tests from auto-libm-test-out-<function> (possibly None)."""
    auto_tests = defaultdict(lambda: defaultdict(dict))
    if test_file is None:
        return auto_tests
    with open(test_file, 'r') as f:
        for line in f:
            if not line.startswith('= '):
                continue
            line = line[len('= '):].rstrip()
            # Function, rounding mode, condition and inputs, outputs
            # and flags.
            m = re.match(r'([^ ]+) ([^ ]+) ([^: ][^ ]* [^:]*) : (.*)\Z', line)
            if not m:
                raise ValueError('bad automatic test line: %s' % line)
            auto_tests[m.group(1)][m.group(2)][m.group(3)] = m.group(4)
    return auto_tests


def beautify(arg):
    """Return a nicer representation of a test argument."""
    if arg in BEAUTIFY_MAP:
        return BEAUTIFY_MAP[arg]
    if arg.startswith('-') and arg[1:] in BEAUTIFY_MAP:
        return '-' + BEAUTIFY_MAP[arg[1:]]
    if re.match(r'-?0x[0-9a-f.]*p[-+][0-9]+f\Z', arg):
        return arg[:-1]
    if re.search(r'[0-9]L\Z', arg):
        return arg[:-1]
    return arg


def complex_beautify(arg_real, arg_imag):
    """Return a nicer representation of a complex test argument."""
    res_real = beautify(arg_real)
    res_imag = beautify(arg_imag)
    if res_imag.startswith('-'):
        return '%s - %s i' % (res_real, res_imag[1:])
    else:
        return '%s + %s i' % (res_real, res_imag)


def apply_lit_token(arg, macro):
    """Apply the LIT or ARG_LIT macro to a single token."""
    # The macro must only be applied to a floating-point constant, not
    # to an integer constant or lit_* value.
    sign_re = r'[+-]?'
    exp_re = r'([+-])?[0-9]+'
    suffix_re = r'[lLfF]?'
    dec_exp_re = r'[eE]' + exp_re
    hex_exp_re = r'[pP]' + exp_re
    dec_frac_re = r'(?:[0-9]*\.[0-9]+|[0-9]+\.)'
    hex_frac_re = r'(?:[0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.)'
    dec_int_re = r'[0-9]+'
    hex_int_re = r'[0-9a-fA-F]+'
    dec_cst_re = r'(?:%s(?:%s)?|%s%s)' % (dec_frac_re, dec_exp_re,
                                          dec_int_re, dec_exp_re)
    hex_cst_re = r'0[xX](?:%s|%s)%s' % (hex_frac_re, hex_int_re, hex_exp_re)
    fp_cst_re = r'(%s(?:%s|%s))%s\Z' % (sign_re, dec_cst_re, hex_cst_re,
                                        suffix_re)
    m = re.match(fp_cst_re, arg)
    if m:
        return '%s (%s)' % (macro, m.group(1))
    else:
        return arg


def apply_lit(arg, macro):
    """Apply the LIT or ARG_LIT macro to constants within an expression."""
    # Assume expressions follow the GNU Coding Standards, with tokens
    # separated by spaces.
    return ' '.join([apply_lit_token(t, macro) for t in arg.split()])


def gen_test_args_res(descr_args, descr_res, args, res_rm):
    """Generate a test given the arguments and per-rounding-mode results."""
    test_snan = False
    all_args_res = list(args)
    for r in res_rm:
        all_args_res.extend(r[:len(r)-1])
    for a in all_args_res:
        if 'snan_value' in a:
            test_snan = True
    # Process the arguments.
    args_disp = []
    args_c = []
    arg_pos = 0
    for d in descr_args:
        if DESCR_NUM_ARGS[d] == 0:
            continue
        if d == 'c':
            args_disp.append(complex_beautify(args[arg_pos],
                                              args[arg_pos + 1]))
            args_c.append(apply_lit(args[arg_pos], 'LIT'))
            args_c.append(apply_lit(args[arg_pos + 1], 'LIT'))
        else:
            args_disp.append(beautify(args[arg_pos]))
            if d == 'f':
                args_c.append(apply_lit(args[arg_pos], 'LIT'))
            elif d == 'a':
                args_c.append(apply_lit(args[arg_pos], 'ARG_LIT'))
            else:
                args_c.append(args[arg_pos])
        arg_pos += DESCR_NUM_ARGS[d]
    args_disp_text = ', '.join(args_disp).replace('"', '\\"')
    # Process the results.
    for rm in range(len(ROUNDING_MODES)):
        res = res_rm[rm]
        res_pos = 0
        rm_args = []
        ignore_result_any = False
        ignore_result_all = True
        special = []
        for d in descr_res:
            if d == '1':
                special.append(res[res_pos])
            elif DESCR_NUM_RES[d] == 1:
                result = res[res_pos]
                if result == 'IGNORE':
                    ignore_result_any = True
                    result = '0'
                else:
                    ignore_result_all = False
                    if d == 'f':
                        result = apply_lit(result, 'LIT')
                rm_args.append(result)
            else:
                # Complex result.
                result1 = res[res_pos]
                if result1 == 'IGNORE':
                    ignore_result_any = True
                    result1 = '0'
                else:
                    ignore_result_all = False
                    result1 = apply_lit(result1, 'LIT')
                rm_args.append(result1)
                result2 = res[res_pos + 1]
                if result2 == 'IGNORE':
                    ignore_result_any = True
                    result2 = '0'
                else:
                    ignore_result_all = False
                    result2 = apply_lit(result2, 'LIT')
                rm_args.append(result2)
            res_pos += DESCR_NUM_RES[d]
        if ignore_result_any and not ignore_result_all:
            raise ValueError('some but not all function results ignored')
        flags = []
        if ignore_result_any:
            flags.append('IGNORE_RESULT')
        if test_snan:
            flags.append('TEST_SNAN')
        flags.append(res[res_pos])
        rm_args.append('|'.join(flags))
        for sp in special:
            if sp == 'IGNORE':
                rm_args.extend(['0', '0'])
            else:
                rm_args.extend(['1', apply_lit(sp, 'LIT')])
        for k in sorted(ROUNDING_MAP.keys()):
            rm_args = [arg.replace(k, ROUNDING_MAP[k][rm]) for arg in rm_args]
        args_c.append('{ %s }' % ', '.join(rm_args))
    return '    { "%s", %s },\n' % (args_disp_text, ', '.join(args_c))


def convert_condition(cond):
    """Convert a condition from auto-libm-test-out to C form."""
    conds = cond.split(':')
    conds_c = []
    for c in conds:
        if not c.startswith('arg_fmt('):
            c = c.replace('-', '_')
        conds_c.append('TEST_COND_' + c)
    return '(%s)' % ' && '.join(conds_c)


def cond_value(cond, if_val, else_val):
    """Return a C conditional expression between two values."""
    if cond == '1':
        return if_val
    elif cond == '0':
        return else_val
    else:
        return '(%s ? %s : %s)' % (cond, if_val, else_val)


def gen_auto_tests(auto_tests, descr_args, descr_res, fn):
    """Generate C code for the auto-libm-test-out-* tests for a function."""
    for rm_idx, rm_name in enumerate(ROUNDING_MODES):
        this_tests = sorted(auto_tests[fn][rm_name].keys())
        if rm_idx == 0:
            rm_tests = this_tests
            if not rm_tests:
                raise ValueError('no automatic tests for %s' % fn)
        else:
            if rm_tests != this_tests:
                raise ValueError('inconsistent lists of tests of %s' % fn)
    test_list = []
    for test in rm_tests:
        fmt_args = test.split()
        fmt = fmt_args[0]
        args = fmt_args[1:]
        test_list.append('#if %s\n' % convert_condition(fmt))
        res_rm = []
        for rm in ROUNDING_MODES:
            test_out = auto_tests[fn][rm][test]
            out_str, flags_str = test_out.split(':', 1)
            this_res = out_str.split()
            flags = flags_str.split()
            flag_cond = {}
            for flag in flags:
                m = re.match(r'([^:]*):(.*)\Z', flag)
                if m:
                    f_name = m.group(1)
                    cond = convert_condition(m.group(2))
                    if f_name in flag_cond:
                        if flag_cond[f_name] != '1':
                            flag_cond[f_name] = ('%s || %s'
                                                 % (flag_cond[f_name], cond))
                    else:
                        flag_cond[f_name] = cond
                else:
                    flag_cond[flag] = '1'
            flags_c = []
            for flag in sorted(FLAGS_SIMPLE.keys()):
                if flag in flag_cond:
                    flags_c.append(cond_value(flag_cond[flag],
                                              FLAGS_SIMPLE[flag], '0'))
            for exc in sorted(EXC_EXPECTED.keys()):
                exc_expected = EXC_EXPECTED[exc]
                exc_ok = EXC_OK[exc]
                no_exc = EXC_NO[exc]
                exc_cond = flag_cond.get(exc, '0')
                exc_ok_cond = flag_cond.get(exc + '-ok', '0')
                flags_c.append(cond_value(exc_cond,
                                          cond_value(exc_ok_cond, exc_ok,
                                                     exc_expected),
                                          cond_value(exc_ok_cond, exc_ok,
                                                     no_exc)))
            if 'errno-edom' in flag_cond and 'errno-erange' in flag_cond:
                raise ValueError('multiple errno values expected')
            if 'errno-edom' in flag_cond:
                if flag_cond['errno-edom'] != '1':
                    raise ValueError('unexpected condition for errno-edom')
                errno_expected = 'ERRNO_EDOM'
            elif 'errno-erange' in flag_cond:
                if flag_cond['errno-erange'] != '1':
                    raise ValueError('unexpected condition for errno-erange')
                errno_expected = 'ERRNO_ERANGE'
            else:
                errno_expected = 'ERRNO_UNCHANGED'
            if 'errno-edom-ok' in flag_cond:
                if ('errno-erange-ok' in flag_cond
                    and (flag_cond['errno-erange-ok']
                         != flag_cond['errno-edom-ok'])):
                    errno_unknown_cond = ('%s || %s'
                                          % (flag_cond['errno-edom-ok'],
                                             flag_cond['errno-erange-ok']))
                else:
                    errno_unknown_cond = flag_cond['errno-edom-ok']
            else:
                errno_unknown_cond = flag_cond.get('errno-erange-ok', '0')
            flags_c.append(cond_value(errno_unknown_cond, '0', errno_expected))
            flags_c = [flag for flag in flags_c if flag != '0']
            if not flags_c:
                flags_c = ['NO_EXCEPTION']
            this_res.append(' | '.join(flags_c))
            res_rm.append(this_res)
        test_list.append(gen_test_args_res(descr_args, descr_res, args,
                                           res_rm))
        test_list.append('#endif\n')
    return ''.join(test_list)


def gen_test_line(descr_args, descr_res, args_str):
    """Generate C code for the tests for a single TEST_* line."""
    test_args = args_str.split(',')
    test_args = test_args[1:]
    test_args = [a.strip() for a in test_args]
    num_args = sum([DESCR_NUM_ARGS[c] for c in descr_args])
    num_res = sum([DESCR_NUM_RES[c] for c in descr_res])
    args = test_args[:num_args]
    res = test_args[num_args:]
    if len(res) == num_res:
        # One set of results for all rounding modes, no flags.
        res.append('0')
        res_rm = [res, res, res, res]
    elif len(res) == num_res + 1:
        # One set of results for all rounding modes, with flags.
        if not ('EXCEPTION' in res[-1]
                or 'ERRNO' in res[-1]
                or 'IGNORE_ZERO_INF_SIGN' in res[-1]
                or 'TEST_NAN_SIGN' in res[-1]
                or 'XFAIL' in res[-1]):
            raise ValueError('wrong number of arguments: %s' % args_str)
        res_rm = [res, res, res, res]
    elif len(res) == (num_res + 1) * 4:
        # One set of results per rounding mode, with flags.
        nr_plus = num_res + 1
        res_rm = [res[:nr_plus], res[nr_plus:2*nr_plus],
                  res[2*nr_plus:3*nr_plus], res[3*nr_plus:]]
    return gen_test_args_res(descr_args, descr_res, args, res_rm)


def generate_testfile(inc_input, auto_tests, c_output):
    """Generate test .c file from .inc input."""
    test_list = []
    with open(inc_input, 'r') as f:
        for line in f:
            line_strip = line.strip()
            if line_strip.startswith('AUTO_TESTS_'):
                m = re.match(r'AUTO_TESTS_([^_]*)_([^_ ]*) *\(([^)]*)\),\Z',
                             line_strip)
                if not m:
                    raise ValueError('bad AUTO_TESTS line: %s' % line)
                test_list.append(gen_auto_tests(auto_tests, m.group(1),
                                                m.group(2), m.group(3)))
            elif line_strip.startswith('TEST_'):
                m = re.match(r'TEST_([^_]*)_([^_ ]*) *\((.*)\),\Z', line_strip)
                if not m:
                    raise ValueError('bad TEST line: %s' % line)
                test_list.append(gen_test_line(m.group(1), m.group(2),
                                               m.group(3)))
            else:
                test_list.append(line)
    with open(c_output, 'w') as f:
        f.write(''.join(test_list))


def generate_err_table_sub(all_ulps, all_functions, fns_complex, platforms):
    """Generate a single table within the overall ulps table section."""
    plat_width = [' {1000 + i 1000}' for p in platforms]
    plat_header = [' @tab %s' % p for p in platforms]
    table_list = ['@multitable {nexttowardf} %s\n' % ''.join(plat_width),
                  '@item Function %s\n' % ''.join(plat_header)]
    for func in all_functions:
        for flt in ALL_FLOATS_MANUAL:
            func_ulps = []
            for p in platforms:
                p_ulps = all_ulps[p]
                if fns_complex[func]:
                    ulp_real = p_ulps.real[func][flt]
                    ulp_imag = p_ulps.imag[func][flt]
                    ulp_str = '%d + i %d' % (ulp_real, ulp_imag)
                    ulp_str = ulp_str if ulp_real or ulp_imag else '-'
                else:
                    ulp = p_ulps.normal[func][flt]
                    ulp_str = str(ulp) if ulp else '-'
                func_ulps.append(ulp_str)
            table_list.append('@item %s%s  @tab %s\n'
                              % (func, ALL_FLOATS_SUFFIX[flt],
                                 ' @tab '.join(func_ulps)))
    table_list.append('@end multitable\n')
    return ''.join(table_list)


def generate_err_table(all_ulps, err_table):
    """Generate ulps table for manual."""
    all_platforms = sorted(all_ulps.keys())
    functions_set = set()
    functions_complex = {}
    for p in all_platforms:
        p_functions, p_complex = all_ulps[p].all_functions()
        functions_set.update(p_functions)
        functions_complex.update(p_complex)
    all_functions = sorted([f for f in functions_set
                            if ('_downward' not in f
                                and '_towardzero' not in f
                                and '_upward' not in f
                                and '_vlen' not in f)])
    err_table_list = []
    # Print five platforms at a time.
    num_platforms = len(all_platforms)
    for i in range((num_platforms + 4) // 5):
        start = i * 5
        end = i * 5 + 5 if num_platforms >= i * 5 + 5 else num_platforms
        err_table_list.append(generate_err_table_sub(all_ulps, all_functions,
                                                     functions_complex,
                                                     all_platforms[start:end]))
    with open(err_table, 'w') as f:
        f.write(''.join(err_table_list))


def main():
    """The main entry point."""
    parser = argparse.ArgumentParser(description='Generate libm tests.')
    parser.add_argument('-a', dest='auto_input', metavar='FILE',
                        help='input file with automatically generated tests')
    parser.add_argument('-c', dest='inc_input', metavar='FILE',
                        help='input file .inc file with tests')
    parser.add_argument('-u', dest='ulps_file', metavar='FILE',
                        help='input file with ulps')
    parser.add_argument('-s', dest='srcdir', metavar='DIR',
                        help='input source directory with all ulps')
    parser.add_argument('-n', dest='ulps_output', metavar='FILE',
                        help='generate sorted ulps file FILE')
    parser.add_argument('-C', dest='c_output', metavar='FILE',
                        help='generate output C file FILE from .inc file')
    parser.add_argument('-H', dest='ulps_header', metavar='FILE',
                        help='generate output ulps header FILE')
    parser.add_argument('-m', dest='err_table', metavar='FILE',
                        help='generate output ulps table for manual FILE')
    args = parser.parse_args()
    ulps = Ulps()
    if args.ulps_file is not None:
        ulps.read(args.ulps_file)
    auto_tests = read_auto_tests(args.auto_input)
    if args.srcdir is not None:
        all_ulps = read_all_ulps(args.srcdir)
    if args.ulps_output is not None:
        ulps.write(args.ulps_output)
    if args.ulps_header is not None:
        ulps.write_header(args.ulps_header)
    if args.c_output is not None:
        generate_testfile(args.inc_input, auto_tests, args.c_output)
    if args.err_table is not None:
        generate_err_table(all_ulps, args.err_table)


if __name__ == '__main__':
    main()
