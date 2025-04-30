#!/usr/bin/python3
# Generate tests for <tgmath.h> macros.
# Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

# As glibc does not support decimal floating point, the types to
# consider for generic parameters are standard and binary
# floating-point types, and integer types which are treated as double.
# The corresponding complex types may also be used (including complex
# integer types, which are a GNU extension, but are currently disabled
# here because they do not work properly with tgmath.h).

# The proposed resolution to TS 18661-1 DR#9
# <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2149.htm#dr_9>
# makes the <tgmath.h> rules for selecting a function to call
# correspond to the usual arithmetic conversions (applied successively
# to the arguments for generic parameters in order), which choose the
# type whose set of values contains that of the other type (undefined
# behavior if neither type's set of values is a superset of the
# other), with interchange types being preferred to standard types
# (long double, double, float), being preferred to extended types
# (_Float128x, _Float64x, _Float32x).

# For the standard and binary floating-point types supported by GCC 7
# on any platform, this means the resulting type is the last of the
# given types in one of the following orders, or undefined behavior if
# types with both ibm128 and binary128 representation are specified.

# If double = long double: _Float16, float, _Float32, _Float32x,
# double, long double, _Float64, _Float64x, _Float128.

# Otherwise: _Float16, float, _Float32, _Float32x, double, _Float64,
# _Float64x, long double, _Float128.

# We generate tests to verify the return type is exactly as expected.
# We also verify that the function called is real or complex as
# expected, and that it is called for the right floating-point format
# (but it is OK to call a double function instead of a long double one
# if they have the same format, for example).  For all the formats
# supported on any given configuration of glibc, the MANT_DIG value
# uniquely determines the format.

import string
import sys

class Type(object):
    """A type that may be used as an argument for generic parameters."""

    # All possible argument or result types.
    all_types_list = []
    # All argument types.
    argument_types_list = []
    # All real argument types.
    real_argument_types_list = []
    # Real argument types that correspond to a standard floating type
    # (float, double or long double; not _FloatN or _FloatNx).
    standard_real_argument_types_list = []
    # Real argument types other than float, double and long double
    # (i.e., those that are valid as arguments to narrowing macros
    # returning _FloatN or _FloatNx).
    non_standard_real_argument_types_list = []
    # The real floating types by their order properties (which are
    # tuples giving the positions in both the possible orders above).
    real_types_order = {}
    # The type double.
    double_type = None
    # The type long double.
    long_double_type = None
    # The type _Complex double.
    complex_double_type = None
    # The type _Float64.
    float64_type = None
    # The type _Complex _Float64.
    complex_float64_type = None
    # The type _Float64x.
    float64x_type = None
    # The type _Float64x if available, otherwise _Float64.
    float32x_ext_type = None

    def __init__(self, name, suffix=None, mant_dig=None, condition='1',
                 order=None, integer=False, complex=False, real_type=None):
        """Initialize a Type object, creating any corresponding complex type
        in the process."""
        self.name = name
        self.suffix = suffix
        self.mant_dig = mant_dig
        self.condition = condition
        self.order = order
        self.integer = integer
        self.complex = complex
        if complex:
            self.complex_type = self
            self.real_type = real_type
        else:
            # complex_type filled in by the caller once created.
            self.complex_type = None
            self.real_type = self

    def register_type(self, internal):
        """Record a type in the lists of all types."""
        Type.all_types_list.append(self)
        if not internal:
            Type.argument_types_list.append(self)
            if not self.complex:
                Type.real_argument_types_list.append(self)
                if not self.name.startswith('_Float'):
                    Type.standard_real_argument_types_list.append(self)
                if self.name not in ('float', 'double', 'long double'):
                    Type.non_standard_real_argument_types_list.append(self)
        if self.order is not None:
            Type.real_types_order[self.order] = self
        if self.name == 'double':
            Type.double_type = self
        if self.name == 'long double':
            Type.long_double_type = self
        if self.name == '_Complex double':
            Type.complex_double_type = self
        if self.name == '_Float64':
            Type.float64_type = self
        if self.name == '_Complex _Float64':
            Type.complex_float64_type = self
        if self.name == '_Float64x':
            Type.float64x_type = self
        if self.name == 'Float32x_ext':
            Type.float32x_ext_type = self

    @staticmethod
    def create_type(name, suffix=None, mant_dig=None, condition='1', order=None,
                    integer=False, complex_name=None, complex_ok=True,
                    internal=False):
        """Create and register a Type object for a real type, creating any
        corresponding complex type in the process."""
        real_type = Type(name, suffix=suffix, mant_dig=mant_dig,
                         condition=condition, order=order, integer=integer,
                         complex=False)
        if complex_ok:
            if complex_name is None:
                complex_name = '_Complex %s' % name
            complex_type = Type(complex_name, condition=condition,
                                integer=integer, complex=True,
                                real_type=real_type)
        else:
            complex_type = None
        real_type.complex_type = complex_type
        real_type.register_type(internal)
        if complex_type is not None:
            complex_type.register_type(internal)

    def floating_type(self, floatn):
        """Return the corresponding floating type."""
        if self.integer:
            if floatn:
                return (Type.complex_float64_type
                        if self.complex
                        else Type.float64_type)
            else:
                return (Type.complex_double_type
                        if self.complex
                        else Type.double_type)
        else:
            return self

    def real_floating_type(self, floatn):
        """Return the corresponding real floating type."""
        return self.real_type.floating_type(floatn)

    def __str__(self):
        """Return string representation of a type."""
        return self.name

    @staticmethod
    def init_types():
        """Initialize all the known types."""
        Type.create_type('_Float16', 'f16', 'FLT16_MANT_DIG',
                         complex_name='__CFLOAT16',
                         condition='defined HUGE_VAL_F16', order=(0, 0))
        Type.create_type('float', 'f', 'FLT_MANT_DIG', order=(1, 1))
        Type.create_type('_Float32', 'f32', 'FLT32_MANT_DIG',
                         complex_name='__CFLOAT32',
                         condition='defined HUGE_VAL_F32', order=(2, 2))
        Type.create_type('_Float32x', 'f32x', 'FLT32X_MANT_DIG',
                         complex_name='__CFLOAT32X',
                         condition='defined HUGE_VAL_F32X', order=(3, 3))
        Type.create_type('double', '', 'DBL_MANT_DIG', order=(4, 4))
        Type.create_type('long double', 'l', 'LDBL_MANT_DIG', order=(5, 7))
        Type.create_type('_Float64', 'f64', 'FLT64_MANT_DIG',
                         complex_name='__CFLOAT64',
                         condition='defined HUGE_VAL_F64', order=(6, 5))
        Type.create_type('_Float64x', 'f64x', 'FLT64X_MANT_DIG',
                         complex_name='__CFLOAT64X',
                         condition='defined HUGE_VAL_F64X', order=(7, 6))
        Type.create_type('_Float128', 'f128', 'FLT128_MANT_DIG',
                         complex_name='__CFLOAT128',
                         condition='defined HUGE_VAL_F128', order=(8, 8))
        Type.create_type('char', integer=True)
        Type.create_type('signed char', integer=True)
        Type.create_type('unsigned char', integer=True)
        Type.create_type('short int', integer=True)
        Type.create_type('unsigned short int', integer=True)
        Type.create_type('int', integer=True)
        Type.create_type('unsigned int', integer=True)
        Type.create_type('long int', integer=True)
        Type.create_type('unsigned long int', integer=True)
        Type.create_type('long long int', integer=True)
        Type.create_type('unsigned long long int', integer=True)
        Type.create_type('__int128', integer=True,
                         condition='defined __SIZEOF_INT128__')
        Type.create_type('unsigned __int128', integer=True,
                         condition='defined __SIZEOF_INT128__')
        Type.create_type('enum e', integer=True, complex_ok=False)
        Type.create_type('_Bool', integer=True, complex_ok=False)
        Type.create_type('bit_field', integer=True, complex_ok=False)
        # Internal types represent the combination of long double with
        # _Float64 or _Float64x, for which the ordering depends on
        # whether long double has the same format as double.
        Type.create_type('long_double_Float64', None, 'LDBL_MANT_DIG',
                         complex_name='complex_long_double_Float64',
                         condition='defined HUGE_VAL_F64', order=(6, 7),
                         internal=True)
        Type.create_type('long_double_Float64x', None, 'FLT64X_MANT_DIG',
                         complex_name='complex_long_double_Float64x',
                         condition='defined HUGE_VAL_F64X', order=(7, 7),
                         internal=True)
        # An internal type for the argument type used by f32x*
        # narrowing macros (_Float64x if available, otherwise
        # _Float64).
        Type.create_type('Float32x_ext', None, 'FLT32X_EXT_MANT_DIG',
                         complex_name='complex_Float32x_ext',
                         condition='1', internal=True)

    @staticmethod
    def can_combine_types(types, floatn):
        """Return a C preprocessor conditional for whether the given list of
        types can be used together as type-generic macro arguments."""
        have_long_double = False
        have_float128 = False
        for t in types:
            t = t.real_floating_type(floatn)
            if t.name == 'long double':
                have_long_double = True
            if t.name == '_Float128' or t.name == '_Float64x':
                have_float128 = True
        if have_long_double and have_float128:
            # If ibm128 format is in use for long double, both
            # _Float64x and _Float128 are binary128 and the types
            # cannot be combined.
            return '(LDBL_MANT_DIG != 106)'
        return '1'

    @staticmethod
    def combine_types(types, floatn):
        """Return the result of combining a set of types."""
        have_complex = False
        combined = None
        for t in types:
            if t.complex:
                have_complex = True
            t = t.real_floating_type(floatn)
            if combined is None:
                combined = t
            else:
                order = (max(combined.order[0], t.order[0]),
                         max(combined.order[1], t.order[1]))
                combined = Type.real_types_order[order]
        return combined.complex_type if have_complex else combined

def list_product_initial(initial, lists):
    """Return a list of lists, with an initial sequence from the first
    argument (a list of lists) followed by each sequence of one
    element from each successive element of the second argument."""
    if not lists:
        return initial
    return list_product_initial([a + [b] for a in initial for b in lists[0]],
                                lists[1:])

def list_product(lists):
    """Return a list of lists, with each sequence of one element from each
    successive element of the argument."""
    return list_product_initial([[]], lists)

try:
    trans_id = str.maketrans(' *', '_p')
except AttributeError:
    trans_id = string.maketrans(' *', '_p')
def var_for_type(name):
    """Return the name of a variable with a given type (name)."""
    return 'var_%s' % name.translate(trans_id)

def vol_var_for_type(name):
    """Return the name of a variable with a given volatile type (name)."""
    return 'vol_var_%s' % name.translate(trans_id)

def define_vars_for_type(name):
    """Return the definitions of variables with a given type (name)."""
    if name == 'bit_field':
        struct_vars = define_vars_for_type('struct s');
        return '%s#define %s %s.bf\n' % (struct_vars,
                                         vol_var_for_type(name),
                                         vol_var_for_type('struct s'))
    return ('%s %s __attribute__ ((unused));\n'
            '%s volatile %s __attribute__ ((unused));\n'
            % (name, var_for_type(name), name, vol_var_for_type(name)))

def if_cond_text(conds, text):
    """Return the result of making some text conditional under #if.  The
    text ends with a newline, as does the return value if not empty."""
    if '0' in conds:
        return ''
    conds = [c for c in conds if c != '1']
    conds = sorted(set(conds))
    if not conds:
        return text
    return '#if %s\n%s#endif\n' % (' && '.join(conds), text)

class Tests(object):
    """The state associated with testcase generation."""

    def __init__(self):
        """Initialize a Tests object."""
        self.header_list = ['#define __STDC_WANT_IEC_60559_TYPES_EXT__\n'
                            '#include <float.h>\n'
                            '#include <stdbool.h>\n'
                            '#include <stdint.h>\n'
                            '#include <stdio.h>\n'
                            '#include <string.h>\n'
                            '#include <tgmath.h>\n'
                            '\n'
                            'struct test\n'
                            '  {\n'
                            '    void (*func) (void);\n'
                            '    const char *func_name;\n'
                            '    const char *test_name;\n'
                            '    int mant_dig;\n'
                            '    int narrow_mant_dig;\n'
                            '  };\n'
                            'int num_pass, num_fail;\n'
                            'volatile int called_mant_dig;\n'
                            'const char *volatile called_func_name;\n'
                            'enum e { E, F };\n'
                            'struct s\n'
                            '  {\n'
                            '    int bf:2;\n'
                            '  };\n']
        float64_text = ('# if LDBL_MANT_DIG == DBL_MANT_DIG\n'
                        'typedef _Float64 long_double_Float64;\n'
                        'typedef __CFLOAT64 complex_long_double_Float64;\n'
                        '# else\n'
                        'typedef long double long_double_Float64;\n'
                        'typedef _Complex long double '
                        'complex_long_double_Float64;\n'
                        '# endif\n')
        float64_text = if_cond_text([Type.float64_type.condition],
                                    float64_text)
        float64x_text = ('# if LDBL_MANT_DIG == DBL_MANT_DIG\n'
                         'typedef _Float64x long_double_Float64x;\n'
                         'typedef __CFLOAT64X complex_long_double_Float64x;\n'
                         '# else\n'
                         'typedef long double long_double_Float64x;\n'
                         'typedef _Complex long double '
                         'complex_long_double_Float64x;\n'
                         '# endif\n')
        float64x_text = if_cond_text([Type.float64x_type.condition],
                                     float64x_text)
        float32x_ext_text = ('#ifdef HUGE_VAL_F64X\n'
                             'typedef _Float64x Float32x_ext;\n'
                             'typedef __CFLOAT64X complex_Float32x_ext;\n'
                             '# define FLT32X_EXT_MANT_DIG FLT64X_MANT_DIG\n'
                             '#else\n'
                             'typedef _Float64 Float32x_ext;\n'
                             'typedef __CFLOAT64 complex_Float32x_ext;\n'
                             '# define FLT32X_EXT_MANT_DIG FLT64_MANT_DIG\n'
                             '#endif\n')
        self.header_list.append(float64_text)
        self.header_list.append(float64x_text)
        self.header_list.append(float32x_ext_text)
        self.types_seen = set()
        for t in Type.all_types_list:
            self.add_type_var(t.name, t.condition)
        self.test_text_list = []
        self.test_array_list = []
        self.macros_seen = set()

    def add_type_var(self, name, cond):
        """Add declarations of variables for a type."""
        if name in self.types_seen:
            return
        t_vars = define_vars_for_type(name)
        self.header_list.append(if_cond_text([cond], t_vars))
        self.types_seen.add(name)

    def add_tests(self, macro, ret, args, complex_func=None):
        """Add tests for a given tgmath.h macro, if that is the macro for
        which tests are to be generated; otherwise just add it to the
        list of macros for which test generation is supported."""
        # 'c' means the function argument or return type is
        # type-generic and complex only (a complex function argument
        # may still have a real macro argument).  'g' means it is
        # type-generic and may be real or complex; 'r' means it is
        # type-generic and may only be real; 's' means the same as
        # 'r', but restricted to float, double and long double.
        self.macros_seen.add(macro)
        if macro != self.macro:
            return
        have_complex = False
        func = macro
        narrowing = False
        narrowing_std = False
        if ret == 'c' or 'c' in args:
            # Complex-only.
            have_complex = True
            complex_func = func
            func = None
        elif ret == 'g' or 'g' in args:
            # Real and complex.
            have_complex = True
            if complex_func == None:
                complex_func = 'c%s' % func
        # For narrowing macros, compute narrow_args, the list of
        # argument types for which there is an actual corresponding
        # function.  If none of those types exist, or the return type
        # does not exist, then the macro is not defined and no tests
        # of it can be run.
        if ret == 'float':
            narrowing = True
            narrowing_std = True
            narrow_cond = '1'
            narrow_args = [Type.double_type, Type.long_double_type]
            narrow_fallback = Type.double_type
        elif ret == 'double':
            narrowing = True
            narrowing_std = True
            narrow_cond = '1'
            narrow_args = [Type.long_double_type]
            narrow_fallback = Type.long_double_type
        elif ret.startswith('_Float'):
            narrowing = True
            narrow_args = []
            nret_type = None
            narrow_fallback = None
            for order, real_type in sorted(Type.real_types_order.items()):
                if real_type.name == ret:
                    nret_type = real_type
                elif nret_type and real_type.name.startswith('_Float'):
                    narrow_args.append(real_type)
                    if (narrow_fallback is None
                        and ret.endswith('x') == real_type.name.endswith('x')):
                        narrow_fallback = real_type
            if narrow_args:
                narrow_cond = ('(%s && (%s))'
                               % (nret_type.condition,
                                  ' || '.join(t.condition
                                              for t in narrow_args)))
                if narrow_fallback is None:
                    narrow_fallback = narrow_args[0]
                if ret == '_Float32x':
                    narrow_fallback = Type.float32x_ext_type
            else:
                # No possible argument types, even conditionally.
                narrow_cond = '0'
        narrowing_nonstd = narrowing and not narrowing_std
        types = [ret] + args
        for t in types:
            if t != 'c' and t != 'g' and t != 'r' and t != 's':
                self.add_type_var(t, '1')
        for t in Type.argument_types_list:
            if t.integer:
                continue
            if t.complex and not have_complex:
                continue
            if func == None and not t.complex:
                continue
            if ret == 's' and t.name.startswith('_Float'):
                continue
            if narrowing and t not in narrow_args:
                continue
            if ret == 'c':
                ret_name = t.complex_type.name
            elif ret == 'g':
                ret_name = t.name
            elif ret == 'r' or ret == 's':
                ret_name = t.real_type.name
            else:
                ret_name = ret
            dummy_func_name = complex_func if t.complex else func
            arg_list = []
            arg_num = 0
            for a in args:
                if a == 'c':
                    arg_name = t.complex_type.name
                elif a == 'g':
                    arg_name = t.name
                elif a == 'r' or a == 's':
                    arg_name = t.real_type.name
                else:
                    arg_name = a
                arg_list.append('%s arg%d __attribute__ ((unused))'
                                % (arg_name, arg_num))
                arg_num += 1
            dummy_func = ('%s\n'
                          '(%s%s) (%s)\n'
                          '{\n'
                          '  called_mant_dig = %s;\n'
                          '  called_func_name = "%s";\n'
                          '  return 0;\n'
                          '}\n' % (ret_name, dummy_func_name,
                                   t.real_type.suffix, ', '.join(arg_list),
                                   t.real_type.mant_dig, dummy_func_name))
            if narrowing:
                dummy_cond = [narrow_cond, t.condition]
            else:
                dummy_cond = [t.condition]
            dummy_func = if_cond_text(dummy_cond, dummy_func)
            self.test_text_list.append(dummy_func)
        arg_types = []
        for t in args:
            if t == 'g' or t == 'c':
                arg_types.append(Type.argument_types_list)
            elif t == 'r':
                if narrowing_std:
                    arg_types.append(Type.standard_real_argument_types_list)
                elif narrowing:
                    arg_types.append(
                        Type.non_standard_real_argument_types_list)
                else:
                    arg_types.append(Type.real_argument_types_list)
            elif t == 's':
                arg_types.append(Type.standard_real_argument_types_list)
        arg_types_product = list_product(arg_types)
        test_num = 0
        for this_args in arg_types_product:
            comb_type = Type.combine_types(this_args, narrowing_nonstd)
            if narrowing:
                # As long as there are no integer arguments, and as
                # long as the chosen argument type is as wide as all
                # the floating-point arguments passed, the semantics
                # of the macro call do not depend on the exact
                # function chosen.  In particular, for f32x functions
                # when _Float64x exists, the chosen type should differ
                # for _Float32x and _Float64 arguments, but it is not
                # always possible to distinguish those types before
                # GCC 7 and the implementation does not attempt to do
                # so before GCC 8.
                narrow_mant_dig = comb_type.real_type.mant_dig
                for arg_type in this_args:
                    if arg_type.integer:
                        narrow_mant_dig = 0
            else:
                narrow_mant_dig = 0
            if (narrowing
                and comb_type not in narrow_args
                and narrow_fallback is not None):
                comb_type = narrow_fallback
            can_comb = Type.can_combine_types(this_args, narrowing_nonstd)
            all_conds = [t.condition for t in this_args]
            all_conds.append(can_comb)
            if narrowing:
                all_conds.append(narrow_cond)
            any_complex = func == None
            for t in this_args:
                if t.complex:
                    any_complex = True
            func_name = complex_func if any_complex else func
            test_name = '%s (%s)' % (macro,
                                     ', '.join([t.name for t in this_args]))
            test_func_name = 'test_%s_%d' % (macro, test_num)
            test_num += 1
            mant_dig = comb_type.real_type.mant_dig
            test_text = '%s, "%s", "%s", %s, %s' % (test_func_name, func_name,
                                                    test_name, mant_dig,
                                                    narrow_mant_dig)
            test_text = '    { %s },\n' % test_text
            test_text = if_cond_text(all_conds, test_text)
            self.test_array_list.append(test_text)
            call_args = []
            call_arg_pos = 0
            for t in args:
                if t == 'g' or t == 'c' or t == 'r' or t == 's':
                    type = this_args[call_arg_pos].name
                    call_arg_pos += 1
                else:
                    type = t
                call_args.append(vol_var_for_type(type))
            call_args_text = ', '.join(call_args)
            if ret == 'g':
                ret_type = comb_type.name
            elif ret == 'r' or ret == 's':
                ret_type = comb_type.real_type.name
            elif ret == 'c':
                ret_type = comb_type.complex_type.name
            else:
                ret_type = ret
            call_text = '%s (%s)' % (macro, call_args_text)
            test_func_text = ('static void\n'
                              '%s (void)\n'
                              '{\n'
                              '  extern typeof (%s) %s '
                              '__attribute__ ((unused));\n'
                              '  %s = %s;\n'
                              '}\n' % (test_func_name, call_text,
                                       var_for_type(ret_type),
                                       vol_var_for_type(ret_type), call_text))
            test_func_text = if_cond_text(all_conds, test_func_text)
            self.test_text_list.append(test_func_text)

    def add_all_tests(self, macro):
        """Add tests for the given tgmath.h macro, if any, and generate the
        list of all supported macros."""
        self.macro = macro
        # C99/C11 real-only functions.
        self.add_tests('atan2', 'r', ['r', 'r'])
        self.add_tests('cbrt', 'r', ['r'])
        self.add_tests('ceil', 'r', ['r'])
        self.add_tests('copysign', 'r', ['r', 'r'])
        self.add_tests('erf', 'r', ['r'])
        self.add_tests('erfc', 'r', ['r'])
        self.add_tests('exp2', 'r', ['r'])
        self.add_tests('expm1', 'r', ['r'])
        self.add_tests('fdim', 'r', ['r', 'r'])
        self.add_tests('floor', 'r', ['r'])
        self.add_tests('fma', 'r', ['r', 'r', 'r'])
        self.add_tests('fmax', 'r', ['r', 'r'])
        self.add_tests('fmin', 'r', ['r', 'r'])
        self.add_tests('fmod', 'r', ['r', 'r'])
        self.add_tests('frexp', 'r', ['r', 'int *'])
        self.add_tests('hypot', 'r', ['r', 'r'])
        self.add_tests('ilogb', 'int', ['r'])
        self.add_tests('ldexp', 'r', ['r', 'int'])
        self.add_tests('lgamma', 'r', ['r'])
        self.add_tests('llrint', 'long long int', ['r'])
        self.add_tests('llround', 'long long int', ['r'])
        # log10 is real-only in ISO C, but supports complex arguments
        # as a GNU extension.
        self.add_tests('log10', 'g', ['g'])
        self.add_tests('log1p', 'r', ['r'])
        self.add_tests('log2', 'r', ['r'])
        self.add_tests('logb', 'r', ['r'])
        self.add_tests('lrint', 'long int', ['r'])
        self.add_tests('lround', 'long int', ['r'])
        self.add_tests('nearbyint', 'r', ['r'])
        self.add_tests('nextafter', 'r', ['r', 'r'])
        self.add_tests('nexttoward', 's', ['s', 'long double'])
        self.add_tests('remainder', 'r', ['r', 'r'])
        self.add_tests('remquo', 'r', ['r', 'r', 'int *'])
        self.add_tests('rint', 'r', ['r'])
        self.add_tests('round', 'r', ['r'])
        self.add_tests('scalbn', 'r', ['r', 'int'])
        self.add_tests('scalbln', 'r', ['r', 'long int'])
        self.add_tests('tgamma', 'r', ['r'])
        self.add_tests('trunc', 'r', ['r'])
        # C99/C11 real-and-complex functions.
        self.add_tests('acos', 'g', ['g'])
        self.add_tests('asin', 'g', ['g'])
        self.add_tests('atan', 'g', ['g'])
        self.add_tests('acosh', 'g', ['g'])
        self.add_tests('asinh', 'g', ['g'])
        self.add_tests('atanh', 'g', ['g'])
        self.add_tests('cos', 'g', ['g'])
        self.add_tests('sin', 'g', ['g'])
        self.add_tests('tan', 'g', ['g'])
        self.add_tests('cosh', 'g', ['g'])
        self.add_tests('sinh', 'g', ['g'])
        self.add_tests('tanh', 'g', ['g'])
        self.add_tests('exp', 'g', ['g'])
        self.add_tests('log', 'g', ['g'])
        self.add_tests('pow', 'g', ['g', 'g'])
        self.add_tests('sqrt', 'g', ['g'])
        self.add_tests('fabs', 'r', ['g'], 'cabs')
        # C99/C11 complex-only functions.
        self.add_tests('carg', 'r', ['c'])
        self.add_tests('cimag', 'r', ['c'])
        self.add_tests('conj', 'c', ['c'])
        self.add_tests('cproj', 'c', ['c'])
        self.add_tests('creal', 'r', ['c'])
        # TS 18661-1 functions.
        self.add_tests('roundeven', 'r', ['r'])
        self.add_tests('nextup', 'r', ['r'])
        self.add_tests('nextdown', 'r', ['r'])
        self.add_tests('fminmag', 'r', ['r', 'r'])
        self.add_tests('fmaxmag', 'r', ['r', 'r'])
        self.add_tests('llogb', 'long int', ['r'])
        self.add_tests('fromfp', 'intmax_t', ['r', 'int', 'unsigned int'])
        self.add_tests('fromfpx', 'intmax_t', ['r', 'int', 'unsigned int'])
        self.add_tests('ufromfp', 'uintmax_t', ['r', 'int', 'unsigned int'])
        self.add_tests('ufromfpx', 'uintmax_t', ['r', 'int', 'unsigned int'])
        for fn in ('add', 'div', 'mul', 'sub'):
            for ret, prefix in (('float', 'f'),
                                ('double', 'd'),
                                ('_Float16', 'f16'),
                                ('_Float32', 'f32'),
                                ('_Float64', 'f64'),
                                ('_Float128', 'f128'),
                                ('_Float32x', 'f32x'),
                                ('_Float64x', 'f64x')):
                self.add_tests(prefix + fn, ret, ['r', 'r'])
        # Miscellaneous functions.
        self.add_tests('scalb', 's', ['s', 's'])

    def tests_text(self):
        """Return the text of the generated testcase."""
        test_list = [''.join(self.test_text_list),
                     'static const struct test tests[] =\n'
                     '  {\n',
                     ''.join(self.test_array_list),
                     '  };\n']
        footer_list = ['static int\n'
                       'do_test (void)\n'
                       '{\n'
                       '  for (size_t i = 0;\n'
                       '       i < sizeof (tests) / sizeof (tests[0]);\n'
                       '       i++)\n'
                       '    {\n'
                       '      called_mant_dig = 0;\n'
                       '      called_func_name = "";\n'
                       '      tests[i].func ();\n'
                       '      if (called_mant_dig == tests[i].mant_dig\n'
                       '          && strcmp (called_func_name,\n'
                       '                     tests[i].func_name) == 0)\n'
                       '        num_pass++;\n'
                       '#if !__GNUC_PREREQ (8, 0)\n'
                       '      else if (tests[i].narrow_mant_dig > 0\n'
                       '               && (called_mant_dig\n'
                       '                   >= tests[i].narrow_mant_dig)\n'
                       '               && strcmp (called_func_name,\n'
                       '                          tests[i].func_name) == 0)\n'
                       '        {\n'
                       '          num_pass++;\n'
                       '          printf ("Test %zu (%s):\\n"\n'
                       '                  "  Expected: %s precision %d\\n"\n'
                       '                  "  Actual: %s precision %d\\n"\n'
                       '                  "  (OK with old GCC)\\n\\n",\n'
                       '                  i, tests[i].test_name,\n'
                       '                  tests[i].func_name,\n'
                       '                  tests[i].mant_dig,\n'
                       '                  called_func_name, called_mant_dig);\n'
                       '        }\n'
                       '#endif\n'
                       '      else\n'
                       '        {\n'
                       '          num_fail++;\n'
                       '          printf ("Test %zu (%s):\\n"\n'
                       '                  "  Expected: %s precision %d\\n"\n'
                       '                  "  Actual: %s precision %d\\n\\n",\n'
                       '                  i, tests[i].test_name,\n'
                       '                  tests[i].func_name,\n'
                       '                  tests[i].mant_dig,\n'
                       '                  called_func_name, called_mant_dig);\n'
                       '        }\n'
                       '    }\n'
                       '  printf ("%d pass, %d fail\\n", num_pass, num_fail);\n'
                       '  return num_fail != 0;\n'
                       '}\n'
                       '\n'
                       '#include <support/test-driver.c>']
        return ''.join(self.header_list + test_list + footer_list)

    def check_macro_list(self, macro_list):
        """Check the list of macros that can be tested."""
        if self.macros_seen != set(macro_list):
            print('error: macro list mismatch')
            sys.exit(1)

def main():
    """The main entry point."""
    Type.init_types()
    t = Tests()
    if sys.argv[1] == 'check-list':
        macro = None
        macro_list = sys.argv[2:]
    else:
        macro = sys.argv[1]
        macro_list = []
    t.add_all_tests(macro)
    if macro:
        print(t.tests_text())
    else:
        t.check_macro_list(macro_list)

if __name__ == '__main__':
    main()
