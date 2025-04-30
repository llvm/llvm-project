#!/usr/bin/python3
# Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

"""Benchmark program generator script

This script takes a function name as input and generates a program using
an input file located in the benchtests directory.  The name of the
input file should be of the form foo-inputs where 'foo' is the name of
the function.
"""

from __future__ import print_function
import sys
import os
import itertools

# Macro definitions for functions that take no arguments.  For functions
# that take arguments, the STRUCT_TEMPLATE, ARGS_TEMPLATE and
# VARIANTS_TEMPLATE are used instead.
DEFINES_TEMPLATE = '''
#define CALL_BENCH_FUNC(v, i) %(func)s ()
#define NUM_VARIANTS (1)
#define NUM_SAMPLES(v) (1)
#define VARIANT(v) FUNCNAME "()"
'''

# Structures to store arguments for the function call.  A function may
# have its inputs partitioned to represent distinct performance
# characteristics or distinct flavors of the function.  Each such
# variant is represented by the _VARIANT structure.  The ARGS structure
# represents a single set of arguments.
STRUCT_TEMPLATE = '''
#define CALL_BENCH_FUNC(v, i, x) %(func)s (x %(func_args)s)

struct args
{
%(args)s
  double timing;
};

struct _variants
{
  const char *name;
  int count;
  struct args *in;
};
'''

# The actual input arguments.
ARGS_TEMPLATE = '''
struct args in%(argnum)d[%(num_args)d] = {
%(args)s
};
'''

# The actual variants, along with macros defined to access the variants.
VARIANTS_TEMPLATE = '''
struct _variants variants[%(num_variants)d] = {
%(variants)s
};

#define NUM_VARIANTS %(num_variants)d
#define NUM_SAMPLES(i) (variants[i].count)
#define VARIANT(i) (variants[i].name)
'''

# Epilogue for the generated source file.
EPILOGUE = '''
#define RESULT(__v, __i) (variants[(__v)].in[(__i)].timing)
#define RESULT_ACCUM(r, v, i, old, new) \\
        ((RESULT ((v), (i))) = (RESULT ((v), (i)) * (old) + (r)) / ((new) + 1))
#define BENCH_FUNC(i, j) ({%(getret)s CALL_BENCH_FUNC (i, j, );})
#define BENCH_FUNC_LAT(i, j) ({%(getret)s CALL_BENCH_FUNC (i, j, %(latarg)s);})
#define BENCH_VARS %(defvar)s
#define FUNCNAME "%(func)s"
#include "bench-skeleton.c"'''


def gen_source(func, directives, all_vals):
    """Generate source for the function

    Generate the C source for the function from the values and
    directives.

    Args:
      func: The function name
      directives: A dictionary of directives applicable to this function
      all_vals: A dictionary input values
    """
    # The includes go in first.
    for header in directives['includes']:
        print('#include <%s>' % header)

    for header in directives['include-sources']:
        print('#include "%s"' % header)

    # Print macros.  This branches out to a separate routine if
    # the function takes arguments.
    if not directives['args']:
        print(DEFINES_TEMPLATE % {'func': func})
        outargs = []
    else:
        outargs = _print_arg_data(func, directives, all_vals)

    # Print the output variable definitions if necessary.
    for out in outargs:
        print(out)

    # If we have a return value from the function, make sure it is
    # assigned to prevent the compiler from optimizing out the
    # call.
    getret = ''
    latarg = ''
    defvar = ''

    if directives['ret']:
        print('static %s volatile ret;' % directives['ret'])
        print('static %s zero __attribute__((used)) = 0;' % directives['ret'])
        getret = 'ret = func_res = '
        # Note this may not work if argument and result type are incompatible.
        latarg = 'func_res * zero +'
        defvar = '%s func_res = 0;' % directives['ret']

    # Test initialization.
    if directives['init']:
        print('#define BENCH_INIT %s' % directives['init'])

    print(EPILOGUE % {'getret': getret, 'func': func, 'latarg': latarg, 'defvar': defvar })


def _print_arg_data(func, directives, all_vals):
    """Print argument data

    This is a helper function for gen_source that prints structure and
    values for arguments and their variants and returns output arguments
    if any are found.

    Args:
      func: Function name
      directives: A dictionary of directives applicable to this function
      all_vals: A dictionary input values

    Returns:
      Returns a list of definitions for function arguments that act as
      output parameters.
    """
    # First, all of the definitions.  We process writing of
    # CALL_BENCH_FUNC, struct args and also the output arguments
    # together in a single traversal of the arguments list.
    func_args = []
    arg_struct = []
    outargs = []

    for arg, i in zip(directives['args'], itertools.count()):
        if arg[0] == '<' and arg[-1] == '>':
            pos = arg.rfind('*')
            if pos == -1:
                die('Output argument must be a pointer type')

            outargs.append('static %s out%d __attribute__((used));' % (arg[1:pos], i))
            func_args.append(' &out%d' % i)
        else:
            arg_struct.append('  %s volatile arg%d;' % (arg, i))
            func_args.append('variants[v].in[i].arg%d' % i)

    print(STRUCT_TEMPLATE % {'args' : '\n'.join(arg_struct), 'func': func,
                             'func_args': ', '.join(func_args)})

    # Now print the values.
    variants = []
    for (k, vals), i in zip(all_vals.items(), itertools.count()):
        out = ['  {%s, 0},' % v for v in vals]

        # Members for the variants structure list that we will
        # print later.
        variants.append('  {"%s", %d, in%d},' % (k, len(vals), i))
        print(ARGS_TEMPLATE % {'argnum': i, 'num_args': len(vals),
                               'args': '\n'.join(out)})

    # Print the variants and the last set of macros.
    print(VARIANTS_TEMPLATE % {'num_variants': len(all_vals),
                               'variants': '\n'.join(variants)})
    return outargs


def _process_directive(d_name, d_val):
    """Process a directive.

    Evaluate the directive name and value passed and return the
    processed value. This is a helper function for parse_file.

    Args:
      d_name: Name of the directive
      d_val: The string value to process

    Returns:
      The processed value, which may be the string as it is or an object
      that describes the directive.
    """
    # Process the directive values if necessary.  name and ret don't
    # need any processing.
    if d_name.startswith('include'):
        d_val = d_val.split(',')
    elif d_name == 'args':
        d_val = d_val.split(':')

    # Return the values.
    return d_val


def parse_file(func):
    """Parse an input file

    Given a function name, open and parse an input file for the function
    and get the necessary parameters for the generated code and the list
    of inputs.

    Args:
      func: The function name

    Returns:
      A tuple of two elements, one a dictionary of directives and the
      other a dictionary of all input values.
    """
    all_vals = {}
    # Valid directives.
    directives = {
            'name': '',
            'args': [],
            'includes': [],
            'include-sources': [],
            'ret': '',
            'init': ''
    }

    try:
        with open('%s-inputs' % func) as f:
            for line in f:
                # Look for directives and parse it if found.
                if line.startswith('##'):
                    try:
                        d_name, d_val = line[2:].split(':', 1)
                        d_name = d_name.strip()
                        d_val = d_val.strip()
                        directives[d_name] = _process_directive(d_name, d_val)
                    except (IndexError, KeyError):
                        die('Invalid directive: %s' % line[2:])

                # Skip blank lines and comments.
                line = line.split('#', 1)[0].rstrip()
                if not line:
                    continue

                # Otherwise, we're an input.  Add to the appropriate
                # input set.
                cur_name = directives['name']
                all_vals.setdefault(cur_name, [])
                all_vals[cur_name].append(line)
    except IOError as ex:
        die("Failed to open input file (%s): %s" % (ex.filename, ex.strerror))

    return directives, all_vals


def die(msg):
    """Exit with an error

    Prints an error message to the standard error stream and exits with
    a non-zero status.

    Args:
      msg: The error message to print to standard error
    """
    print('%s\n' % msg, file=sys.stderr)
    sys.exit(os.EX_DATAERR)


def main(args):
    """Main function

    Use the first command line argument as function name and parse its
    input file to generate C source that calls the function repeatedly
    for the input.

    Args:
      args: The command line arguments with the program name dropped

    Returns:
      os.EX_USAGE on error and os.EX_OK on success.
    """
    if len(args) != 1:
        print('Usage: %s <function>' % sys.argv[0])
        return os.EX_USAGE

    directives, all_vals = parse_file(args[0])
    gen_source(args[0], directives, all_vals)
    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
