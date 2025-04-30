#!/usr/bin/python
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
"""Compare results of string functions

Given a string benchmark result file, print a table with comparisons with a
baseline.  The baseline is the first function, which typically is the builtin
function.
"""
import matplotlib as mpl
mpl.use('Agg')

import sys
import os
import json
import pylab
import argparse
import traceback

try:
    import jsonschema as validator
except ImportError:
    print('Could not find jsonschema module.')
    raise


def parse_file(filename, schema_filename):
    try:
        with open(schema_filename, 'r') as schemafile:
            schema = json.load(schemafile)
            with open(filename, 'r') as benchfile:
                bench = json.load(benchfile)
                validator.validate(bench, schema)
        return bench
    except:
        print(traceback.format_exc(limit=1))
        sys.exit(os.EX_NOINPUT)

def draw_graph(f, v, ifuncs, results):
    """Plot graphs for functions

    Plot line graphs for each of the ifuncs

    Args:
        f: Function name
        v: Benchmark variant for the function.
        ifuncs: List of ifunc names
        results: Dictionary of results for each test criterion
    """
    print('Generating graph for %s, variant \'%s\'' % (f, v))
    xkeys = results.keys()

    pylab.clf()
    fig = pylab.figure(frameon=False)
    fig.set_size_inches(32, 18)
    pylab.ylabel('Performance improvement from base')
    X = range(len(xkeys))
    pylab.xticks(X, xkeys)

    i = 0

    while i < len(ifuncs):
        Y = [results[k][i] for k in xkeys]
        lines = pylab.plot(X, Y, label=':'+ifuncs[i])
        i = i + 1

    pylab.legend()
    pylab.grid()
    pylab.savefig('%s-%s.png' % (f, v), bbox_inches='tight')


def process_results(results, attrs, funcs, base_func, graph, no_diff, no_header):
    """ Process results and print them

    Args:
        results: JSON dictionary of results
        attrs: Attributes that form the test criteria
        funcs: Functions that are selected
    """

    for f in results['functions'].keys():

        v = results['functions'][f]['bench-variant']

        selected = {}
        index = 0
        base_index = 0
        if funcs:
            ifuncs = []
            first_func = True
            for i in results['functions'][f]['ifuncs']:
                if i in funcs:
                    if first_func:
                        base_index = index
                        first_func = False
                    selected[index] = 1
                    ifuncs.append(i)
                else:
                    selected[index] = 0
                index += 1
        else:
            ifuncs = results['functions'][f]['ifuncs']
            for i in ifuncs:
                selected[index] = 1
                index += 1

        if base_func:
            try:
                base_index = results['functions'][f]['ifuncs'].index(base_func)
            except ValueError:
                sys.stderr.write('Invalid -b "%s" parameter. Options: %s.\n' %
                                 (base_func, ', '.join(results['functions'][f]['ifuncs'])))
                sys.exit(os.EX_DATAERR)

        if not no_header:
            print('Function: %s' % f)
            print('Variant: %s' % v)
            print("%36s%s" % (' ', '\t'.join(ifuncs)))
            print("=" * 120)

        graph_res = {}
        for res in results['functions'][f]['results']:
            try:
                attr_list = ['%s=%s' % (a, res[a]) for a in attrs]
            except KeyError as ke:
                sys.stderr.write('Invalid -a %s parameter. Options: %s.\n'
                                 % (ke, ', '.join([a for a in res.keys() if a != 'timings'])))
                sys.exit(os.EX_DATAERR)
            i = 0
            key = ', '.join(attr_list)
            sys.stdout.write('%36s: ' % key)
            graph_res[key] = res['timings']
            for t in res['timings']:
                if selected[i]:
                    sys.stdout.write ('%12.2f' % t)
                    if not no_diff:
                        if i != base_index:
                            base = res['timings'][base_index]
                            diff = (base - t) * 100 / base
                            sys.stdout.write (' (%6.2f%%)' % diff)
                    sys.stdout.write('\t')
                i = i + 1
            print('')

        if graph:
            draw_graph(f, v, results['functions'][f]['ifuncs'], graph_res)


def main(args):
    """Program Entry Point

    Take a string benchmark output file and compare timings.
    """

    base_func = None
    filename = args.input
    schema_filename = args.schema
    base_func = args.base
    attrs = args.attributes.split(',')
    if args.functions:
        funcs = args.functions.split(',')
        if base_func and not base_func in funcs:
            print('Baseline function (%s) not found.' % base_func)
            sys.exit(os.EX_DATAERR)
    else:
        funcs = None

    results = parse_file(args.input, args.schema)
    process_results(results, attrs, funcs, base_func, args.graph, args.no_diff, args.no_header)
    return os.EX_OK


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The required arguments.
    req = parser.add_argument_group(title='required arguments')
    req.add_argument('-a', '--attributes', required=True,
                        help='Comma separated list of benchmark attributes.')
    req.add_argument('-i', '--input', required=True,
                        help='Input JSON benchmark result file.')
    req.add_argument('-s', '--schema', required=True,
                        help='Schema file to validate the result file.')

    # Optional arguments.
    parser.add_argument('-f', '--functions',
                        help='Comma separated list of functions.')
    parser.add_argument('-b', '--base',
                        help='IFUNC variant to set as baseline.')
    parser.add_argument('-g', '--graph', action='store_true',
                        help='Generate a graph from results.')
    parser.add_argument('--no-diff', action='store_true',
                        help='Do not print the difference from baseline.')
    parser.add_argument('--no-header', action='store_true',
                        help='Do not print the header.')

    args = parser.parse_args()
    sys.exit(main(args))
