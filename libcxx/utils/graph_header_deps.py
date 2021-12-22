#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import argparse
import os
import re
import sys


def is_config_header(h):
    return os.path.basename(h) in ['__config', '__libcpp_version', '__undef_macros', 'version']


def is_experimental_header(h):
    return ('experimental/' in h) or ('ext/' in h)


def is_support_header(h):
    return '__support/' in h


class FileEntry:
    def __init__(self, includes, individual_linecount):
        self.includes = includes
        self.individual_linecount = individual_linecount
        self.cumulative_linecount = None  # documentation: this gets filled in later
        self.is_graph_root = None  # documentation: this gets filled in later


def list_all_roots_under(root):
    result = []
    for root, _, files in os.walk(root):
        for fname in files:
            if os.path.basename(root).startswith('__') or fname.startswith('__'):
                pass
            elif ('.' in fname and not fname.endswith('.h')):
                pass
            else:
                result.append(root + '/' + fname)
    return result


def build_file_entry(fname, options):
    assert os.path.exists(fname)

    def locate_header_file(h, paths):
        for p in paths:
            fullname = p + '/' + h
            if os.path.exists(fullname):
                return fullname
        if options.error_on_file_not_found:
            raise RuntimeError('Header not found: %s, included by %s' % (h, fname))
        return None

    local_includes = []
    system_includes = []
    linecount = 0
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            linecount += 1
            m = re.match(r'\s*#\s*include\s+"([^"]*)"', line)
            if m is not None:
                local_includes.append(m.group(1))
            m = re.match(r'\s*#\s*include\s+<([^>]*)>', line)
            if m is not None:
                system_includes.append(m.group(1))

    fully_qualified_includes = [
        locate_header_file(h, options.search_dirs)
        for h in system_includes
    ] + [
        locate_header_file(h, os.path.dirname(fname))
        for h in local_includes
    ]

    return FileEntry(
        # If file-not-found wasn't an error, then skip non-found files
        includes = [h for h in fully_qualified_includes if h is not None],
        individual_linecount = linecount,
    )


def transitive_closure_of_includes(graph, h1):
    visited = set()
    def explore(graph, h1):
        if h1 not in visited:
            visited.add(h1)
            for h2 in graph[h1].includes:
                explore(graph, h2)
    explore(graph, h1)
    return visited


def transitively_includes(graph, h1, h2):
    return (h1 != h2) and (h2 in transitive_closure_of_includes(graph, h1))


def build_graph(roots, options):
    original_roots = list(roots)
    graph = {}
    while roots:
        frontier = roots
        roots = []
        for fname in frontier:
            if fname not in graph:
                graph[fname] = build_file_entry(fname, options)
                graph[fname].is_graph_root = (fname in original_roots)
                roots += graph[fname].includes
    for fname, entry in graph.items():
        entry.cumulative_linecount = sum(graph[h].individual_linecount for h in transitive_closure_of_includes(graph, fname))
    return graph


def get_friendly_id(fname):
    i = fname.index('include/')
    assert(i >= 0)
    result = fname[i+8:]
    return result


def get_graphviz(graph, options):

    def get_decorators(fname, entry):
        result = ''
        if entry.is_graph_root:
            result += ' [style=bold]'
        if options.show_individual_line_counts and options.show_cumulative_line_counts:
            result += ' [label="%s\\n%d indiv, %d cumul"]' % (
                get_friendly_id(fname), entry.individual_linecount, entry.cumulative_linecount
            )
        elif options.show_individual_line_counts:
            result += ' [label="%s\\n%d indiv"]' % (get_friendly_id(fname), entry.individual_linecount)
        elif options.show_cumulative_line_counts:
            result += ' [label="%s\\n%d cumul"]' % (get_friendly_id(fname), entry.cumulative_linecount)
        return result

    result = ''
    result += 'strict digraph {\n'
    result += '    rankdir=LR;\n'
    result += '    layout=dot;\n\n'
    for fname, entry in graph.items():
        result += '    "%s"%s;\n' % (get_friendly_id(fname), get_decorators(fname, entry))
        for h in entry.includes:
            if any(transitively_includes(graph, i, h) for i in entry.includes) and not options.show_transitive_edges:
                continue
            result += '        "%s" -> "%s";\n' % (get_friendly_id(fname), get_friendly_id(h))
    result += '}\n'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Produce a dependency graph of libc++ headers, in GraphViz dot format.\n' +
                    'For example, ./graph_header_deps.py | dot -Tpng > graph.png',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--root', default=None, metavar='FILE', help='File or directory to be the root of the dependency graph')
    parser.add_argument('-I', dest='search_dirs', default=[], action='append', metavar='DIR', help='Path(s) to search for local includes')
    parser.add_argument('--show-transitive-edges', action='store_true', help='Show edges to headers that are transitively included anyway')
    parser.add_argument('--show-config-headers', action='store_true', help='Show universally included headers, such as __config')
    parser.add_argument('--show-experimental-headers', action='store_true', help='Show headers in the experimental/ and ext/ directories')
    parser.add_argument('--show-support-headers', action='store_true', help='Show headers in the __support/ directory')
    parser.add_argument('--show-individual-line-counts', action='store_true', help='Include an individual line count in each node')
    parser.add_argument('--show-cumulative-line-counts', action='store_true', help='Include a total line count in each node')
    parser.add_argument('--error-on-file-not-found', action='store_true', help="Don't ignore failure to open an #included file")

    options = parser.parse_args()

    if options.root is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        options.root = os.path.join(curr_dir, '../include')

    if options.search_dirs == [] and os.path.isdir(options.root):
        options.search_dirs = [options.root]

    options.root = os.path.abspath(options.root)
    options.search_dirs = [os.path.abspath(p) for p in options.search_dirs]

    if os.path.isdir(options.root):
        roots = list_all_roots_under(options.root)
    elif os.path.isfile(options.root):
        roots = [options.root]
    else:
        raise RuntimeError('--root seems to be invalid')

    graph = build_graph(roots, options)

    # Eliminate certain kinds of "visual noise" headers, if asked for.
    def should_keep(fname):
        return all([
            options.show_config_headers or not is_config_header(fname),
            options.show_experimental_headers or not is_experimental_header(fname),
            options.show_support_headers or not is_support_header(fname),
        ])

    for fname in list(graph.keys()):
        if should_keep(fname):
            graph[fname].includes = [h for h in graph[fname].includes if should_keep(h)]
        else:
            del graph[fname]

    # Look for cycles.
    no_cycles_detected = True
    for fname, entry in graph.items():
        for h in entry.includes:
            if h == fname:
                sys.stderr.write('Cycle detected: %s includes itself\n' % (
                    get_friendly_id(fname)
                ))
                no_cycles_detected = False
            elif transitively_includes(graph, h, fname):
                sys.stderr.write('Cycle detected between %s and %s\n' % (
                    get_friendly_id(fname), get_friendly_id(h)
                ))
                no_cycles_detected = False
    assert no_cycles_detected

    print(get_graphviz(graph, options))
