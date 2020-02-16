#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from argparse import ArgumentParser
import os
import shutil
import sys
import shlex
import json
import re
import libcxx.graph as dot
import libcxx.util

def print_and_exit(msg):
    sys.stderr.write(msg + '\n')
    sys.exit(1)

def libcxx_include_path():
    curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    include_dir = os.path.join(curr_dir, 'include')
    return include_dir

def get_libcxx_headers():
    headers = []
    include_dir = libcxx_include_path()
    for fname in os.listdir(include_dir):
        f = os.path.join(include_dir, fname)
        if not os.path.isfile(f):
            continue
        base, ext = os.path.splitext(fname)
        if (ext == '' or ext == '.h') and (not fname.startswith('__') or fname == '__config'):
            headers += [f]
    return headers


def rename_headers_and_remove_test_root(graph):
    inc_root = libcxx_include_path()
    to_remove = set()
    for n in graph.nodes:
        assert 'label' in n.attributes
        l = n.attributes['label']
        if not l.startswith('/') and os.path.exists(os.path.join('/', l)):
            l = '/' + l
        if l.endswith('.tmp.cpp'):
            to_remove.add(n)
        if l.startswith(inc_root):
            l = l[len(inc_root):]
            if l.startswith('/'):
                l = l[1:]
        n.attributes['label'] = l
    for n in to_remove:
        graph.removeNode(n)

def remove_non_std_headers(graph):
    inc_root = libcxx_include_path()
    to_remove = set()
    for n in graph.nodes:
        test_file = os.path.join(inc_root, n.attributes['label'])
        if not test_file.startswith(inc_root):
            to_remove.add(n)
    for xn in to_remove:
        graph.removeNode(xn)

class DependencyCommand(object):
    def __init__(self, compile_commands, output_dir, new_std=None):
        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(output_dir):
            print_and_exit('"%s" must point to a directory' % output_dir)
        self.output_dir = output_dir
        self.new_std = new_std
        cwd,bcmd =  self._get_base_command(compile_commands)
        self.cwd = cwd
        self.base_cmd = bcmd

    def run_for_headers(self, header_list):
        outputs = []
        for header in header_list:
            header_name = os.path.basename(header)
            out = os.path.join(self.output_dir, ('%s.dot' % header_name))
            outputs += [out]
            cmd =  self.base_cmd + ["-fsyntax-only", "-Xclang", "-dependency-dot", "-Xclang", "%s" % out, '-xc++', '-']
            libcxx.util.executeCommandOrDie(cmd, cwd=self.cwd, input='#include <%s>\n\n' % header_name)
        return outputs

    def _get_base_command(self, command_file):
        commands = None
        with open(command_file, 'r') as f:
            commands = json.load(f)
        for compile_cmd in commands:
            file = compile_cmd['file']
            if not file.endswith('src/algorithm.cpp'):
                continue
            wd = compile_cmd['directory']
            cmd_str = compile_cmd['command']
            cmd = shlex.split(cmd_str)
            out_arg = cmd.index('-o')
            del cmd[out_arg]
            del cmd[out_arg]
            in_arg = cmd.index('-c')
            del cmd[in_arg]
            del cmd[in_arg]
            if self.new_std is not None:
                for f in cmd:
                    if f.startswith('-std='):
                        del cmd[cmd.index(f)]
                        cmd += [self.new_std]
                        break
            return wd, cmd
        print_and_exit("failed to find command to build algorithm.cpp")

def post_process_outputs(outputs, libcxx_only):
    graphs = []
    for dot_file in outputs:
        g = dot.DirectedGraph.fromDotFile(dot_file)
        rename_headers_and_remove_test_root(g)
        if libcxx_only:
            remove_non_std_headers(g)
        graphs += [g]
        g.toDotFile(dot_file)
    return graphs

def build_canonical_names(graphs):
    canonical_names = {}
    next_idx = 0
    for g in graphs:
        for n in g.nodes:
            if n.attributes['label'] not in canonical_names:
                name = 'header_%d' % next_idx
                next_idx += 1
                canonical_names[n.attributes['label']] = name
    return canonical_names



class CanonicalGraphBuilder(object):
    def __init__(self, graphs):
        self.graphs = list(graphs)
        self.canonical_names = build_canonical_names(graphs)

    def build(self):
        self.canonical = dot.DirectedGraph('all_headers')
        for k,v in self.canonical_names.iteritems():
            n = dot.Node(v, edges=[], attributes={'shape': 'box', 'label': k})
            self.canonical.addNode(n)
        for g in self.graphs:
            self._merge_graph(g)
        return self.canonical

    def _merge_graph(self, g):
        for n in g.nodes:
            new_name = self.canonical.getNodeByLabel(n.attributes['label']).id
            for e in n.edges:
                to_node = self.canonical.getNodeByLabel(e.attributes['label']).id
                self.canonical.addEdge(new_name, to_node)


def main():
    parser = ArgumentParser(
        description="Generate a graph of libc++ header dependencies")
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', default=False)
    parser.add_argument(
        '-o', '--output', dest='output', required=True,
        help='The output file. stdout is used if not given',
        type=str, action='store')
    parser.add_argument(
        '--no-compile', dest='no_compile', action='store_true', default=False)
    parser.add_argument(
        '--libcxx-only', dest='libcxx_only', action='store_true', default=False)
    parser.add_argument(
        'compile_commands', metavar='compile-commands-file',
        help='the compile commands database')

    args = parser.parse_args()
    builder = DependencyCommand(args.compile_commands, args.output, new_std='-std=c++2a')
    if not args.no_compile:
        outputs = builder.run_for_headers(get_libcxx_headers())
        graphs = post_process_outputs(outputs, args.libcxx_only)
    else:
        outputs = [os.path.join(args.output, l) for l in os.listdir(args.output) if not l.endswith('all_headers.dot')]
        graphs = [dot.DirectedGraph.fromDotFile(o) for o in outputs]

    canon = CanonicalGraphBuilder(graphs).build()
    canon.toDotFile(os.path.join(args.output, 'all_headers.dot'))
    all_graphs = graphs + [canon]

    found_cycles = False
    for g in all_graphs:
        cycle_finder = dot.CycleFinder(g)
        all_cycles = cycle_finder.findCyclesInGraph()
        if len(all_cycles):
            found_cycles = True
            print("cycle in graph %s" % g.name)
            for start, path in all_cycles:
                print("Cycle for %s = %s" % (start, path))
    if not found_cycles:
        print("No cycles found")



if __name__ == '__main__':
    main()
