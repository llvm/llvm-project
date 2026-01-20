#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

def step(message: str) -> None:
    print(message, file=sys.stderr)

def directory_path(string):
    if os.path.isdir(string):
        return pathlib.Path(string)
    else:
        raise NotADirectoryError(string)

def main(argv):
    parser = argparse.ArgumentParser(
        prog='benchmark-for-lnt',
        description='Benchmark libc++ at the given commit for submitting to LNT.')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='-',
        help='Path to the file where the resulting LNT report containing benchmark results is written. '
             'By default, stdout.')
    parser.add_argument('--benchmark-commit', type=str, required=True,
        help='The SHA representing the version of the library to benchmark.')
    parser.add_argument('--test-suite-commit', type=str, required=True,
        help='The SHA representing the version of the test suite to use for benchmarking.')
    parser.add_argument('--machine', type=str, required=True,
        help='The name of the machine for reporting LNT results.')
    parser.add_argument('--spec-dir', type=pathlib.Path, required=False,
        help='Optional path to a SPEC installation to use for benchmarking.')
    parser.add_argument('--git-repo', type=directory_path, default=os.getcwd(),
        help='Optional path to the Git repository to use. By default, the current working directory is used.')
    parser.add_argument('--dry-run', action='store_true',
        help='Only print what would be executed.')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Print the output of all subcommands.')
    args = parser.parse_args(argv)

    def run(command, *posargs, **kwargs):
        command = [str(c) for c in command]
        if args.dry_run:
            print(f'$ {" ".join(command)}')
        else:
            # If we're running with verbose, print everything but redirect output to stderr since
            # we already output the json to stdout in some cases. Otherwise, hush everything.
            if args.verbose:
                if 'stdout' not in kwargs:
                    kwargs.update({'stdout': sys.stderr})
            else:
                if 'stdout' not in kwargs:
                    kwargs.update({'stdout': subprocess.DEVNULL})
                if 'stderr' not in kwargs:
                    kwargs.update({'stderr': subprocess.DEVNULL})
            subprocess.check_call(command, *posargs, **kwargs)

    with tempfile.TemporaryDirectory() as build_dir:
        build_dir = pathlib.Path(build_dir)

        step(f'Building libc++ at commit {args.benchmark_commit}')
        run([args.git_repo / 'libcxx/utils/build-at-commit',
                        '--git-repo', args.git_repo,
                        '--install-dir', build_dir / 'install',
                        '--commit', args.benchmark_commit,
                        '--', '-DCMAKE_BUILD_TYPE=RelWithDebInfo'])

        if args.spec_dir is not None:
            step(f'Running SPEC benchmarks from {args.test_suite_commit} against libc++ {args.benchmark_commit}')
            run([args.git_repo / 'libcxx/utils/test-at-commit',
                        '--git-repo', args.git_repo,
                        '--build-dir', build_dir / 'spec',
                        '--test-suite-commit', args.test_suite_commit,
                        '--libcxx-installation', build_dir / 'install',
                        '--',
                        '-j1', '--time-tests',
                        '--param', 'optimization=speed',
                        '--param', 'std=c++17',
                        '--param', f'spec_dir={args.spec_dir}',
                        build_dir / 'spec/libcxx/test',
                        '--filter', 'benchmarks/spec.gen.py'])

        # TODO: For now, we run only a subset of the benchmarks because running the whole test suite is too slow.
        #       Run the whole test suite once https://github.com/llvm/llvm-project/issues/173032 is resolved.
        step(f'Running microbenchmarks from {args.test_suite_commit} against libc++ {args.benchmark_commit}')
        run([args.git_repo / 'libcxx/utils/test-at-commit',
                        '--git-repo', args.git_repo,
                        '--build-dir', build_dir / 'micro',
                        '--test-suite-commit', args.test_suite_commit,
                        '--libcxx-installation', build_dir / 'install',
                        '--',
                        '-j1', '--time-tests',
                        '--param', 'optimization=speed',
                        '--param', 'std=c++26',
                        build_dir / 'micro/libcxx/test',
                        '--filter', 'benchmarks/(algorithms|containers|iterators|locale|memory|streams|numeric|utility)'])

        step('Installing LNT')
        run(['python', '-m', 'venv', build_dir / '.venv'])
        run([build_dir / '.venv/bin/pip', 'install', 'llvm-lnt'])

        step('Consolidating benchmark results and creating JSON report')
        if args.spec_dir is not None:
            with open(build_dir / 'benchmarks.lnt', 'w') as f:
                run([args.git_repo / 'libcxx/utils/consolidate-benchmarks', build_dir / 'spec'], stdout=f)
        with open(build_dir / 'benchmarks.lnt', 'a') as f:
            run([args.git_repo / 'libcxx/utils/consolidate-benchmarks', build_dir / 'micro'], stdout=f)
        order = len(subprocess.check_output(['git', '-C', args.git_repo, 'rev-list', args.benchmark_commit]).splitlines())
        commit_info = subprocess.check_output(['git', '-C', args.git_repo, 'show', args.benchmark_commit, '--no-patch']).decode()
        run([build_dir / '.venv/bin/lnt', 'importreport', '--order', str(order), '--machine', args.machine,
                '--run-info', f'commit_info={commit_info}',
                build_dir / 'benchmarks.lnt', build_dir / 'benchmarks.json'])

        if not args.dry_run:
            with open(build_dir / 'benchmarks.json', 'r') as f:
                args.output.write(f.read())


if __name__ == '__main__':
    main(sys.argv[1:])
