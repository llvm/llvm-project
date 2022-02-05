#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import textwrap


def main():
    parser = argparse.ArgumentParser(description=textwrap.dedent('''
            This script builds two versions of BOLT (with the current and
            previous revision) and sets up symlink for llvm-bolt-wrapper.
            Passes the options through to llvm-bolt-wrapper.
            '''))
    parser.add_argument('build_dir', nargs='?', default=os.getcwd(),
                        help='Path to BOLT build directory, default is current directory')
    args, wrapper_args = parser.parse_known_args()
    bolt_path = f'{args.build_dir}/bin/llvm-bolt'

    source_dir = None
    # find the repo directory
    with open(f'{args.build_dir}/CMakeCache.txt') as f:
        for line in f:
            m = re.match(r'LLVM_SOURCE_DIR:STATIC=(.*)', line)
            if m:
                source_dir = m.groups()[0]
    if not source_dir:
        sys.exit("Source directory is not found")

    wrapper_path = os.path.abspath(
        f'{source_dir}/../bolt/utils/llvm-bolt-wrapper.py')
    # build the current commit
    subprocess.run(shlex.split("cmake --build . --target llvm-bolt"),
                   cwd=args.build_dir)
    # rename llvm-bolt
    os.replace(bolt_path, f'{bolt_path}.new')
    # check out the previous commit
    subprocess.run(shlex.split("git checkout -f HEAD^"),
                   cwd=source_dir)
    # build the previous commit
    subprocess.run(shlex.split("cmake --build . --target llvm-bolt"),
                   cwd=args.build_dir)
    # rename llvm-bolt
    os.replace(bolt_path, f'{bolt_path}.old')
    # set up llvm-bolt-wrapper.ini
    ini = subprocess.check_output(
        shlex.split(
            f"{wrapper_path} {bolt_path}.old {bolt_path}.new") + wrapper_args,
        text=True)
    with open(f'{args.build_dir}/bin/llvm-bolt-wrapper.ini', 'w') as f:
        f.write(ini)
    # symlink llvm-bolt-wrapper
    os.symlink(wrapper_path, bolt_path)


if __name__ == "__main__":
    main()
