#!/usr/bin/env python3
"""Trace libc++ header dependencies via clang -E -H.

Outputs a TSV manifest (type, relative_path, absolute_path) for
EmbedLibcxxHeaders.cmake to embed.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


def trace_headers(clang, libcxx_dir, config_site, target, headers):
    """Run clang -E -H to discover all transitive header dependencies."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        for h in headers:
            f.write(f'#include <{h}>\n')
        test_file = f.name

    config_dir = tempfile.mkdtemp()
    shutil.copy2(os.path.abspath(config_site),
                 os.path.join(config_dir, '__config_site'))

    cmd = [
        clang, '-E', '-H',
        '-nostdinc++', '-nostdlibinc',
        '-x', 'c++', '-std=c++17',
        f'--target={target}',
        '-isystem', config_dir,
        '-isystem', libcxx_dir,
        test_file,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.unlink(test_file)
        os.unlink(os.path.join(config_dir, '__config_site'))
        os.rmdir(config_dir)

    if result.returncode != 0:
        non_trace = [l for l in result.stderr.splitlines()
                     if not l.startswith('.')]
        print(f'warning: clang -H exited with code {result.returncode}',
              file=sys.stderr)
        for line in non_trace:
            print(f'  {line}', file=sys.stderr)

    libcxx_real = os.path.realpath(libcxx_dir) + '/'
    libcxx_headers = set()
    clang_headers = set()

    for line in result.stderr.splitlines():
        m = re.match(r'^\.+ (.+)$', line)
        if not m:
            continue
        path = os.path.realpath(m.group(1).strip())
        if path.startswith(libcxx_real):
            libcxx_headers.add(path)
        elif '/lib/clang/' in path and '/include/' in path:
            clang_headers.add(path)

    return sorted(libcxx_headers), sorted(clang_headers), libcxx_real


def clang_resource_prefix(path):
    """Extract the resource dir prefix from a clang resource header path.

    E.g., /build/lib/clang/23/include/stdint.h -> /build/lib/clang/23/include/
    """
    idx = path.find('/lib/clang/')
    if idx < 0:
        return None
    inc_idx = path.find('/include/', idx)
    if inc_idx < 0:
        return None
    return path[:inc_idx + len('/include/')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clang', required=True)
    parser.add_argument('--libcxx-dir', required=True)
    parser.add_argument('--config-site', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--headers', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    libcxx_headers, clang_headers, libcxx_real = trace_headers(
        args.clang, args.libcxx_dir, args.config_site, args.target,
        args.headers)

    clang_resource_dir = None
    for h in clang_headers:
        clang_resource_dir = clang_resource_prefix(h)
        if clang_resource_dir:
            break

    print(f'Header trace: {len(libcxx_headers)} libc++ files, '
          f'{len(clang_headers)} Clang builtin files', file=sys.stderr)

    entries = []

    # Custom __config_site for HIPRTC
    entries.append(('libcxx', '__config_site', os.path.abspath(args.config_site)))

    # __assertion_handler if it exists in the vendor directory
    assertion_handler = os.path.join(
        os.path.dirname(args.libcxx_dir),
        'vendor', 'llvm', 'default_assertion_handler.in')
    if os.path.exists(assertion_handler):
        entries.append(('libcxx', '__assertion_handler',
                       os.path.abspath(assertion_handler)))

    for path in libcxx_headers:
        rel = os.path.relpath(path, libcxx_real)
        entries.append(('libcxx', rel, path))

    for path in clang_headers:
        if clang_resource_dir:
            rel = os.path.relpath(path, clang_resource_dir)
        else:
            rel = os.path.basename(path)
        entries.append(('clang', rel, path))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for type_name, rel_path, abs_path in entries:
            f.write(f'{type_name}\t{rel_path}\t{abs_path}\n')

    print(f'Generated manifest {args.output} with '
          f'{sum(1 for e in entries if e[0] == "libcxx")} libc++ + '
          f'{sum(1 for e in entries if e[0] == "clang")} clang headers',
          file=sys.stderr)


if __name__ == '__main__':
    main()
