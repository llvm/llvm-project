# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# REQUIRES: enable-spec-benchmarks

# RUN: mkdir -p %T
# RUN: echo "%{cxx}" > %T/cxx.subs
# RUN: echo "%{compile_flags}" > %T/compile_flags.subs
# RUN: echo "%{flags}" > %T/flags.subs
# RUN: echo "%{link_flags}" > %T/link_flags.subs
# RUN: echo "%{spec_dir}" > %T/spec_dir.subs
# RUN: %{python} %s %T
# END.

import json
import pathlib
import sys

test_dir = pathlib.Path(sys.argv[1])
cxx = (test_dir / 'cxx.subs').open().read().strip()
compile_flags = (test_dir / 'compile_flags.subs').open().read().strip()
flags = (test_dir / 'flags.subs').open().read().strip()
link_flags = (test_dir / 'link_flags.subs').open().read().strip()
spec_dir = pathlib.Path((test_dir / 'spec_dir.subs').open().read().strip())

# Setup the configuration file
test_dir.mkdir(parents=True, exist_ok=True)
spec_config = test_dir / 'spec-config.cfg'
spec_config.write_text(f"""
default:
    ignore_errors        = 1
    iterations           = 1
    label                = spec-stdlib
    log_line_width       = 4096
    makeflags            = --jobs=8
    mean_anyway          = 1
    output_format        = csv
    preenv               = 0
    reportable           = 0
    tune                 = base
    copies               = 1
    threads              = 1
    CC                   = cc -O3
    CXX                  = {cxx} {compile_flags} {flags} {link_flags} -Wno-error
    CC_VERSION_OPTION    = --version
    CXX_VERSION_OPTION   = --version
    EXTRA_PORTABILITY    = -DSPEC_NO_CXX17_SPECIAL_MATH_FUNCTIONS # because libc++ doesn't implement the special math functions yet
""")

# Build the list of benchmarks. We take all intrate and fprate benchmarks that contain C++ and
# discard the ones that contain Fortran, since this test suite isn't set up to build Fortran code.
spec_benchmarks = set()
no_fortran = set()
with open(spec_dir / 'benchspec' / 'CPU' / 'intrate_any_cpp.bset', 'r') as f:
    spec_benchmarks.update(json.load(f)['benchmarks'])
with open(spec_dir / 'benchspec' / 'CPU' / 'fprate_any_cpp.bset', 'r') as f:
    spec_benchmarks.update(json.load(f)['benchmarks'])
with open(spec_dir / 'benchspec' / 'CPU' / 'no_fortran.bset', 'r') as f:
    no_fortran.update(json.load(f)['benchmarks'])
spec_benchmarks &= no_fortran

for benchmark in spec_benchmarks:
    print(f'#--- {benchmark}.sh.test')
    print(f'RUN: rm -rf %T') # clean up any previous (potentially incomplete) run
    print(f'RUN: mkdir %T')
    print(f'RUN: cp {spec_config} %T/spec-config.cfg')
    print(f'RUN: %{{spec_dir}}/bin/runcpu --config %T/spec-config.cfg --size train --output-root %T --rebuild {benchmark}')
    print(f'RUN: rm -rf %T/benchspec') # remove the temporary directory, which can become quite large

    # Parse the results into a LNT-compatible format. This also errors out if there are no CSV files, which
    # means that the benchmark didn't run properly (the `runcpu` command above never reports a failure).
    print(f'RUN: %{{libcxx-dir}}/utils/parse-spec-results %T/result/CPUv8.001.*.train.csv --output-format=lnt > %T/results.lnt')
    print(f'RUN: cat %T/results.lnt')
