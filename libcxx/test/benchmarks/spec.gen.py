# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# REQUIRES: enable-spec-benchmarks

# RUN: mkdir -p %{temp}
# RUN: echo "%{spec_dir}" > %{temp}/spec_dir.subs
# RUN: %{python} %s %{temp}
# END.

import json
import pathlib
import sys

test_dir = pathlib.Path(sys.argv[1])
spec_dir = pathlib.Path((test_dir / 'spec_dir.subs').open().read().strip())

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
    print(f'RUN: %{{python}} %{{libcxx-dir}}/utils/run-spec-benchmark --spec-dir %{{spec_dir}} --temp-dir %{{temp}} --benchmark {benchmark} --clean -- %{{cxx}} %{{compile_flags}} %{{flags}} %{{link_flags}}')
