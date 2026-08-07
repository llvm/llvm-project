# RUN: %{python} %s %{libcxx-dir}/utils %{include-dir} %{target-include-dir}

import pathlib
import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import all_headers

# Check both the generic modulemap and the target-specific modulemap, if there is one.
include = pathlib.Path(sys.argv[2])
target_include = pathlib.Path(sys.argv[3])
with open(include / "module.modulemap") as f:
    modulemap = f.read()
if target_include.resolve() != include.resolve():
    with open(target_include / "module.modulemap") as f:
        modulemap += f.read()

isHeaderMissing = False
for header in all_headers:
    if not header.is_in_modulemap():
        continue

    if not str(header) in modulemap:
        print(f"Header {header} seems to be missing from the modulemap!")
        isHeaderMissing = True

if isHeaderMissing:
    exit(1)
