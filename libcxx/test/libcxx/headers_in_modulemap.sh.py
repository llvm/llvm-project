# RUN: %{python} %s %{libcxx-dir}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import all_headers, libcxx_include

with open(libcxx_include / "module.modulemap") as f:
    modulemap = f.read()

isHeaderMissing = False
for header in all_headers:
    if not header.is_in_modulemap():
        continue

    if not str(header) in modulemap:
        print(f"Header {header} seems to be missing from the modulemap!")
        isHeaderMissing = True

if isHeaderMissing:
    exit(1)
