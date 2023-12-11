# RUN: %{python} %s %{libcxx}/utils %{include}

import sys

sys.path.append(sys.argv[1])

import pathlib
import sys
from libcxx.header_information import is_modulemap_header, is_header

headers = list(pathlib.Path(sys.argv[2]).rglob("*"))
modulemap = open(f"{sys.argv[2]}/module.modulemap").read()

isHeaderMissing = False

for header in headers:
    if not is_header(header):
        continue

    header = header.relative_to(pathlib.Path(sys.argv[2])).as_posix()

    if not is_modulemap_header(header):
        continue

    if not str(header) in modulemap:
        print(f"Header {header} seems to be missing from the modulemap!")
        isHeaderMissing = True

if isHeaderMissing:
    exit(1)
