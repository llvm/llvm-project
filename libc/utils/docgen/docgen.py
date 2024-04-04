#!/usr/bin/env python
#
# ====- Generate documentation for libc functions  ------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#
from argparse import ArgumentParser
from pathlib import Path
import sys
import yaml


def load_api():
    p = Path(Path(__file__).resolve().parent, "api.yml")
    api = p.read_text(encoding="utf-8")
    return yaml.load(api, Loader=yaml.FullLoader)


# TODO: we may need to get more sophisticated for less generic implementations.
# Does libc/src/{hname minus .h suffix}/{fname}.cpp exist?
def is_implemented(hname, fname):
    return Path(
        Path(__file__).resolve().parent.parent.parent,
        "src",
        hname.rstrip(".h"),
        fname + ".cpp",
    ).exists()


def print_functions(header, functions):
    for key in sorted(functions.keys()):
        print(f"  * - {key}")

        if is_implemented(header, key):
            print("    - |check|")
        else:
            print("    -")

        if "defined" in functions[key]:
            print(f'    - {functions[key]["defined"]}')
        else:
            print("    -")


def print_header(header, api):
    fns = f"{header} Functions"
    print(fns)
    print("=" * (len(fns)))
    print(
        f"""
.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - Function
    - Implemented
    - Standard"""
    )
    # TODO: how do we want to signal implementation of macros?
    print_functions(header, api["functions"])


def parse_args(header_choices):
    parser = ArgumentParser()
    parser.add_argument("header_name", choices=header_choices)
    return parser.parse_args()


if __name__ == "__main__":
    api = load_api()
    args = parse_args(api.keys())

    print_header(args.header_name, api[args.header_name])
