#!/usr/bin/env python
#
# ====- Generate documentation for libc functions  ------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import sys
import json


def load_api(hname: str) -> Dict:
    p = Path(__file__).parent / Path(hname).with_suffix(".json")
    api = p.read_text(encoding="utf-8")
    return json.loads(api)


# TODO: we may need to get more sophisticated for less generic implementations.
# Does libc/src/{hname minus .h suffix}/{fname}.cpp exist?
def is_implemented(hname: str, fname: str) -> bool:
    return Path(
        Path(__file__).parent.parent.parent,
        "src",
        hname.rstrip(".h"),
        fname + ".cpp",
    ).exists()


def print_functions(header: str, functions: Dict):
    for key in sorted(functions.keys()):
        print(f"  * - {key}")

        if is_implemented(header, key):
            print("    - |check|")
        else:
            print("    -")

        # defined is optional. Having any content is optional.
        if functions[key] is not None and "defined" in functions[key]:
            print(f'    - {functions[key]["defined"]}')
        else:
            print("    -")


def print_header(header: str, api: Dict):
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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    choices = [p.with_suffix(".h").name for p in Path(__file__).parent.glob("*.json")]
    parser.add_argument("header_name", choices=choices)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    api = load_api(args.header_name)

    print_header(args.header_name, api)
