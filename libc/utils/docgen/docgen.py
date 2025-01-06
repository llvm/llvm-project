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
import os
import sys
import yaml

from header import Header


class DocgenAPIFormatError(Exception):
    """Raised on fatal formatting errors with a description of a formatting error"""


def check_api(header: Header, api: Dict):
    """
    Checks that docgen yaml files are properly formatted. If there are any
    fatal formatting errors, raises exceptions with error messages useful for
    fixing formatting. Warnings are printed to stderr on non-fatal formatting
    errors. The code that runs after ``check_api(api)`` is called expects that
    ``check_api`` executed without raising formatting exceptions so the yaml
    matches the formatting specified here.

    The yaml file may contain:
    * an optional macros object
    * an optional functions object

    Formatting of ``macros`` and ``functions`` objects
    ==================================================

    If a macros or functions object is present, then it may contain nested
    objects. Each of these nested objects should have a name matching a macro
    or function's name, and each nested object must have the property:
    ``"c-definition"`` or ``"posix-definition"``.

    Description of properties
    =========================
    The defined property is intended to be a reference to a part of the
    standard that defines the function or macro. For the ``"c-definition"`` property,
    this should be a C standard section number. For the ``"posix-definition"`` property,
    this should be a link to the definition.

    :param api: docgen yaml file contents parsed into a dict
    """
    errors = []
    # We require entries to have at least one of these.
    possible_keys = [
        "c-definition",
        "in-latest-posix",
        "removed-in-posix-2008",
    ]

    # Validate macros
    if "macros" in api:
        if not header.macro_file_exists():
            print(
                f"warning: Macro definitions are listed for {header.name}, but no macro file can be found in the directory tree rooted at {header.macros_dir}. All macros will be listed as not implemented.",
                file=sys.stderr,
            )

        macros = api["macros"]

        for name, obj in macros.items():
            if not any(k in obj for k in possible_keys):
                err = f"error: Macro {name} does not contain at least one required property: {possible_keys}"
                errors.append(err)

    # Validate functions
    if "functions" in api:
        if not header.fns_dir_exists():
            print(
                f"warning: Function definitions are listed for {header.name}, but no function implementation directory exists at {header.fns_dir}. All functions will be listed as not implemented.",
                file=sys.stderr,
            )

        fns = api["functions"]
        for name, obj in fns.items():
            if not any(k in obj for k in possible_keys):
                err = f"error: function {name} does not contain at least one required property: {possible_keys}"
                errors.append(err)

    if errors:
        raise DocgenAPIFormatError("\n".join(errors))


def load_api(header: Header) -> Dict:
    api = header.docgen_yaml.read_text(encoding="utf-8")
    return yaml.safe_load(api)


def print_tbl_dir(name):
    print(
        f"""
.. list-table::
  :widths: auto
  :align: center
  :header-rows: 1

  * - {name}
    - Implemented
    - C23 Standard Section
    - POSIX Docs"""
    )


def print_functions_rst(header: Header, functions: Dict):
    tbl_hdr = "Functions"
    print(tbl_hdr)
    print("=" * len(tbl_hdr))

    print_tbl_dir("Function")

    for name in sorted(functions.keys()):
        print(f"  * - {name}")

        if header.fns_dir_exists() and header.implements_fn(name):
            print("    - |check|")
        else:
            print("    -")

        if "c-definition" in functions[name]:
            print(f'    - {functions[name]["c-definition"]}')
        else:
            print("    -")

        if "in-latest-posix" in functions[name]:
            print(
                f"    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/functions/{name}.html>`__"
            )
        elif "removed-in-posix-2008" in functions[name]:
            print(
                f"    - `removed in POSIX.1-2008 <https://pubs.opengroup.org/onlinepubs/007904875/functions/{name}.html>`__"
            )
        else:
            print("    -")


def print_macros_rst(header: Header, macros: Dict):
    tbl_hdr = "Macros"
    print(tbl_hdr)
    print("=" * len(tbl_hdr))

    print_tbl_dir("Macro")

    for name in sorted(macros.keys()):
        print(f"  * - {name}")

        if header.macro_file_exists() and header.implements_macro(name):
            print("    - |check|")
        else:
            print("    -")

        if "c-definition" in macros[name]:
            print(f'    - {macros[name]["c-definition"]}')
        else:
            print("    -")

        if "in-latest-posix" in macros[name]:
            print(
                f"    - `POSIX.1-2024 <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/{header.name}.html>`__"
            )
        else:
            print("    -")
    print()


def print_impl_status_rst(header: Header, api: Dict):
    if os.sep in header.name:
        print(".. include:: ../../check.rst\n")
    else:
        print(".. include:: ../check.rst\n")

    print("=" * len(header.name))
    print(header.name)
    print("=" * len(header.name))
    print()

    # the macro and function sections are both optional
    if "macros" in api:
        print_macros_rst(header, api["macros"])

    if "functions" in api:
        print_functions_rst(header, api["functions"])


# This code implicitly relies on docgen.py being in the same dir as the yaml
# files and is likely to need to be fixed when re-integrating docgen into
# hdrgen.
def get_choices() -> list:
    choices = []
    for path in Path(__file__).parent.rglob("*.yaml"):
        fname = path.with_suffix(".h").name
        if path.parent != Path(__file__).parent:
            fname = path.parent.name + os.sep + fname
        choices.append(fname)
    return choices


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("header_name", choices=get_choices())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    header = Header(args.header_name)
    api = load_api(header)
    check_api(header, api)

    print_impl_status_rst(header, api)
