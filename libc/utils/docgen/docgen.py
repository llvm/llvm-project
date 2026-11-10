#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
from argparse import ArgumentParser, Namespace
import html
from pathlib import Path
from typing import Callable, Dict
import os
import sys

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
        "removed-in-posix-2024",
    ]

    # Validate macros
    if "macros" in api:
        if not header.macro_file_exists() and not any(
            header.implements_macro(m) for m in api["macros"]
        ):
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
        if not header.fns_dir_exists() and not any(
            header.implements_fn(f) for f in api["functions"]
        ):
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


def _status_badge_html(implemented: bool) -> str:
    if implemented:
        return (
            '<span class="hc-status-yes" title="Implemented">'
            '<span aria-hidden="true">&#10003;</span>'
            '<span class="hc-sr-only">Implemented</span></span>'
        )
    return (
        '<span class="hc-status-no" title="Not Implemented">'
        '<span aria-hidden="true">&ndash;</span>'
        '<span class="hc-sr-only">Not Implemented</span></span>'
    )


def _c_section_html(item: Dict) -> str:
    sec = item.get("c-definition")
    if sec:
        return f'<span class="hc-sec-badge">{html.escape(str(sec))}</span>'
    return '<span class="hc-status-no" aria-label="Not in C23">&ndash;</span>'


def _posix_link_html(header: Header, name: str, kind: str, item: Dict) -> str:
    base = (
        f"functions/{name}.html"
        if kind == "function"
        else f"basedefs/{header.name.replace('/', '_')}.html"
    )
    if "in-latest-posix" in item:
        url = f"https://pubs.opengroup.org/onlinepubs/9799919799/{base}"
        return f'<a class="hc-posix-link" href="{url}">POSIX.1-2024 &#8599;</a>'
    if "removed-in-posix-2008" in item:
        url = f"https://pubs.opengroup.org/onlinepubs/007904875/{base}"
        return f'<a class="hc-posix-link" href="{url}">Removed (2008) &#8599;</a>'
    if "removed-in-posix-2024" in item:
        url = f"https://pubs.opengroup.org/onlinepubs/9699919799.2018edition/{base}"
        return f'<a class="hc-posix-link" href="{url}">Removed (2024) &#8599;</a>'
    return '<span class="hc-status-no" aria-label="Not in POSIX">&ndash;</span>'


def print_symbol_table_md(
    header: Header,
    section_title: str,
    col_label: str,
    items: Dict,
    is_implemented_fn: Callable[[str], bool],
    kind: str,
):
    print(
        f"## {section_title}\n\n```{{raw}} html\n"
        '<div class="hc-table-wrap"><table class="hc-table">\n'
        f"  <thead><tr><th>{col_label}</th>"
        '<th class="hc-center">Status</th>'
        '<th class="hc-center">C23 Section</th>'
        '<th class="hc-center">POSIX Docs</th></tr></thead>\n'
        "  <tbody>"
    )
    for name in sorted(items.keys()):
        obj = items[name] or {}
        row_id = html.escape(f"{kind}-{name.replace(' ', '-')}", quote=True)
        status = _status_badge_html(is_implemented_fn(name))
        c_sec = _c_section_html(obj)
        posix = _posix_link_html(header, name, kind, obj)
        print(
            f'    <tr id="{row_id}">'
            f'<td><code class="hc-sym">{html.escape(name)}</code></td>'
            f'<td class="hc-center">{status}</td>'
            f'<td class="hc-center">{c_sec}</td>'
            f'<td class="hc-center">{posix}</td></tr>'
        )
    print("  </tbody>\n</table></div>\n```\n")


def print_header_status_md(header: Header, api: Dict):
    print(f"# {header.name}\n")

    if api.get("macros"):
        print_symbol_table_md(
            header,
            "Macros",
            "Macro",
            api["macros"],
            header.implements_macro,
            "macro",
        )

    if api.get("functions"):
        print_symbol_table_md(
            header,
            "Functions",
            "Function",
            api["functions"],
            header.implements_fn,
            "function",
        )


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
    return sorted(choices)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("header_name", choices=get_choices())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    header = Header(args.header_name)
    check_api(header, header.api)

    print_header_status_md(header, header.api)
