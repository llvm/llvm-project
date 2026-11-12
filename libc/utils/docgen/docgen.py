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
import os
from pathlib import Path
import re
import sys
from typing import Callable, Dict, Optional

from header import Header

_CPPREFERENCE_BASE = "https://en.cppreference.com/w/c"


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
    for key in ("c-reference-root", "standard", "posix", "provider"):
        if key in api and not isinstance(api[key], str):
            errors.append(f"error: {key} must be a string")
    # We require entries to have at least one of these.
    possible_keys = [
        "c-definition",
        "in-latest-posix",
        "removed-in-posix-2008",
        "removed-in-posix-2024",
        "extension",
    ]

    if "macros" in api and (
        not header.macro_file_exists()
        and not any(header.implements_macro(m) for m in api["macros"])
    ):
        print(
            f"warning: Macro definitions are listed for {header.name}, "
            "but no macro file can be found in the directory tree rooted "
            f"at {header.macros_dir}. All macros will be listed as not "
            "implemented.",
            file=sys.stderr,
        )

    if "functions" in api and (
        not header.fns_dir_exists()
        and not any(header.implements_fn(f) for f in api["functions"])
    ):
        print(
            f"warning: Function definitions are listed for {header.name}, "
            "but no function implementation directory exists at "
            f"{header.fns_dir}. All functions will be listed as not "
            "implemented.",
            file=sys.stderr,
        )

    for section, label in (
        ("macros", "Macro"),
        ("functions", "function"),
        ("types", "type"),
        ("objects", "object"),
    ):
        if section not in api:
            continue
        entries = api[section]
        if not isinstance(entries, dict):
            errors.append(f"error: {section} must be a dict")
            continue
        for name, obj in entries.items():
            if not isinstance(obj, dict) or not any(k in obj for k in possible_keys):
                errors.append(
                    f"error: {label} {name} does not contain at least one "
                    f"required property: {possible_keys}"
                )
                continue
            if "c-reference" in obj and not isinstance(obj["c-reference"], str):
                errors.append(f"error: {label} {name} c-reference must be a string")

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


def cppreference_url(
    header: Header,
    symbol_name: str = "",
    kind: str = "function",
    item: Optional[Dict] = None,
    api: Optional[Dict] = None,
) -> str:
    if item and "c-reference" in item:
        ref = item["c-reference"]
        return f"{_CPPREFERENCE_BASE}/{ref}" if ref else ""
    root = (api if api is not None else header.api).get("c-reference-root")
    if not root:
        return ""
    if (
        not symbol_name
        or symbol_name.startswith("__")
        or kind not in ("function", "family")
    ):
        return f"{_CPPREFERENCE_BASE}/{root}"
    return f"{_CPPREFERENCE_BASE}/{root}/{symbol_name}"


def posix_url(header: Header, symbol_name: str, kind: str, item: Dict) -> str:
    subpath = (
        f"functions/{symbol_name}.html"
        if kind in ("function", "family") and symbol_name
        else f"basedefs/{header.name.replace('/', '_')}.html"
    )
    if "in-latest-posix" in item:
        return f"https://pubs.opengroup.org/onlinepubs/9799919799/{subpath}"
    if "removed-in-posix-2008" in item:
        return f"https://pubs.opengroup.org/onlinepubs/007904875/{subpath}"
    if "removed-in-posix-2024" in item:
        return f"https://pubs.opengroup.org/onlinepubs/9699919799.2018edition/{subpath}"
    return ""


_C_SECTION_RE = re.compile(r"^([0-9]+|[FGHJK])\.")


def _c_section_html(
    header: Header,
    name: str,
    kind: str,
    item: Dict,
    api: Optional[Dict] = None,
) -> str:
    sec = item.get("c-definition")
    if not sec:
        return '<span class="hc-status-no" aria-label="Not in C23">&ndash;</span>'
    label = html.escape(str(sec))
    if _C_SECTION_RE.match(str(sec)):
        url = cppreference_url(header, name, kind, item, api)
        if url:
            href = html.escape(url, quote=True)
            return (
                f'<a class="hc-c-link" href="{href}" '
                f'title="C23 section {label} on cppreference.com">'
                f'{label} <span aria-hidden="true">&#8599;</span></a>'
            )
    return f'<span class="hc-sec-badge">{label}</span>'


def _posix_link_html(header: Header, name: str, kind: str, item: Dict) -> str:
    url = posix_url(header, name, kind, item)
    if not url:
        return '<span class="hc-status-no" aria-label="Not in POSIX">&ndash;</span>'
    if "removed-in-posix-2008" in item:
        label = "Removed (2008)"
    elif "removed-in-posix-2024" in item:
        label = "Removed (2024)"
    else:
        label = "POSIX.1-2024"
    href = html.escape(url, quote=True)
    return (
        f'<a class="hc-posix-link" href="{href}">'
        f'{label} <span aria-hidden="true">&#8599;</span></a>'
    )


def print_symbol_table_md(
    header: Header,
    section_title: str,
    col_label: str,
    items: Dict,
    is_implemented_fn: Callable[[str], bool],
    kind: str,
    api: Optional[Dict] = None,
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
        c_sec = _c_section_html(header, name, kind, obj, api)
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

    if api.get("types"):
        print_symbol_table_md(
            header,
            "Types",
            "Type",
            api["types"],
            header.implements_type,
            "type",
            api,
        )

    if api.get("objects"):
        print_symbol_table_md(
            header,
            "Global Variables",
            "Global Variable",
            api["objects"],
            header.implements_object,
            "object",
            api,
        )

    if api.get("macros"):
        print_symbol_table_md(
            header,
            "Macros",
            "Macro",
            api["macros"],
            header.implements_macro,
            "macro",
            api,
        )

    if api.get("functions"):
        print_symbol_table_md(
            header,
            "Functions",
            "Function",
            api["functions"],
            header.implements_fn,
            "function",
            api,
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
