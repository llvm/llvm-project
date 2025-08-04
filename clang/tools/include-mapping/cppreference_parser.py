#!/usr/bin/env python3
# ===- cppreference_parser.py -  ------------------------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#

from bs4 import BeautifulSoup, NavigableString, Tag

import collections
import multiprocessing
import os
import re
import signal
import sys


class Symbol:
    def __init__(self, name, namespace, headers):
        # unqualifed symbol name, e.g. "move"
        self.name = name
        # namespace of the symbol (with trailing "::"), e.g. "std::", "" (global scope)
        # None for C symbols.
        self.namespace = namespace
        # a list of corresponding headers
        self.headers = headers

    def __lt__(self, other):
        if self.namespace != other.namespace:
            return str(self.namespace) < str(other.namespace)
        return self.name < other.name

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return (
            self.name == other.name
            and self.namespace == other.namespace
            and set(self.headers) == set(other.headers)
        )

    def __hash__(self):
        return hash((self.name, self.namespace, tuple(self.headers)))


def _HasClass(tag, *classes):
    for c in tag.get("class", []):
        if c in classes:
            return True
    return False


def _ParseSymbolPage(symbol_page_html, symbol_name, qual_name):
    """Parse symbol page and retrieve the include header defined in this page.
    The symbol page provides header for the symbol, specifically in
    "Defined in header <header>" section. An example:

    <tr class="t-dsc-header">
      <td colspan="2"> <div>Defined in header <code>&lt;ratio&gt;</code> </div>
    </td></tr>

    Returns a list of headers.
    """
    headers = []
    current_headers = []

    soup = BeautifulSoup(symbol_page_html, "html.parser")
    # Rows in table are like:
    #   Defined in header <foo>      .t-dsc-header
    #   Defined in header <bar>      .t-dsc-header
    #   decl1                        .t-dcl
    #   Defined in header <baz>      .t-dsc-header
    #   decl2                        .t-dcl
    for table in soup.select("table.t-dcl-begin, table.t-dsc-begin"):
        header_found = False
        was_decl = False
        for row in table.select("tr"):
            if _HasClass(row, "t-dcl", "t-dsc"):
                was_decl = True
                # Break before "See also" section, to avoid false positives
                if any(_HasClass(tag, "t-dsc-see-tt") for tag in row.find_all(True)):
                    break
                # Symbols are usually in the first cell, except alternative operators.
                for col in row.select("td"):
                    found_symbols = []
                    for sym in col.stripped_strings:
                        found_symbols.extend(re.findall(r"[A-Za-z_]\w*", sym))
                    if any(
                        sym == symbol_name or sym == qual_name for sym in found_symbols
                    ):
                        headers.extend(current_headers)
                    if headers:
                        break
            elif _HasClass(row, "t-dsc-header"):
                # There are also .t-dsc-header for "defined in namespace".
                if not "Defined in header " in row.text:
                    continue
                # If we saw a decl since the last header, this is a new block of headers
                # for a new block of decls.
                if not header_found or was_decl:
                    current_headers = []
                header_found = True
                was_decl = False
                # The interesting header content (e.g. <cstdlib>) is wrapped in <code>.
                for header_code in row.find_all("code"):
                    current_headers.append(header_code.text)
    return headers


def _ParseSymbolVariant(caption):
    if not (isinstance(caption, NavigableString) and "(" in caption):
        return None

    if ")" in caption.text:  # (locale), (algorithm), etc.
        return caption.text.strip(" ()")

    second_part = caption.next_sibling
    if isinstance(second_part, Tag) and second_part.name == "code":
        # (<code>std::complex</code>), etc.
        third_part = second_part.next_sibling
        if isinstance(third_part, NavigableString) and third_part.text.startswith(")"):
            return second_part.text
    return None


def _ParseIndexPage(index_page_html):
    """Parse index page.
    The index page lists all std symbols and hrefs to their detailed pages
    (which contain the defined header). An example:

    <a href="abs.html" title="abs"><tt>abs()</tt></a> (int) <br>
    <a href="acos.html" title="acos"><tt>acos()</tt></a> <br>

    Returns a list of tuple (symbol_name, relative_path_to_symbol_page, variant).
    """
    symbols = []
    soup = BeautifulSoup(index_page_html, "html.parser")
    for symbol_href in soup.select("a[title]"):
        # Ignore annotated symbols like "acos<>() (std::complex)".
        # These tend to be overloads, and we the primary is more useful.
        # This accidentally accepts begin/end despite the (iterator) caption: the
        # (since C++11) note is first. They are good symbols, so the bug is unfixed.
        caption = symbol_href.next_sibling
        variant = _ParseSymbolVariant(caption)
        symbol_tt = symbol_href.find("tt")
        if symbol_tt:
            symbols.append(
                (
                    symbol_tt.text.rstrip("<>()"),  # strip any trailing <>()
                    symbol_href["href"],
                    variant,
                )
            )
    return symbols


def _ReadSymbolPage(path, name, qual_name):
    with open(path, encoding="utf-8") as f:
        return _ParseSymbolPage(f.read(), name, qual_name)


def _GetSymbols(pool, root_dir, index_page_name, namespace, variants_to_accept):
    """Get all symbols listed in the index page. All symbols should be in the
    given namespace.

    Returns a list of Symbols.
    """

    # Workflow steps:
    #   1. Parse index page which lists all symbols to get symbol
    #      name (unqualified name) and its href link to the symbol page which
    #      contains the defined header.
    #   2. Parse the symbol page to get the defined header.
    index_page_path = os.path.join(root_dir, index_page_name)
    with open(index_page_path, "r", encoding="utf-8") as f:
        # Read each symbol page in parallel.
        results = []  # (symbol_name, promise of [header...])
        for symbol_name, symbol_page_path, variant in _ParseIndexPage(f.read()):
            # Variant symbols (e.g. the std::locale version of isalpha) add ambiguity.
            # FIXME: use these as a fallback rather than ignoring entirely.
            qualified_symbol_name = (namespace or "") + symbol_name
            variants_for_symbol = variants_to_accept.get(qualified_symbol_name, ())
            if variant and variant not in variants_for_symbol:
                continue
            path = os.path.join(root_dir, symbol_page_path)
            if os.path.isfile(path):
                results.append(
                    (
                        symbol_name,
                        pool.apply_async(
                            _ReadSymbolPage, (path, symbol_name, qualified_symbol_name)
                        ),
                    )
                )
            else:
                sys.stderr.write(
                    "Discarding information for symbol: %s. Page %s does not exist.\n"
                    % (symbol_name, path)
                )

        # Build map from symbol name to a set of headers.
        symbol_headers = collections.defaultdict(list)
        for symbol_name, lazy_headers in results:
            headers = lazy_headers.get()
            if symbol_name not in symbol_headers:
                for header in headers:
                    if header not in symbol_headers[symbol_name]:
                        symbol_headers[symbol_name].append(header)
            elif set(symbol_headers[symbol_name]) != set(headers):
                symbol_headers[symbol_name] = []

    symbols = []
    for name, headers in sorted(symbol_headers.items(), key=lambda t: t[0]):
        symbols.append(Symbol(name, namespace, list(headers)))
    return symbols


def signal_ignore_initializer():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)


def GetSymbols(parse_pages):
    """Get all symbols by parsing the given pages.

    Args:
      parse_pages: a list of tuples (page_root_dir, index_page_name, namespace)
    """
    # By default we prefer the non-variant versions, as they're more common. But
    # there are some symbols, whose variant is more common. This list describes
    # those symbols.
    variants_to_accept = {
        # std::remove<> has variant algorithm.
        "std::remove": ("algorithm"),
        # These functions don't have a generic version, and all variants are defined in <chrono>
        "std::chrono::abs": ("std::chrono::duration"),
        "std::chrono::ceil": ("std::chrono::duration"),
        "std::chrono::floor": ("std::chrono::duration"),
        "std::chrono::from_stream": ("std::chrono::day"),
        "std::chrono::round": ("std::chrono::duration"),
        # Same, but in <filesystem>
        "std::filesystem::begin": ("std::filesystem::directory_iterator"),
        "std::filesystem::end": ("std::filesystem::directory_iterator"),
        "std::ranges::get": ("std::ranges::subrange"),
    }
    symbols = []
    # Run many workers to process individual symbol pages under the symbol index.
    # Don't allow workers to capture Ctrl-C.
    pool = multiprocessing.Pool(initializer=signal_ignore_initializer)
    try:
        for root_dir, page_name, namespace in parse_pages:
            symbols.extend(
                _GetSymbols(pool, root_dir, page_name, namespace, variants_to_accept)
            )
    finally:
        pool.terminate()
        pool.join()
    return sorted(set(symbols))
