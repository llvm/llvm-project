#!/usr/bin/env python
#===- gen_std.py -  ------------------------------------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

"""gen_std.py is a tool to generate a lookup table (from qualified names to
include headers) for C++ Standard Library symbols by parsing archieved HTML
files from cppreference.

Caveats and FIXMEs:
  - only symbols directly in "std" namespace are added, we should also add std's
    subnamespace symbols (e.g. chrono).
  - symbols with multiple variants or defined in multiple headers aren't added,
    e.g. std::move, std::swap

Usage:
  1. Install BeautifulSoup dependency, see instruction:
       https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-beautiful-soup
  2. Download cppreference offline HTML files (e.g. html_book_20181028.zip) at
       https://en.cppreference.com/w/Cppreference:Archives
  3. Unzip the zip file from step 2 to directory </cppreference>, you should
     get a "reference" directory in </cppreference>
  4. Run the command:
       gen_std.py -cppreference </cppreference/reference> > StdSymbolMap.inc
"""

from bs4 import BeautifulSoup, NavigableString

import argparse
import collections
import datetime
import multiprocessing
import os
import re
import signal
import sys

STDGEN_CODE_PREFIX = """\
//===-- gen_std.py generated file -------------------------------*- C++ -*-===//
//
// Used to build a lookup table (qualified names => include headers) for C++
// Standard Library symbols.
//
// Automatically generated file, DO NOT EDIT!
//
// Generated from cppreference offline HTML book (modified on %s).
//===----------------------------------------------------------------------===//
"""

def HasClass(tag, *classes):
  for c in tag.get('class', []):
    if c in classes:
      return True
  return False

def ParseSymbolPage(symbol_page_html, symbol_name):
  """Parse symbol page and retrieve the include header defined in this page.
  The symbol page provides header for the symbol, specifically in
  "Defined in header <header>" section. An example:

  <tr class="t-dsc-header">
    <td colspan="2"> <div>Defined in header <code>&lt;ratio&gt;</code> </div>
  </td></tr>

  Returns a list of headers.
  """
  headers = set()
  all_headers = set()

  soup = BeautifulSoup(symbol_page_html, "html.parser")
  # Rows in table are like:
  #   Defined in header <foo>      .t-dsc-header
  #   Defined in header <bar>      .t-dsc-header
  #   decl1                        .t-dcl
  #   Defined in header <baz>      .t-dsc-header
  #   decl2                        .t-dcl
  for table in soup.select('table.t-dcl-begin, table.t-dsc-begin'):
    current_headers = []
    was_decl = False
    for row in table.select('tr'):
      if HasClass(row, 't-dcl', 't-dsc'):
        was_decl = True
        # Symbols are in the first cell.
        found_symbols = row.find('td').stripped_strings
        if not symbol_name in found_symbols:
          continue
        headers.update(current_headers)
      elif HasClass(row, 't-dsc-header'):
        # If we saw a decl since the last header, this is a new block of headers
        # for a new block of decls.
        if was_decl:
          current_headers = []
        was_decl = False
        # There are also .t-dsc-header for "defined in namespace".
        if not "Defined in header " in row.text:
          continue
        # The interesting header content (e.g. <cstdlib>) is wrapped in <code>.
        for header_code in row.find_all("code"):
          current_headers.append(header_code.text)
          all_headers.add(header_code.text)
  # If the symbol was never named, consider all named headers.
  return headers or all_headers


def ParseIndexPage(index_page_html):
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
    variant = isinstance(caption, NavigableString) and "(" in caption
    symbol_tt = symbol_href.find("tt")
    if symbol_tt:
      symbols.append((symbol_tt.text.rstrip("<>()"), # strip any trailing <>()
                      symbol_href["href"], variant))
  return symbols

class Symbol:

  def __init__(self, name, namespace, headers):
    # unqualifed symbol name, e.g. "move"
    self.name = name
    # namespace of the symbol (with trailing "::"), e.g. "std::"
    self.namespace = namespace
    # a list of corresponding headers
    self.headers = headers


def ReadSymbolPage(path, name):
  with open(path) as f:
    return ParseSymbolPage(f.read(), name)


def GetSymbols(pool, root_dir, index_page_name, namespace):
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
  with open(index_page_path, "r") as f:
    # Read each symbol page in parallel.
    results = [] # (symbol_name, promise of [header...])
    for symbol_name, symbol_page_path, variant in ParseIndexPage(f.read()):
      # Variant symbols (e.g. the std::locale version of isalpha) add ambiguity.
      # FIXME: use these as a fallback rather than ignoring entirely.
      if variant:
        continue
      path = os.path.join(root_dir, symbol_page_path)
      results.append((symbol_name,
                      pool.apply_async(ReadSymbolPage, (path, symbol_name))))

    # Build map from symbol name to a set of headers.
    symbol_headers = collections.defaultdict(set)
    for symbol_name, lazy_headers in results:
      symbol_headers[symbol_name].update(lazy_headers.get())

  symbols = []
  for name, headers in sorted(symbol_headers.items(), key=lambda t : t[0]):
    symbols.append(Symbol(name, namespace, list(headers)))
  return symbols


def ParseArg():
  parser = argparse.ArgumentParser(description='Generate StdGen file')
  parser.add_argument('-cppreference', metavar='PATH',
                      default='',
                      help='path to the cppreference offline HTML directory',
                      required=True
                      )
  return parser.parse_args()


def main():
  args = ParseArg()
  cpp_root = os.path.join(args.cppreference, "en", "cpp")
  symbol_index_root = os.path.join(cpp_root, "symbol_index")
  if not os.path.exists(symbol_index_root):
    exit("Path %s doesn't exist!" % symbol_index_root)

  parse_pages =  [
    (cpp_root, "symbol_index.html", "std::"),
    # std sub-namespace symbols have separated pages.
    # We don't index std literal operators (e.g.
    # std::literals::chrono_literals::operator""d), these symbols can't be
    # accessed by std::<symbol_name>.
    # FIXME: index std::placeholders symbols, placeholders.html page is
    # different (which contains one entry for _1, _2, ..., _N), we need special
    # handling.
    (symbol_index_root, "chrono.html", "std::chrono::"),
    (symbol_index_root, "filesystem.html", "std::filesystem::"),
    (symbol_index_root, "pmr.html", "std::pmr::"),
    (symbol_index_root, "regex_constants.html", "std::regex_constants::"),
    (symbol_index_root, "this_thread.html", "std::this_thread::"),
  ]

  symbols = []
  # Run many workers to process individual symbol pages under the symbol index.
  # Don't allow workers to capture Ctrl-C.
  pool = multiprocessing.Pool(
      initializer=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
  try:
    for root_dir, page_name, namespace in parse_pages:
      symbols.extend(GetSymbols(pool, root_dir, page_name, namespace))
  finally:
    pool.terminate()
    pool.join()

  # We don't have version information from the unzipped offline HTML files.
  # so we use the modified time of the symbol_index.html as the version.
  index_page_path = os.path.join(cpp_root, "symbol_index.html")
  cppreference_modified_date = datetime.datetime.fromtimestamp(
    os.stat(index_page_path).st_mtime).strftime('%Y-%m-%d')
  print STDGEN_CODE_PREFIX % cppreference_modified_date
  for symbol in symbols:
    if len(symbol.headers) == 1:
      # SYMBOL(unqualified_name, namespace, header)
      print "SYMBOL(%s, %s, %s)" % (symbol.name, symbol.namespace,
                                    symbol.headers[0])
    elif len(symbol.headers) == 0:
      sys.stderr.write("No header found for symbol %s\n" % symbol.name)
    else:
      # FIXME: support symbols with multiple headers (e.g. std::move).
      sys.stderr.write("Ambiguous header for symbol %s: %s\n" % (
          symbol.name, ', '.join(symbol.headers)))


if __name__ == '__main__':
  main()
