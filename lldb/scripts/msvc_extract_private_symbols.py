"""A tool for extracting a list of private lldb symbols to export for MSVC.

When exporting symbols from a dll or exe we either need to mark the symbols in
the source code as __declspec(dllexport) or supply a list of symbols to the
linker. Private symbols in LLDB don't explicitly specific dllexport, so we
automate that by examining the symbol table.
"""

import argparse
import os
import re
import subprocess
import sys


def extract_symbols(
    nm_path: str, lib: str, regex: re.Pattern[str], nm_flags: list[str]
):
    """Extract all of the private lldb symbols from the given path to llvm-nm and
    library to extract from."""

    # '-p' do not waste time sorting the symbols.
    process = subprocess.Popen(
        [nm_path, "-p", lib] + nm_flags,
        bufsize=1,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        universal_newlines=True,
    )
    process.stdin.close()

    lldb_symbols = set()
    for line in process.stdout:
        match = re.match(regex, line)
        if match:
            symbol = match.group("symbol")
            assert (
                symbol.count(" ") == 0
            ), "Regex matched too much, probably got undecorated name as well"
            # Deleting destructors start with ?_G or ?_E and can be discarded
            # because link.exe gives you a warning telling you they can't be
            # exported if you don't.
            if symbol.startswith("??_G") or symbol.startswith("??_E"):
                continue
            lldb_symbols.add(symbol)

    return lldb_symbols


def extract_exports(nm_path: str, lib: str):
    # Matches mangled symbols containing 'lldb_private'.
    regex = re.compile(r"[0-9a-zA-Z]* [BT] (?P<symbol>[?]+[^?].*lldb_private.*)")
    # '-g': Only get the global symbols
    return extract_symbols(nm_path, lib, regex, ["-g"])


def extract_undef(nm_path: str, lib: str):
    # Matches mangled symbols containing 'lldb_private' or 'llvm'.
    regex_undef = re.compile(r"^0* +U (?P<symbol>[?]+[^?].*(?:lldb_private|llvm).*)")
    regex_global = re.compile(
        r"[0-9a-zA-Z]* [BT] (?P<symbol>[?]+[^?].*(?:lldb_private|llvm).*)"
    )
    # '-u': Only get the undefined symbols
    undef = extract_symbols(nm_path, lib, regex_undef, ["-u"])
    # '-g': Only get the global symbols
    globals = extract_symbols(nm_path, lib, regex_global, ["-g"])
    # If this is a static library, only return symbols undefined in all object
    # files.
    return undef - globals


def main():
    parser = argparse.ArgumentParser(description="Generate LLDB dll exports")
    parser.add_argument(
        "-o", metavar="file", type=str, help="The name of the resultant export file."
    )
    parser.add_argument("--nm", help="Path to the llvm-nm executable.")
    parser.add_argument(
        "--libs",
        metavar="lib",
        type=str,
        nargs="+",
        help="Libraries to extract exported symbols from.",
    )
    parser.add_argument(
        "--consuming-libs",
        metavar="lib",
        type=str,
        nargs="*",
        help="Libraries to extract undefined symbols from.",
    )
    args = parser.parse_args()

    # Get the list of libraries to extract symbols from
    libs = list()
    for input_libs, is_consuming in ((args.libs, False), (args.consuming_libs, True)):
        for lib in input_libs:
            # When invoked by cmake the arguments are the cmake target names of the
            # libraries, so we need to add .lib/.a to the end and maybe lib to the
            # start to get the filename. Also allow objects.
            suffixes = [".lib", ".a", ".obj", ".o"]
            if not any([lib.endswith(s) for s in suffixes]):
                for suffix in suffixes:
                    if os.path.exists(lib + suffix):
                        lib = lib + suffix
                        break
                    if os.path.exists("lib" + lib + suffix):
                        lib = "lib" + lib + suffix
                        break
            if not any([lib.endswith(s) for s in suffixes]):
                print(
                    "Unknown extension type for library argument: " + lib,
                    file=sys.stderr,
                )
                exit(1)
            libs.append((lib, is_consuming))

    # Extract symbols from the input libraries.
    symbols = set()
    for lib, is_consuming in libs:
        symbols |= (
            extract_undef(args.nm, lib)
            if is_consuming
            else extract_exports(args.nm, lib)
        )

    # Write out the symbols to the output file.
    with open(args.o, "w", newline="") as f:
        for s in sorted(symbols):
            f.write(f"{s}\n")


if __name__ == "__main__":
    main()
