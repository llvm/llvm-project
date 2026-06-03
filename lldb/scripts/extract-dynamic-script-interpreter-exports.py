"""Emit the LLDB-side imports of a script interpreter plugin's objects.

The result is fed into liblldb's exports file so the dynamic loader can
satisfy the plugin's references at runtime without growing liblldb's
re-export surface beyond what the plugin actually consumes.
"""

import argparse
import subprocess
import sys


# Mangled-name prefixes liblldb owns. Symbols outside these namespaces
# are resolved through other libraries on the loader path.
EXPORT_PREFIXES = (
    "_ZN12lldb_private",
    "_ZNK12lldb_private",
    "_ZTVN12lldb_private",
    "_ZTSN12lldb_private",
    "_ZTIN12lldb_private",
    "_ZN4lldb",
    "_ZNK4lldb",
    "_ZTVN4lldb",
    "_ZTSN4lldb",
    "_ZTIN4lldb",
    "_ZN4llvm",
    "_ZNK4llvm",
    "_ZTVN4llvm",
    "_ZTSN4llvm",
    "_ZTIN4llvm",
)


def normalize(sym, mach_o):
    # Mach-O reports an extra leading underscore that the .exports file
    # format does not carry.
    if mach_o and sym.startswith("_"):
        return sym[1:]
    return sym


def collect(nm, obj, mach_o):
    """Return (undefined, defined) symbol-name sets for `obj`.

    Both halves are needed so cross-object references that the plugin's
    own objects satisfy can be subtracted before emitting.
    """
    undefined = set()
    defined = set()

    out = subprocess.check_output(
        [nm, "--undefined-only", "--just-symbol-name", obj], text=True
    )
    for line in out.splitlines():
        sym = line.strip()
        if sym:
            undefined.add(normalize(sym, mach_o))

    out = subprocess.check_output(
        [nm, "--defined-only", "--extern-only", "--just-symbol-name", obj],
        text=True,
    )
    for line in out.splitlines():
        sym = line.strip()
        if sym:
            defined.add(normalize(sym, mach_o))

    return undefined, defined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm", required=True, help="Path to llvm-nm.")
    parser.add_argument("-o", "--output", required=True, help="Output exports file.")
    parser.add_argument(
        "--mach-o",
        action="store_true",
        help="Strip the leading underscore Mach-O prepends " "to every symbol.",
    )
    parser.add_argument(
        "objects", nargs="+", help="Object files (or archives) to scan."
    )
    args = parser.parse_args()

    undefined = set()
    defined = set()
    for obj in args.objects:
        u, d = collect(args.nm, obj, args.mach_o)
        undefined.update(u)
        defined.update(d)

    syms = {s for s in (undefined - defined) if s.startswith(EXPORT_PREFIXES)}

    with open(args.output, "w", newline="\n") as f:
        for s in sorted(syms):
            f.write(s + "\n")


if __name__ == "__main__":
    sys.exit(main())
