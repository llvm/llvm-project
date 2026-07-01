#!/usr/bin/env python3

"""Filters clang-scan-deps full output down to a chosen set of fields.

clang-scan-deps prints every computed field in its `experimental-full`
output. Pipe the full output through this script to keep only the fields a
test cares about, then hand that to FileCheck.

Usage:

  clang-scan-deps -format experimental-full ... \\
    | scan-deps-filter.py --fields=name,file-deps \\
    | FileCheck %s

In lit tests the script is invoked through the `%scan-deps-filter`
substitution rather than by path.

Each --fields entry is one of:

  * a bare key (e.g. `file-deps`), which matches that key wherever it
    appears at any depth, or
  * a dotted path (e.g. `modules.command-line`), which matches only that
    key when reached along the given path from the document root. Every
    object key on the path must be named. A translation unit's command line,
    for instance, is at `translation-units.commands.command-line`.

The result is deterministically ordered JSON, in the same key order
clang-scan-deps emitted.
"""

import argparse
import json
import sys


def parse_patterns(fields):
    """Turn each --fields entry into a list of path segments. A bare key
    becomes a wildcard-prefixed pattern that matches anywhere; a dotted entry
    becomes an anchored path from the root."""
    patterns = []
    for field in fields:
        if "." in field:
            patterns.append(field.split("."))
        else:
            # "**" marks a bare key: match its single trailing segment at any
            # depth (it is not glob-style recursive descent).
            patterns.append(["**", field])
    return patterns


def matches(keypath, patterns):
    """Does the key path to the current value (array indices omitted) match
    any pattern? A bare-key ("**") pattern matches when its trailing segment
    equals the end of the key path; a dotted pattern must equal the whole
    path from the root."""
    for pattern in patterns:
        if pattern[0] == "**":
            tail = pattern[1:]
            if keypath[-len(tail) :] == tail:
                return True
        elif keypath == pattern:
            return True
    return False


def prune(value, keypath, patterns):
    """Return a copy of `value` keeping only paths that reach a matching key,
    or None when nothing under `value` matches."""
    if matches(keypath, patterns):
        # Selected key: keep its entire subtree verbatim.
        return value
    if isinstance(value, dict):
        result = {}
        for key, sub in value.items():
            pruned = prune(sub, keypath + [key], patterns)
            if pruned is not None:
                result[key] = pruned
        return result if result else None
    if isinstance(value, list):
        # Arrays are transparent: items keep the enclosing key path.
        result = [prune(item, keypath, patterns) for item in value]
        result = [item for item in result if item is not None]
        return result if result else None
    # A scalar only survives when reached through a matching key (above).
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Full dependencies JSON file (defaults to stdin).",
    )
    parser.add_argument(
        "--fields",
        required=True,
        help="Comma-separated list of keys or dotted paths to keep.",
    )
    args = parser.parse_args()

    fields = [f for f in args.fields.split(",") if f]
    if not fields:
        parser.error("--fields requires at least one field name")
    patterns = parse_patterns(fields)

    with open(args.input) if args.input else sys.stdin as f:
        data = json.load(f)

    pruned = prune(data, [], patterns)
    if pruned is None:
        # Nothing matched; emit an empty object but warn so a mistyped field
        # name does not silently feed empty input to FileCheck.
        sys.stderr.write(
            "scan-deps-filter: warning: no fields matched: %s\n" % ",".join(fields)
        )
        pruned = {}

    json.dump(pruned, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
