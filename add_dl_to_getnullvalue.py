#!/usr/bin/env python3
"""Add DataLayout argument to single-arg Constant::getNullValue calls in a file."""
import sys


def process_file(filepath, dl_expr):
    with open(filepath, "r") as f:
        content = f.read()

    result = []
    i = 0
    pattern = "Constant::getNullValue("
    changes = 0

    while i < len(content):
        pos = content.find(pattern, i)
        if pos == -1:
            result.append(content[i:])
            break

        result.append(content[i:pos])

        line_start = content.rfind("\n", 0, pos) + 1
        line_prefix = content[line_start:pos]
        if "//" in line_prefix:
            result.append(pattern)
            i = pos + len(pattern)
            continue

        paren_start = pos + len(pattern)
        depth = 1
        j = paren_start

        while j < len(content) and depth > 0:
            ch = content[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1

        if depth != 0:
            result.append(pattern)
            i = paren_start
            continue

        arg = content[paren_start : j - 1]

        d = 0
        has_top_level_comma = False
        for ch in arg:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            elif ch == "," and d == 0:
                has_top_level_comma = True
                break

        if has_top_level_comma:
            result.append(content[pos:j])
        else:
            result.append(pattern + arg + ", " + dl_expr + ")")
            changes += 1

        i = j

    new_content = "".join(result)
    if changes > 0:
        with open(filepath, "w") as f:
            f.write(new_content)

    return changes


if __name__ == "__main__":
    filepath = sys.argv[1]
    dl_expr = sys.argv[2]
    n = process_file(filepath, dl_expr)
    print(f"{filepath}: {n} changes")
