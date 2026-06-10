"""Generate the "Python API enumerators and constants" documentation page.

LLDB exposes the enumerators from `lldb-enumerations.h` and the constants from
`lldb-defines.h` as attributes of the `lldb` Python module. This script parses
those two headers and emits a Markdown page documenting every public value, so
the page can no longer drift out of sync with the source the way a
hand-maintained copy does.

The page is generated at build time and pulled into `python_api_enums.md` via
the `{build-include}` directive (see `lldb/docs/_ext/build_include.py`).
"""

import argparse
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

# Matches the start of an enum declaration up to and including the opening
# brace, capturing the enum name. Covers plain `enum Name {`, scoped
# `enum Name : type {`, and the `FLAGS_ENUM(Name){` / `FLAGS_ANONYMOUS_ENUM()`
# macros from lldb-enumerations.h. Enum bodies never contain nested braces, so
# the matching `}` is simply the next one in the text.
ENUM_RE = re.compile(
    r"(?:enum\s+(?P<name>\w+)\s*(?::\s*[\w:]+\s*)?"
    r"|FLAGS_ENUM\(\s*(?P<flags_name>\w+)\s*\)"
    r"|FLAGS_ANONYMOUS_ENUM\(\s*\))\s*\{"
)

# Doxygen inline commands that wrap a following word for emphasis or reference.
# We drop the command itself and keep its argument.
DOXYGEN_CMD_RE = re.compile(r"\\(?:a|b|c|e|p|ref|see|link|endlink)\b\s?")

# Constants are grouped editorially to match the long-standing layout of the
# page. The classifier is prefix-based so new constants land in a sensible
# group without further maintenance; anything unrecognized falls into
# "Miscellaneous constants".
CONSTANT_GROUP_ORDER = [
    "Generic register numbers",
    "Invalid value definitions",
    "CPU types",
    "Option set definitions",
    "Miscellaneous constants",
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def clean_comment(text: str) -> str:
    """Strip a doc-comment fragment down to its prose."""
    return DOXYGEN_CMD_RE.sub("", text).rstrip()


@dataclass
class Member:
    name: str
    desc: list[str] = field(default_factory=list)  # lines; "" marks a paragraph break


def parse_enum_body(body: str) -> list[Member]:
    """Parse the body of an enum into a list of documented members.

    Comment association follows Doxygen conventions, with one accommodation for
    the header's occasional misuse of `///<` on its own line as a *leading*
    comment (see WatchpointValueKind): a trailing `///<` documents the member on
    its own line, while a standalone doc comment that isn't continuing a
    trailing comment is treated as a leading comment for the next member.
    """
    members = []
    pending_lead = []  # leading doc lines awaiting the next member
    current = None  # most recently named member (target of trailing comments)
    in_trailing = False  # currently extending a member's trailing comment
    awaiting_name = True  # next identifier starts a new member
    depth = 0  # parenthesis nesting, to find top-level commas

    def attach_lead(member: Member) -> None:
        # Drop a leading line that merely repeats the member name (the style
        # used by CommandFlags) along with its trailing blank.
        lead = pending_lead[:]
        while lead and lead[0] == "":
            lead.pop(0)
        if lead and lead[0] == member.name:
            lead.pop(0)
            while lead and lead[0] == "":
                lead.pop(0)
        member.desc.extend(lead)

    for line in body.splitlines():
        comment_start = line.find("//")
        if comment_start == -1:
            code, comment = line, None
        else:
            code, comment = line[:comment_start], line[comment_start:]

        # Walk the code, picking out member names and top-level commas.
        i = 0
        while i < len(code):
            ch = code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                assert depth >= 0
            elif ch == "," and depth == 0:
                awaiting_name = True
            elif awaiting_name and (ch.isalpha() or ch == "_"):
                j = i
                while j < len(code) and (code[j].isalnum() or code[j] == "_"):
                    j += 1
                name = code[i:j]
                current = Member(name)
                attach_lead(current)
                pending_lead = []
                in_trailing = False
                awaiting_name = False
                # Only public enumerators (the `e` prefix) are documented;
                # `k`-prefixed sentinels like kNumFormats are internal.
                if name.startswith("e"):
                    members.append(current)
                i = j
                continue
            i += 1

        if comment is not None:
            has_code = bool(code.strip())
            if comment.startswith("///<"):
                text = clean_comment(comment.removeprefix("///<").lstrip())
                if has_code and current is not None:
                    current.desc.append(text)
                    in_trailing = True
                elif in_trailing and current is not None:
                    current.desc.append(text)
                else:
                    pending_lead.append(text)
            elif comment.startswith("///"):
                text = clean_comment(comment.removeprefix("///").lstrip())
                if has_code and current is not None:
                    current.desc.append(text)
                    in_trailing = True
                elif in_trailing and current is not None:
                    current.desc.append(text)
                else:
                    pending_lead.append(text)
            # A plain `//` comment is an internal note; ignore it.
        elif not code.strip():
            # Blank line: ends any trailing-comment continuation and separates
            # paragraphs in an accumulating leading comment.
            in_trailing = False
            if pending_lead and pending_lead[-1] != "":
                pending_lead.append("")

    return members


def parse_enums(text: str) -> Iterator[tuple[str, list[str], list[Member]]]:
    """Yield (name, description_lines, members) for each enum in the header."""
    for match in ENUM_RE.finditer(text):
        name = match.group("name") or match.group("flags_name")
        if name is None:
            continue  # anonymous flag enums have no name to document
        close = text.index("}", match.end())
        members = parse_enum_body(text[match.end() : close])
        if not members:
            continue
        yield name, leading_description(text[: match.start()]), members


def leading_description(preceding_text: str) -> list[str]:
    """Collect the `///` doc comment immediately above a declaration."""
    lines = []
    for line in reversed(preceding_text.splitlines()):
        if not line.strip().startswith("//"):
            break
        lines.append(line)
    lines.reverse()

    desc = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("///"):
            desc.append(clean_comment(stripped.removeprefix("///").lstrip()))
    while desc and desc[0] == "":
        desc.pop(0)
    while desc and desc[-1] == "":
        desc.pop()
    return desc


def classify_constant(name: str) -> str:
    if name.startswith("LLDB_REGNUM_GENERIC_"):
        return "Generic register numbers"
    if name == "LLDB_INVALID_CPUTYPE" or name.startswith("LLDB_ARCH_"):
        return "CPU types"
    if name.startswith("LLDB_INVALID_"):
        return "Invalid value definitions"
    if name == "LLDB_MAX_NUM_OPTION_SETS" or name.startswith("LLDB_OPT_SET_"):
        return "Option set definitions"
    return "Miscellaneous constants"


def parse_constants(text: str) -> dict[str, list[Member]]:
    """Parse value `#define LLDB_*` constants grouped for presentation."""
    # Join backslash line continuations so a define and its trailing comment
    # form a single logical line.
    logical = re.sub(r"\\\n", " ", text)

    groups = {name: [] for name in CONSTANT_GROUP_ORDER}
    # A `(` immediately after the name (no space) marks a function-like macro;
    # a `(` after whitespace is just a parenthesized value like `(1u << 0)`.
    define_re = re.compile(r"^#define\s+(LLDB_\w+)(\()?(.*)$")
    for line in logical.splitlines():
        match = define_re.match(line.strip())
        if not match:
            continue
        name, is_macro, rest = match.groups()
        if is_macro:
            continue  # function-like macro, not a Python-visible constant
        if not rest.strip():
            continue  # value-less define such as the include guard
        desc = ""
        comment_start = rest.find("//")
        if comment_start != -1:
            desc = clean_comment(rest[comment_start:].lstrip("/").lstrip())
        groups[classify_constant(name)].append(Member(name, [desc] if desc else []))
    return groups


def format_directive(out: list[str], member: Member) -> None:
    out.append("```{eval-rst}")
    out.append(f".. py:data:: {member.name}")
    desc = member.desc[:]
    while desc and desc[0] == "":
        desc.pop(0)
    while desc and desc[-1] == "":
        desc.pop()
    if desc:
        out.append("")
        for line in desc:
            out.append(f"   {line}" if line else "")
    out.append("```")
    out.append("")


def format_paragraphs(out: list[str], lines: list[str]) -> None:
    for line in lines:
        out.append(line)
    if lines:
        out.append("")


def generate(enums_text: str, defines_text: str) -> str:
    out = []
    out.append("# Python API enumerators and constants")
    out.append("")
    out.append("```{eval-rst}")
    out.append(".. py:currentmodule:: lldb")
    out.append("```")
    out.append("")

    out.append("## Constants")
    out.append("")
    groups = parse_constants(defines_text)
    for group in CONSTANT_GROUP_ORDER:
        members = groups[group]
        if not members:
            continue
        out.append(f"({slugify(group)})=")
        out.append("")
        out.append(f"### {group}")
        out.append("")
        for member in members:
            format_directive(out, member)

    out.append("## Enumerators")
    out.append("")
    for name, desc, members in parse_enums(enums_text):
        out.append(f"({name.lower()})=")
        out.append("")
        out.append(f"### {name}")
        out.append("")
        format_paragraphs(out, desc)
        for member in members:
            format_directive(out, member)

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gen-python-api-enums",
        description="Generate the Python API enums/constants doc from headers",
    )
    parser.add_argument(
        "--enumerations", required=True, help="Path to lldb-enumerations.h"
    )
    parser.add_argument("--defines", required=True, help="Path to lldb-defines.h")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    with open(args.enumerations, encoding="utf-8") as f:
        enums_text = f.read()
    with open(args.defines, encoding="utf-8") as f:
        defines_text = f.read()

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(generate(enums_text, defines_text))


if __name__ == "__main__":
    main()
