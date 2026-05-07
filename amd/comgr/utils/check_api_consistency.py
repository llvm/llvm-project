#!/usr/bin/env python3
"""Check Comgr API version consistency across the header and exportmap.

Run with no arguments from the comgr source root, or pass --comgr-dir.

The checks:

1. ``VERSION.txt`` (MAJOR.MINOR) is at least the highest
   ``AMD_COMGR_VERSION_X_Y`` macro tag declared in ``include/amd_comgr.h.in``.
   Catches "added/tagged a new API but forgot to bump VERSION.txt."

2. The set of functions declared with the ``AMD_COMGR_API`` qualifier in
   ``include/amd_comgr.h.in`` matches the set of ``amd_comgr_*`` symbols in
   ``src/exportmap.in`` (the Linux symbol-version script). Catches drift
   between the public header and the export list -- either an API declared
   but not exported, or a symbol exported without a public declaration.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Version:
    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor) < (other.major, other.minor)


@dataclass
class HeaderInfo:
    """Parsed contents of ``amd_comgr.h.in``."""

    version_tags: set[Version] = field(default_factory=set)
    declared_apis: set[str] = field(default_factory=set)


def parse_version_txt(path: Path) -> Version:
    """Parse ``#COMGR_VERSION_MAJOR\\nN\\n#COMGR_VERSION_MINOR\\nM`` format."""
    text = path.read_text()
    major_match = re.search(r"#COMGR_VERSION_MAJOR\s*\n\s*(\d+)", text)
    minor_match = re.search(r"#COMGR_VERSION_MINOR\s*\n\s*(\d+)", text)
    if not major_match or not minor_match:
        raise ValueError(
            f"{path}: could not parse #COMGR_VERSION_MAJOR / "
            f"#COMGR_VERSION_MINOR"
        )
    return Version(int(major_match.group(1)), int(minor_match.group(1)))


def parse_header(path: Path) -> HeaderInfo:
    """Extract version tags and AMD_COMGR_API function names."""
    text = path.read_text()
    info = HeaderInfo()

    # AMD_COMGR_VERSION_X_Y macro definitions (the version-tag list).
    for m in re.finditer(r"#define\s+AMD_COMGR_VERSION_(\d+)_(\d+)\b", text):
        info.version_tags.add(Version(int(m.group(1)), int(m.group(2))))

    # Function declarations qualified with AMD_COMGR_API. The declarations
    # span multiple lines and look like:
    #
    #   amd_comgr_status_t AMD_COMGR_API
    #   amd_comgr_foo(
    #       ...args...) AMD_COMGR_VERSION_X_Y;
    #
    # or the single-line variant:
    #
    #   void AMD_COMGR_API amd_comgr_foo(...) AMD_COMGR_VERSION_X_Y;
    pattern = re.compile(
        r"\bAMD_COMGR_API\b\s+(?:[A-Za-z_][\w*\s]*\s+)?(amd_comgr_\w+)\s*\("
    )
    info.declared_apis = set(pattern.findall(text))

    return info


def parse_exportmap(path: Path) -> set[str]:
    """Extract ``amd_comgr_*`` symbol names from a linker version script.

    Strips the CMake ``@amd_comgr_NAME@`` configure_file placeholders that
    name each version block, so they aren't mistaken for symbols.
    """
    text = path.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    # Drop the @...@ CMake substitution tokens used in version-block names.
    text = re.sub(r"@[^@]+@", "", text)
    return set(re.findall(r"\bamd_comgr_[A-Za-z0-9_]+\b", text))


def check_version(version: Version, header: HeaderInfo) -> list[str]:
    if not header.version_tags:
        return ["amd_comgr.h.in: no AMD_COMGR_VERSION_X_Y macros found"]
    highest = max(header.version_tags)
    if version < highest:
        return [
            f"VERSION.txt is {version} but amd_comgr.h.in tags an API at "
            f"{highest}; bump VERSION.txt to at least {highest}"
        ]
    return []


def check_header_vs_exportmap(
    header: HeaderInfo, exported: set[str]
) -> list[str]:
    errors: list[str] = []
    declared = header.declared_apis
    declared_only = sorted(declared - exported)
    exported_only = sorted(exported - declared)
    if declared_only:
        errors.append(
            "AMD_COMGR_API functions declared in amd_comgr.h.in but missing "
            "from src/exportmap.in (will not be exported by the shared "
            "library):\n  " + "\n  ".join(declared_only)
        )
    if exported_only:
        errors.append(
            "Symbols exported in src/exportmap.in but not declared with "
            "AMD_COMGR_API in amd_comgr.h.in (orphaned export entries):\n  "
            + "\n  ".join(exported_only)
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comgr-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to amd/comgr (defaults to the script's parent directory)",
    )
    args = parser.parse_args()

    comgr_dir: Path = args.comgr_dir
    version_path = comgr_dir / "VERSION.txt"
    header_path = comgr_dir / "include" / "amd_comgr.h.in"
    exportmap_path = comgr_dir / "src" / "exportmap.in"

    for p in (version_path, header_path, exportmap_path):
        if not p.is_file():
            raise FileNotFoundError(f"expected file not found: {p}")

    version = parse_version_txt(version_path)
    header = parse_header(header_path)
    exported = parse_exportmap(exportmap_path)

    errors: list[str] = []
    errors.extend(check_version(version, header))
    errors.extend(check_header_vs_exportmap(header, exported))

    if errors:
        print("Comgr API consistency check FAILED:\n", file=sys.stderr)
        for err in errors:
            print(f"* {err}\n", file=sys.stderr)
        return 1

    print(
        f"Comgr API consistency check OK "
        f"(VERSION.txt={version}, "
        f"{len(header.declared_apis)} APIs declared and exported)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
