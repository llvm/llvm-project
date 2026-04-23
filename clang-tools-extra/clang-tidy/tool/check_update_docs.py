#!/usr/bin/env python3
#
# ===-----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

"""

Clang-Tidy Check List Checker
=============================

This wrapper script runs `add_new_check.py --update-docs` on
a temporary copy of clang-tools-extra/{clang-tidy,docs} and
writes the generated docs/clang-tidy/checks/list.rst to the
requested output path.
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Final, Sequence


EXTRA_DIR: Final = os.path.join(os.path.dirname(__file__), "../..")
CLANG_TIDY_DIR: Final = os.path.join(EXTRA_DIR, "clang-tidy")
DOCS_DIR: Final = os.path.join(EXTRA_DIR, "docs")
LIST_DOC: Final = os.path.join(DOCS_DIR, "clang-tidy", "checks", "list.rst")


def read_text(path: str) -> str:
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def generate_updated_list() -> str:
    with tempfile.TemporaryDirectory() as td:
        temp_root = os.path.join(td, "clang-tools-extra")
        temp_clang_tidy_dir = os.path.join(temp_root, "clang-tidy")
        temp_docs_dir = os.path.join(temp_root, "docs")

        shutil.copytree(CLANG_TIDY_DIR, temp_clang_tidy_dir)
        shutil.copytree(DOCS_DIR, temp_docs_dir)

        subprocess.run(
            [
                sys.executable,
                os.path.join(temp_clang_tidy_dir, "add_new_check.py"),
                "--update-docs",
            ],
            cwd=temp_clang_tidy_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return read_text(
            os.path.join(temp_docs_dir, "clang-tidy", "checks", "list.rst")
        )


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", dest="out", required=True)
    args = ap.parse_args(argv)

    generated = generate_updated_list()
    write_text(args.out, generated)

    if read_text(LIST_DOC) != generated:
        sys.stderr.write(
            "\n'clang-tools-extra/docs/clang-tidy/checks/list.rst' is out of date.\n"
            "Fix it by running 'clang-tools-extra/clang-tidy/add_new_check.py --update-docs'.\n\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
