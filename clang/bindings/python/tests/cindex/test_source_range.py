import os
from clang.cindex import Config

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest
from clang.cindex import SourceLocation, SourceRange, TranslationUnit

from .util import get_tu


def create_range(tu, line1, column1, line2, column2):
    return SourceRange.from_locations(
        SourceLocation.from_position(tu, tu.get_file(tu.spelling), line1, column1),
        SourceLocation.from_position(tu, tu.get_file(tu.spelling), line2, column2),
    )


class TestSourceRange(unittest.TestCase):
    def test_contains(self):
        tu = get_tu(
            """aaaaa
aaaaa
aaaaa
aaaaa"""
        )
        file = tu.get_file(tu.spelling)

        l13 = SourceLocation.from_position(tu, file, 1, 3)
        l21 = SourceLocation.from_position(tu, file, 2, 1)
        l22 = SourceLocation.from_position(tu, file, 2, 2)
        l23 = SourceLocation.from_position(tu, file, 2, 3)
        l24 = SourceLocation.from_position(tu, file, 2, 4)
        l25 = SourceLocation.from_position(tu, file, 2, 5)
        l33 = SourceLocation.from_position(tu, file, 3, 3)
        l31 = SourceLocation.from_position(tu, file, 3, 1)
        r22_24 = create_range(tu, 2, 2, 2, 4)
        r23_23 = create_range(tu, 2, 3, 2, 3)
        r24_32 = create_range(tu, 2, 4, 3, 2)
        r14_32 = create_range(tu, 1, 4, 3, 2)

        assert l13 not in r22_24  # Line before start
        assert l21 not in r22_24  # Column before start
        assert l22 in r22_24  # Colum on start
        assert l23 in r22_24  # Column in range
        assert l24 in r22_24  # Column on end
        assert l25 not in r22_24  # Column after end
        assert l33 not in r22_24  # Line after end

        assert l23 in r23_23  # In one-column range

        assert l23 not in r24_32  # Outside range in first line
        assert l33 not in r24_32  # Outside range in last line
        assert l25 in r24_32  # In range in first line
        assert l31 in r24_32  # In range in last line

        assert l21 in r14_32  # In range at start of center line
        assert l25 in r14_32  # In range at end of center line

        # In range within included file
        tu2 = TranslationUnit.from_source(
            "main.c",
            unsaved_files=[
                (
                    "main.c",
                    """int a[] = {
#include "numbers.inc"
};
""",
                ),
                (
                    "./numbers.inc",
                    """1,
2,
3,
4
                 """,
                ),
            ],
        )

        r_curly = create_range(tu2, 1, 11, 3, 1)
        l_f2 = SourceLocation.from_position(tu2, tu2.get_file("./numbers.inc"), 4, 1)
        assert l_f2 in r_curly
