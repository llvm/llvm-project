import unittest

from clang.cindex import SourceLocation, SourceRange

from .util import get_tu


def create_location(tu, line, column):
    return SourceLocation.from_position(tu, tu.get_file(tu.spelling), line, column)


def create_range(tu, line1, column1, line2, column2):
    return SourceRange.from_locations(
        create_location(tu, line1, column1), create_location(tu, line2, column2)
    )


class TestSourceRange(unittest.TestCase):
    def test_contains(self):
        tu = get_tu(
            """aaaaa
aaaaa
aaaaa
aaaaa"""
        )

        l13 = create_location(tu, 1, 3)
        l21 = create_location(tu, 2, 1)
        l22 = create_location(tu, 2, 2)
        l23 = create_location(tu, 2, 3)
        l24 = create_location(tu, 2, 4)
        l25 = create_location(tu, 2, 5)
        l33 = create_location(tu, 3, 3)
        l31 = create_location(tu, 3, 1)
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
