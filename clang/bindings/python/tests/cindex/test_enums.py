import unittest

from clang.cindex import (
    TokenKind,
    CursorKind,
    TemplateArgumentKind,
    ExceptionSpecificationKind,
    AvailabilityKind,
    AccessSpecifier,
    TypeKind,
    RefQualifierKind,
    LinkageKind,
    TLSKind,
    StorageClass,
)


class TestCursorKind(unittest.TestCase):
    enums = [
        TokenKind,
        CursorKind,
        TemplateArgumentKind,
        ExceptionSpecificationKind,
        AvailabilityKind,
        AccessSpecifier,
        TypeKind,
        RefQualifierKind,
        LinkageKind,
        TLSKind,
        StorageClass,
    ]

    def test_from_id(self):
        """Check that kinds can be constructed from valid IDs"""
        for enum in self.enums:
            self.assertEqual(enum.from_id(2), enum(2))
            max_value = max([variant.value for variant in enum])
            with self.assertRaises(ValueError):
                enum.from_id(max_value + 1)
            with self.assertRaises(ValueError):
                enum.from_id(-1)
