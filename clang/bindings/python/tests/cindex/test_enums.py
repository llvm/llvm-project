import unittest

from clang.cindex import (
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
            self.assertEqual(enum.from_id(2), enum._kinds[2])
            with self.assertRaises(ValueError):
                enum.from_id(len(enum._kinds))
            with self.assertRaises(ValueError):
                enum.from_id(-1)

    def test_unique_kinds(self):
        """Check that no kind name has been used multiple times"""
        for enum in self.enums:
            for id in range(len(enum._kinds)):
                try:
                    enum.from_id(id).name
                except ValueError:
                    pass
