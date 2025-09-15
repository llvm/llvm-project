import unittest
from pathlib import Path

from clang.cindex import (
    AccessSpecifier,
    AvailabilityKind,
    BinaryOperator,
    CursorKind,
    ExceptionSpecificationKind,
    LanguageKind,
    LinkageKind,
    RefQualifierKind,
    StorageClass,
    TemplateArgumentKind,
    TLSKind,
    TokenKind,
    TranslationUnit,
    TypeKind,
    PrintingPolicyProperty,
    BaseEnumeration,
)


class TestEnums(unittest.TestCase):
    enums = BaseEnumeration.__subclasses__()

    def test_from_id(self):
        """Check that kinds can be constructed from valid IDs"""
        for enum in self.enums:
            self.assertEqual(enum.from_id(2), enum(2))
            max_value = max([variant.value for variant in enum])
            with self.assertRaises(ValueError):
                enum.from_id(max_value + 1)
            with self.assertRaises(ValueError):
                enum.from_id(-1)

    def test_duplicate_ids(self):
        """Check that no two kinds have the same id"""
        for enum in self.enums:
            num_declared_variants = len(enum._member_map_.keys())
            num_unique_variants = len(list(enum))
            self.assertEqual(num_declared_variants, num_unique_variants)

    def test_all_variants(self):
        """Check that all libclang enum values are also defined in cindex.py"""
        cenum_to_pythonenum = {
            "CX_CXXAccessSpecifier": AccessSpecifier,
            "CX_StorageClass": StorageClass,
            "CXAvailabilityKind": AvailabilityKind,
            "CXBinaryOperatorKind": BinaryOperator,
            "CXCursorKind": CursorKind,
            "CXCursor_ExceptionSpecificationKind": ExceptionSpecificationKind,
            "CXLanguageKind": LanguageKind,
            "CXLinkageKind": LinkageKind,
            "CXPrintingPolicyProperty": PrintingPolicyProperty,
            "CXRefQualifierKind": RefQualifierKind,
            "CXTemplateArgumentKind": TemplateArgumentKind,
            "CXTLSKind": TLSKind,
            "CXTokenKind": TokenKind,
            "CXTypeKind": TypeKind,
        }

        indexheader = (
            Path(__file__).parent.parent.parent.parent.parent
            / "include/clang-c/Index.h"
        )
        tu = TranslationUnit.from_source(indexheader, ["-x", "c++"])

        enum_variant_map = {}
        # For all enums in self.enums, extract all enum variants defined in Index.h
        for cursor in tu.cursor.walk_preorder():
            type_class = cenum_to_pythonenum.get(cursor.type.spelling)
            if (
                cursor.kind == CursorKind.ENUM_CONSTANT_DECL
                and type_class in self.enums
            ):
                if type_class not in enum_variant_map:
                    enum_variant_map[type_class] = []
                enum_variant_map[type_class].append(cursor.enum_value)

        for enum in self.enums:
            with self.subTest(enum):
                python_kinds = set([kind.value for kind in enum])
                # Defined in Index.h but not in cindex.py
                missing_python_kinds = c_kinds - python_kinds
                self.assertEqual(
                    missing_python_kinds,
                    set(),
                    f"Please ensure these are defined in {enum} in cindex.py.",
                )
                # Defined in cindex.py but not in Index.h
                superfluous_python_kinds = python_kinds - c_kinds
                self.assertEqual(
                    superfluous_python_kinds,
                    set(),
                    f"Please ensure that all {enum} kinds defined in cindex.py have an equivalent in Index.h",
                )
