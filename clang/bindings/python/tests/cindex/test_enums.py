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
        # FIXME: Index.h is a C file, but we read it as a C++ file because we
        # don't get ENUM_CONSTANT_DECL cursors otherwise, which we need here
        # See bug report: https://github.com/llvm/llvm-project/issues/159075
        tu = TranslationUnit.from_source(indexheader, ["-x", "c++"])

        enum_variant_map = {}
        # For all enums in self.enums, extract all enum variants defined in Index.h
        for cursor in tu.cursor.walk_preorder():
            if cursor.kind == CursorKind.ENUM_CONSTANT_DECL:
                python_enum = cenum_to_pythonenum.get(cursor.type.spelling)
                if python_enum not in enum_variant_map:
                    enum_variant_map[python_enum] = dict()
                enum_variant_map[python_enum][cursor.enum_value] = cursor.spelling

        for enum in self.enums:
            with self.subTest(enum):
                # This ensures only the custom assert message below is printed
                self.longMessage = False

                python_kinds = set([kind.value for kind in enum])
                num_to_c_kind = enum_variant_map[enum]
                c_kinds = set(num_to_c_kind.keys())
                # Defined in Index.h but not in cindex.py
                missing_python_kinds = c_kinds - python_kinds
                missing_names = set(
                    [num_to_c_kind[kind] for kind in missing_python_kinds]
                )
                self.assertEqual(
                    missing_names,
                    set(),
                    f"{missing_names} variants are missing. "
                    f"Please ensure these are defined in {enum} in cindex.py.",
                )
                # Defined in cindex.py but not in Index.h
                superfluous_python_kinds = python_kinds - c_kinds
                missing_names = set(
                    [enum.from_id(kind) for kind in superfluous_python_kinds]
                )
                self.assertEqual(
                    missing_names,
                    set(),
                    f"{missing_names} variants only exist in the Python bindings. "
                    f"Please ensure that all {enum} kinds defined in cindex.py have an equivalent in Index.h",
                )
