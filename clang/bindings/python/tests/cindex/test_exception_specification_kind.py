import os

from clang.cindex import Config, CursorKind, ExceptionSpecificationKind

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest

from .util import get_tu


def find_function_declarations(node, declarations=[]):
    if node.kind == CursorKind.FUNCTION_DECL:
        declarations.append(node)
    for child in node.get_children():
        declarations = find_function_declarations(child, declarations)
    return declarations


class TestExceptionSpecificationKind(unittest.TestCase):
    def test_exception_specification_kind(self):
        source = """int square1(int x);
                    int square2(int x) noexcept;
                    int square3(int x) noexcept(noexcept(x * x));"""

        tu = get_tu(source, lang="cpp", flags=["-std=c++14"])

        declarations = find_function_declarations(tu.cursor)
        expected = [
            ("square1", ExceptionSpecificationKind.NONE),
            ("square2", ExceptionSpecificationKind.BASIC_NOEXCEPT),
            ("square3", ExceptionSpecificationKind.COMPUTED_NOEXCEPT),
        ]
        from_cursor = [
            (node.spelling, node.exception_specification_kind) for node in declarations
        ]
        from_type = [
            (node.spelling, node.type.get_exception_specification_kind())
            for node in declarations
        ]
        self.assertListEqual(from_cursor, expected)
        self.assertListEqual(from_type, expected)
