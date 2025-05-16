import os

import clang.cindex

if "CLANG_LIBRARY_PATH" in os.environ:
    clang.cindex.Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest
import ast


class TestLib(unittest.TestCase):
    def test_functions_registered(self):
        def get_function_spelling(node):
            # The call expressions we are interested in have
            # their spelling in .attr, not .id
            if hasattr(node, "attr"):
                return node.attr
            return ""

        filename = clang.cindex.__file__
        with open(filename) as file:
            root = ast.parse(file.read())
        functions = [
            get_function_spelling(node.func)
            for node in ast.walk(root)
            if isinstance(node, ast.Call)
        ]
        used_functions = set([func for func in functions if func.startswith("clang_")])
        registered_functions = set([item[0] for item in clang.cindex.FUNCTION_LIST])
        self.assertEqual(used_functions - registered_functions, set())
