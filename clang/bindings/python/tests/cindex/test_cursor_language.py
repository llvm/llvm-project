import os

from clang.cindex import Config, LanguageKind

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest

from .util import get_cursor, get_tu


class TestCursorLanguage(unittest.TestCase):
    def test_c(self):
        tu = get_tu("int a;", lang="c")
        main_func = get_cursor(tu.cursor, "a")
        self.assertEqual(main_func.language, LanguageKind.C)

    def test_c(self):
        tu = get_tu("class Cls {};", lang="cpp")
        main_func = get_cursor(tu.cursor, "Cls")
        self.assertEqual(main_func.language, LanguageKind.C_PLUS_PLUS)

    def test_obj_c(self):
        tu = get_tu("@interface If : NSObject", lang="objc")
        main_func = get_cursor(tu.cursor, "If")
        self.assertEqual(main_func.language, LanguageKind.OBJ_C)
