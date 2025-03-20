import os

from clang.cindex import Config, File, Index, TranslationUnit

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest

kInputsDir = os.path.join(os.path.dirname(__file__), "INPUTS")

class TestFile(unittest.TestCase):
    def test_file(self):
        index = Index.create()
        tu = index.parse("t.c", unsaved_files=[("t.c", "")])
        file = File.from_name(tu, "t.c")
        self.assertEqual(str(file), "t.c")
        self.assertEqual(file.name, "t.c")
        self.assertEqual(repr(file), "<File: t.c>")

    def test_file_eq(self):
        path = os.path.join(kInputsDir, "hello.cpp")
        header_path = os.path.join(kInputsDir, "header3.h")
        tu = TranslationUnit.from_source(path)
        file1 = File.from_name(tu, path)
        file2 = File.from_name(tu, header_path)
        file2_2 = File.from_name(tu, header_path)

        self.assertEqual(file1, file1)
        self.assertEqual(file2, file2_2)
        self.assertNotEqual(file1, file2)
        self.assertNotEqual(file1, "t.c")
