import os

from clang.cindex import Config, File, Index, TranslationUnit

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest

inputs_dir = os.path.join(os.path.dirname(__file__), "INPUTS")

class TestFile(unittest.TestCase):
    def test_file(self):
        index = Index.create()
        tu = index.parse("t.c", unsaved_files=[("t.c", "")])
        file = File.from_name(tu, "t.c")
        self.assertEqual(str(file), "t.c")
        self.assertEqual(file.name, "t.c")
        self.assertEqual(repr(file), "<File: t.c>")

    def test_file_eq(self):
        path = os.path.join(inputs_dir, "hello.cpp")
        header_path = os.path.join(inputs_dir, "header3.h")
        tu = TranslationUnit.from_source(path)
        file1 = File.from_name(tu, path)
        file2 = File.from_name(tu, header_path)
        file2_2 = File.from_name(tu, header_path)

        self.assertEqual(file1, file1)
        self.assertEqual(file2, file2_2)
        self.assertNotEqual(file1, file2)
        self.assertNotEqual(file1, "t.c")

    def test_file_eq_failing(self):
        index = Index.create()
        tu = index.parse(
            "t.c",
            unsaved_files=[
                ("t.c", "int a = 729;"),
                ("s.c", "int a = 729;"),
            ],
        )
        file1 = File.from_name(tu, "t.c")
        file2 = File.from_name(tu, "s.c")
        # FIXME: These files are not supposed to be equal
        self.assertEqual(file1, file2)

    def test_file_eq_failing_2(self):
        index = Index.create()
        tu = index.parse(
            "t.c",
            unsaved_files=[
                ("t.c", "int a = 729;"),
                ("s.c", "int a = 728;"),
            ],
        )
        file1 = File.from_name(tu, "t.c")
        file2 = File.from_name(tu, "s.c")
        # FIXME: These files are not supposed to be equal
        self.assertEqual(file1, file2)

    def test_file_eq_failing_3(self):
        index = Index.create()
        tu = index.parse(
            "t.c",
            unsaved_files=[
                ("t.c", '#include "a.c"\n#include "b.c";'),
                ("a.c", "int a = 729;"),
                ("b.c", "int b = 729;"),
            ],
        )
        file1 = File.from_name(tu, "t.c")
        file2 = File.from_name(tu, "a.c")
        file3 = File.from_name(tu, "b.c")
        # FIXME: These files are not supposed to be equal
        self.assertEqual(file2, file3)
        self.assertEqual(file1, file2)
        self.assertEqual(file1, file3)

    def test_file_eq_failing_4(self):
        path = os.path.join(inputs_dir, "testfile.c")
        tu = TranslationUnit.from_source(path)
        file1 = File.from_name(tu, "testfile.c")
        file2 = File.from_name(tu, "a.c")
        file3 = File.from_name(tu, "b.c")
        # FIXME: These files are not supposed to be equal
        self.assertEqual(file2, file3)
        self.assertEqual(file1, file2)
        self.assertEqual(file1, file3)
