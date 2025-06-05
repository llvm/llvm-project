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
        path = os.path.join(inputs_dir, "testfile.c")
        path_a = os.path.join(inputs_dir, "a.inc")
        path_b = os.path.join(inputs_dir, "b.inc")
        tu = TranslationUnit.from_source(path)
        main_file = File.from_name(tu, path)
        a_file = File.from_name(tu, path_a)
        a_file2 = File.from_name(tu, path_a)
        b_file = File.from_name(tu, path_b)

        self.assertEqual(a_file, a_file2)
        self.assertNotEqual(a_file, b_file)
        self.assertNotEqual(main_file, a_file)
        self.assertNotEqual(main_file, b_file)
        self.assertNotEqual(main_file, "t.c")

    def test_file_eq_in_memory(self):
        tu = TranslationUnit.from_source(
            "testfile.c",
            unsaved_files=[
                (
                    "testfile.c",
                    """
int a[] = { 
    #include "a.inc"
};
int b[] = { 
    #include "b.inc"
};
""",
                ),
                ("a.inc", "1,2,3"),
                ("b.inc", "1,2,3"),
            ],
        )

        path = os.path.join(inputs_dir, "testfile.c")
        path_a = os.path.join(inputs_dir, "a.inc")
        path_b = os.path.join(inputs_dir, "b.inc")
        tu = TranslationUnit.from_source(path)
        main_file = File.from_name(tu, path)
        a_file = File.from_name(tu, path_a)
        a_file2 = File.from_name(tu, path_a)
        b_file = File.from_name(tu, path_b)

        self.assertEqual(a_file, a_file2)
        self.assertNotEqual(a_file, b_file)
        self.assertNotEqual(main_file, a_file)
        self.assertNotEqual(main_file, b_file)
        self.assertNotEqual(main_file, "a.inc")
