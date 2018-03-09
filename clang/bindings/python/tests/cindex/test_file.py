from clang.cindex import Index, File

import unittest


class TestFile(unittest.TestCase):
    def test_file(self):
        index = Index.create()
        tu = index.parse('t.c', unsaved_files = [('t.c', "")])
        file = File.from_name(tu, "t.c")
        self.assertEqual(str(file), "t.c")
        self.assertEqual(file.name, "t.c")
        self.assertEqual(repr(file), "<File: t.c>")
