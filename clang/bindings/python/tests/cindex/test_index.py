import os

from clang.cindex import Index, TranslationUnit


import unittest

INPUTS_DIR = os.path.join(os.path.dirname(__file__), "INPUTS")


class TestIndex(unittest.TestCase):
    def test_create(self):
        Index.create()

    # FIXME: test Index.read

    def test_parse(self):
        index = Index.create()
        self.assertIsInstance(index, Index)
        tu = index.parse(os.path.join(INPUTS_DIR, "hello.cpp"))
        self.assertIsInstance(tu, TranslationUnit)
        tu = index.parse(None, ["-c", os.path.join(INPUTS_DIR, "hello.cpp")])
        self.assertIsInstance(tu, TranslationUnit)
