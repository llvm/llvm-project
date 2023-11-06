import sys
import io
import unittest
import tempfile

from clang.cindex import (
    Rewriter,
    TranslationUnit,
    Config,
    File,
    SourceLocation,
    SourceRange,
)


class TestRewrite(unittest.TestCase):
    code = """
int test1;

void test2(void);

int f(int c) {
    return c;
}
"""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".cpp", buffering=0)
        self.tmp.write(TestRewrite.code.encode("utf-8"))
        self.tmp.flush()
        self.tu = TranslationUnit.from_source(self.tmp.name)
        self.rew = Rewriter.create(self.tu)
        self.file = File.from_name(self.tu, self.tmp.name)

    def tearDown(self):
        self.tmp.close()

    def test_insert(self):
        snip = "#include <cstdio>\n"

        beginning = SourceLocation.from_offset(self.tu, self.file, 0)
        self.rew.insertTextBefore(beginning, snip)
        self.rew.overwriteChangedFiles()

        with open(self.tmp.name, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), snip + TestRewrite.code)

    def test_replace(self):
        pattern = "test2"
        replacement = "func"

        offset = TestRewrite.code.find(pattern)
        pattern_range = SourceRange.from_locations(
            SourceLocation.from_offset(self.tu, self.file, offset),
            SourceLocation.from_offset(self.tu, self.file, offset + len(pattern)),
        )
        self.rew.replaceText(pattern_range, replacement)
        self.rew.overwriteChangedFiles()

        with open(self.tmp.name, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), TestRewrite.code.replace(pattern, replacement))

    def test_remove(self):
        pattern = "int c"

        offset = TestRewrite.code.find(pattern)
        pattern_range = SourceRange.from_locations(
            SourceLocation.from_offset(self.tu, self.file, offset),
            SourceLocation.from_offset(self.tu, self.file, offset + len(pattern)),
        )
        self.rew.removeText(pattern_range)
        self.rew.overwriteChangedFiles()

        with open(self.tmp.name, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), TestRewrite.code.replace(pattern, ""))
