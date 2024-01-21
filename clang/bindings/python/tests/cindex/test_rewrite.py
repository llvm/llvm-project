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
    code = """int main() { return 0; }"""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".cpp", buffering=0)
        self.tmp.write(TestRewrite.code.encode("utf-8"))
        self.tmp.flush()
        self.tu = TranslationUnit.from_source(self.tmp.name)
        self.rew = Rewriter.create(self.tu)
        self.file = File.from_name(self.tu, self.tmp.name)

    def tearDown(self):
        self.tmp.close()

    def get_content(self) -> str:
        with open(self.tmp.name, "r", encoding="utf-8") as f:
            return f.read()

    def test_replace(self):
        rng = SourceRange.from_locations(
            SourceLocation.from_position(self.tu, self.file, 1, 5),
            SourceLocation.from_position(self.tu, self.file, 1, 9),
        )
        self.rew.replaceText(rng, "MAIN")
        self.rew.overwriteChangedFiles()
        self.assertEqual(self.get_content(), "int MAIN() { return 0; }")

    def test_replace_shorter(self):
        rng = SourceRange.from_locations(
            SourceLocation.from_position(self.tu, self.file, 1, 5),
            SourceLocation.from_position(self.tu, self.file, 1, 9),
        )
        self.rew.replaceText(rng, "foo")
        self.rew.overwriteChangedFiles()
        self.assertEqual(self.get_content(), "int foo() { return 0; }")

    def test_replace_longer(self):
        rng = SourceRange.from_locations(
            SourceLocation.from_position(self.tu, self.file, 1, 5),
            SourceLocation.from_position(self.tu, self.file, 1, 9),
        )
        self.rew.replaceText(rng, "patatino")
        self.rew.overwriteChangedFiles()
        self.assertEqual(self.get_content(), "int patatino() { return 0; }")

    def test_insert(self):
        pos = SourceLocation.from_position(self.tu, self.file, 1, 5)
        self.rew.insertTextBefore(pos, "ro")
        self.rew.overwriteChangedFiles()
        self.assertEqual(self.get_content(), "int romain() { return 0; }")

    def test_remove(self):
        rng = SourceRange.from_locations(
            SourceLocation.from_position(self.tu, self.file, 1, 5),
            SourceLocation.from_position(self.tu, self.file, 1, 9),
        )
        self.rew.removeText(rng)
        self.rew.overwriteChangedFiles()
        self.assertEqual(self.get_content(), "int () { return 0; }")
