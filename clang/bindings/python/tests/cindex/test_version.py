import unittest
import re
from clang.cindex import Config


class TestClangVersion(unittest.TestCase):
    def test_get_version_format(self):
        conf = Config()
        version = conf.get_version()

        self.assertRegex(version, r"^clang version \d+\.\d+\.\d+")
