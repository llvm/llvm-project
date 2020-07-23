#!/usr/bin/env python

# Tests for litlint.py
#
# Usage: python litlint_test.py
#
# Returns nonzero if any test fails

import litlint
import unittest

class TestLintLine(unittest.TestCase):
  def test_missing_run(self):
    f = litlint.LintLine
    self.assertEqual(f(' %t '),     ('missing %run before %t', 2))
    self.assertEqual(f(' %t\n'),    ('missing %run before %t', 2))
    self.assertEqual(f(' %t.so '),  (None, None))
    self.assertEqual(f(' %t.o '),   (None, None))
    self.assertEqual(f('%run %t '), (None, None))
    self.assertEqual(f('-o %t '),   (None, None))

if __name__ == '__main__':
  unittest.main()
