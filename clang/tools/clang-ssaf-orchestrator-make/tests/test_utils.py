# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import utils


class MakeParents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_it_returns_the_path_it_was_given(self):
        path = os.path.join(self.tmp, "a", "b", "c.json")
        self.assertEqual(path, utils.make_parents(path))
        self.assertTrue(os.path.isdir(os.path.dirname(path)))

    def test_an_existing_directory_is_not_an_error(self):
        path = os.path.join(self.tmp, "a", "c.json")
        utils.make_parents(path)
        utils.make_parents(path)
        self.assertTrue(os.path.isdir(os.path.dirname(path)))

    def test_a_bare_name_has_no_parent_to_make(self):
        self.assertEqual("c.json", utils.make_parents("c.json"))

    def test_a_parent_that_cannot_be_made_is_not_swallowed(self):
        blocked = os.path.join(self.tmp, "file")
        open(blocked, "w").close()
        with self.assertRaises(OSError):
            utils.make_parents(os.path.join(blocked, "a", "c.json"))


class Json(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "doc.json")

    def test_a_document_round_trips(self):
        utils.write_json(self.path, {"a": [1, 2]})
        self.assertEqual({"a": [1, 2]}, utils.read_json(self.path))

    def test_writing_leaves_no_temporary_behind(self):
        utils.write_json(self.path, {})
        self.assertEqual(["doc.json"], os.listdir(self.tmp))

    def test_a_write_replaces_rather_than_appends(self):
        utils.write_json(self.path, {"a": 1})
        utils.write_json(self.path, {"b": 2})
        self.assertEqual({"b": 2}, utils.read_json(self.path))

    def test_an_absent_file_reads_as_nothing_and_says_nothing(self):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.assertIsNone(utils.read_json(os.path.join(self.tmp, "absent")))
        self.assertEqual("", stream.getvalue())

    def test_a_corrupt_file_reads_as_nothing_and_says_so(self):
        with open(self.path, "w") as handle:
            handle.write('{"a": ')
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            self.assertIsNone(utils.read_json(self.path))
        self.assertIn("is not valid JSON", stream.getvalue())
        self.assertIn(self.path, stream.getvalue())

    def test_the_two_cases_are_distinguishable(self):
        absent = io.StringIO()
        corrupt = io.StringIO()
        with open(self.path, "w") as handle:
            handle.write("{")
        with mock.patch("sys.stderr", absent):
            utils.read_json(os.path.join(self.tmp, "absent"))
        with mock.patch("sys.stderr", corrupt):
            utils.read_json(self.path)
        self.assertNotEqual(absent.getvalue(), corrupt.getvalue())

    def test_what_is_written_is_readable_json(self):
        utils.write_json(self.path, {"kind": "link", "inputs": ["a.o"]})
        with open(self.path) as handle:
            self.assertEqual({"kind": "link", "inputs": ["a.o"]}, json.load(handle))


if __name__ == "__main__":
    unittest.main()
