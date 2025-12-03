import unittest
import unittest.mock
import sys
import os
from clang.cindex import Config


class TestEnvironementVariable(unittest.TestCase):

    def test_working_libclang_library_file(self):
        ref_libclang_library_file = Config().get_filename()
        with unittest.mock.patch.dict(
            os.environ, {"LIBCLANG_LIBRARY_FILE": ref_libclang_library_file}
        ):
            Config().lib

    @unittest.mock.patch.dict("os.environ", {"LIBCLANG_LIBRARY_FILE": "/dev/null"})
    def _test_non_working_libclang_library_file(self):
        with self.assertRaises(clang.cindex.LibclangError):
            Config().lib

    def test_working_libclang_library_path(self):
        ref_libclang_library_file = Config().get_filename()
        ref_libclang_library_path, filename = os.path.split(ref_libclang_library_file)
        filename_root, filename_ext = os.path.splitext(filename)

        # Config only recognizes the default libclang filename.
        # If LIBCLANG_LIBRARY_FILE points to a non-standard name, skip this test.

        if not (
            filename_root == "libclang" and filename_ext in (".so", ".dll", ".dylib")
        ):
            self.skipTest(f"Skipping because {filename} is not a default libclang name")

        with unittest.mock.patch.dict(
            os.environ, {"LIBCLANG_LIBRARY_PATH": ref_libclang_library_path}
        ):
            Config().lib

    @unittest.mock.patch.dict("os.environ", {"LIBCLANG_LIBRARY_PATH": "not_a_real_dir"})
    def _test_non_working_libclang_library_path(self):
        with self.assertRaises(clang.cindex.LibclangError):
            Config().lib
