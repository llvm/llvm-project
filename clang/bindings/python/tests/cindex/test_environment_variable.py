import unittest
import unittest.mock
import sys
import os


def reset_import_and_get_fresh_config():
    # Reloads the clang.cindex module to reset any class-level state in Config.
    sys.modules.pop("clang.cindex", None)
    sys.modules.pop("clang", None)
    from clang.cindex import Config

    return Config()


class TestEnvironementVariable(unittest.TestCase):
    def test_working_libclang_library_file(self):
        ref_libclang_library_file = reset_import_and_get_fresh_config().get_filename()
        with unittest.mock.patch.dict(
            os.environ, {"LIBCLANG_LIBRARY_FILE": ref_libclang_library_file}
        ):
            reset_import_and_get_fresh_config().lib

    @unittest.mock.patch.dict("os.environ", {"LIBCLANG_LIBRARY_FILE": "/dev/null"})
    def test_non_working_libclang_library_file(self):
        config = reset_import_and_get_fresh_config()
        import clang.cindex

        with self.assertRaises(clang.cindex.LibclangError):
            config.lib

    def test_working_libclang_library_path(self):
        # Get adequate libclang path
        ref_libclang_library_file = reset_import_and_get_fresh_config().get_filename()
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
            # Remove LIBCLANG_LIBRARY_FILE to avoid it taking precedence if set by the user
            # Need to be in the mocked environement
            os.environ.pop("LIBCLANG_LIBRARY_FILE", None)
            reset_import_and_get_fresh_config().lib

    @unittest.mock.patch.dict("os.environ", {"LIBCLANG_LIBRARY_PATH": "not_a_real_dir"})
    def test_non_working_libclang_library_path(self):
        # Remove LIBCLANG_LIBRARY_FILE to avoid it taking precedence if set by the user
        os.environ.pop("LIBCLANG_LIBRARY_FILE", None)

        config = reset_import_and_get_fresh_config()
        import clang.cindex

        with self.assertRaises(clang.cindex.LibclangError):
            config.lib
