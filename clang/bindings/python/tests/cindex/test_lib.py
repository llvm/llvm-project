import os

from clang.cindex import Config, conf, FUNCTION_LIST

if "CLANG_LIBRARY_PATH" in os.environ:
    Config.set_library_path(os.environ["CLANG_LIBRARY_PATH"])

import unittest


class TestIndex(unittest.TestCase):
    def test_functions_registered(self):
        IGNORED = set(["_FuncPtr", "_name", "_handle"])
        lib_functions = set(vars(conf.lib).keys())
        registered_functions = set([item[0] for item in FUNCTION_LIST])
        unregistered_functions = lib_functions - registered_functions - IGNORED
        self.assertEqual(unregistered_functions, set())
