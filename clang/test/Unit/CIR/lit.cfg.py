# -*- Python -*-

import os

# Load the Clang unit-test suite, then limit discovery to CIRUnitTests.
lit_config.load_config(
    config, os.path.join(os.path.dirname(__file__), "..", "lit.cfg.py")
)

config.name = "Clang-CIR-Unit"
config.test_exec_root = os.path.join(config.clang_obj_root, "unittests", "CIR")
config.test_source_root = config.test_exec_root
