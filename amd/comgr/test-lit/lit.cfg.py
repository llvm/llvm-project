import os

import lit.formats
import lit.util

config.name = "Comgr"
config.suffixes = {".hip", ".cl", ".c", ".cpp"}
config.test_format = lit.formats.ShTest(True)

config.excludes = ["comgr-sources"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root

if not config.comgr_disable_spirv:
    config.available_features.add("comgr-has-spirv")

# By default, disable the cache for the tests.
# Test for the cache must explicitly enable this variable.
config.environment['AMD_COMGR_CACHE_DIR'] = ""
