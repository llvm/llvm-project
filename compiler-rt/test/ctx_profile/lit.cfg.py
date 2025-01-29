# -*- Python -*-

import os
import platform
import re

import lit.formats

# Only run the tests on supported OSs.
if config.host_os not in ["Linux"]:
    config.unsupported = True


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg.py " % attr_name
        )
    return attr_value


# Setup config name.
config.name = "CtxProfile" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)
# Default test suffixes.
config.suffixes = [".c", ".cpp", ".test"]

config.substitutions.append(
    ("%clangxx ", " ".join([config.clang] + config.cxx_mode_flags) + " -ldl -lpthread ")
)

config.substitutions.append(
    (
        "%ctxprofilelib",
        "-L%s -lclang_rt.ctx_profile%s" % (config.compiler_rt_libdir, config.target_suffix)
    )
)
