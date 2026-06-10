# -*- Python -*-

import os


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg.py " % attr_name
        )
    return attr_value


config.name = "ConcurrencySanitizer-GPU" + config.name_suffix
config.test_source_root = os.path.dirname(__file__)
config.suffixes = [".c", ".cpp"]

if not config.emulator:
    config.unsupported = True


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


target_cflags = [get_required_attr(config, "target_cflags")]
clang_csan_cflags = ["-fsanitize=concurrency", "-gline-tables-only"] + target_cflags
clang_csan_cxxflags = config.cxx_mode_flags + clang_csan_cflags

config.substitutions.append(("%clang_csan ", build_invocation(clang_csan_cflags)))
config.substitutions.append(("%clangxx_csan ", build_invocation(clang_csan_cxxflags)))
