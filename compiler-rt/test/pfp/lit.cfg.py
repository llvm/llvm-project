# -*- Python -*-

import os

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool

# Setup config name.
config.name = "pfp" + config.name_suffix

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

if config.host_os not in ["Linux"]:
    config.unsupported = True

# Setup source root.
config.test_source_root = os.path.dirname(__file__)
# Setup default compiler flags used with -fsanitize=memory option.
clang_cflags = [config.target_cflags] + config.debug_info_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags
clang_pfp_tagged_common_cflags = clang_cflags + [
    "-fexperimental-pointer-field-protection=tagged"
]


clang_pfp_cxxflags = config.cxx_mode_flags + clang_pfp_tagged_common_cflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))
config.substitutions.append(("%clangxx_pfp ", build_invocation(clang_pfp_cxxflags)))
