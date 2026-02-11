# -*- Python -*-

import os

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool

# Setup config name.
config.name = "pfp" + config.name_suffix

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

# Setup source root.
config.test_source_root = os.path.dirname(__file__)
# Setup default compiler flags used with -fsanitize=memory option.
clang_cflags = [config.target_cflags] + config.debug_info_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags
clang_pfp_tagged_common_cflags = clang_cflags + [
    "-fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged"
]


clang_pfp_cxxflags = config.cxx_mode_flags + clang_pfp_tagged_common_cflags
clang_pfp_cxxflags = clang_pfp_cxxflags + ["-fuse-ld=lld --rtlib=compiler-rt --unwindlib=libunwind  -static-libgcc"]


def build_invocation(compile_flags, with_lto=False):
    lto_flags = []
    if with_lto and config.lto_supported:
        lto_flags += config.lto_flags

    return " " + " ".join([config.clang] + lto_flags + compile_flags) + " "


config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))
config.substitutions.append(("%clangxx_pfp ", build_invocation(clang_pfp_cxxflags, config.use_thinlto)))
