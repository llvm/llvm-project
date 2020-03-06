# -*- Python -*-

from lit.llvm import llvm_config

# Added this line file to prevent lit from discovering these tests
# See Issue #1052
config.suffixes = []
