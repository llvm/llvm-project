import os

config.name = "SanitizerBinaryMetadata"
config.test_source_root = os.path.dirname(__file__)
config.suffixes = [".cpp"]
# Binary metadata is currently emitted only for ELF binaries
# and sizes of stack arguments depend on the arch.
if config.target_os not in ["Linux"] or config.target_arch not in ["x86_64"]:
    config.unsupported = True
