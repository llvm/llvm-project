import os

config.name = 'SanitizerBinaryMetadata'
config.test_source_root = os.path.dirname(__file__)
config.suffixes = ['.cpp']
# Binary metadata is currently emited only for ELF binaries
# and sizes of stack arguments depend on the arch.
if config.host_os not in ['Linux'] or config.target_arch not in ['x86_64']:
   config.unsupported = True
