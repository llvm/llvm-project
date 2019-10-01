# -*- Python -*-

import os

# Setup config name.
config.name = 'ShadowCallStack'

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.c', '.cc', '.cpp', '.m', '.mm', '.ll', '.test']

# Add clang substitutions.
config.substitutions.append( ("%clang_noscs ", config.clang + ' -O0 -fno-sanitize=shadow-call-stack ' + config.target_cflags + ' ') )

scs_arch_cflags = config.target_cflags
if config.target_arch == 'aarch64':
  scs_arch_cflags += ' -ffixed-x18 '
config.substitutions.append( ("%clang_scs ", config.clang + ' -O0 -fsanitize=shadow-call-stack ' + scs_arch_cflags + ' ') )

if config.host_os not in ['Linux'] or config.target_arch not in ['aarch64']:
   config.unsupported = True
