import os

import lit.Test

config.name = 'flang-OldUnit'

config.suffixes = [".test"]

config.test_source_root = os.path.join(config.flang_obj_root, 'unittests')
config.test_exec_root = config.test_source_root

config.test_format = lit.formats.ExecutableTest()

path = os.path.pathsep.join((config.flang_libs_dir, config.llvm_libs_dir,
                              config.environment.get('LD_LIBRARY_PATH','')))
config.environment['LD_LIBRARY_PATH'] = path
