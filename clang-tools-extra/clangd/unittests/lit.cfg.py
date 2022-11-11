import lit.formats
config.name = "Clangd Unit Tests"
config.test_format = lit.formats.GoogleTest('.', 'Tests')
config.test_source_root = config.clangd_binary_dir + "/unittests"
config.test_exec_root = config.clangd_binary_dir + "/unittests"

# Point the dynamic loader at dynamic libraries in 'lib'.
# FIXME: it seems every project has a copy of this logic. Move it somewhere.
import platform
if platform.system() == 'Darwin':
    shlibpath_var = 'DYLD_LIBRARY_PATH'
elif platform.system() == 'Windows':
    shlibpath_var = 'PATH'
else:
    shlibpath_var = 'LD_LIBRARY_PATH'
config.environment[shlibpath_var] = os.path.pathsep.join((
    "@SHLIBDIR@", "@LLVM_LIBS_DIR@",
    config.environment.get(shlibpath_var,'')))

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
