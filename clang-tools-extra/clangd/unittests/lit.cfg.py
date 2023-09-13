import lit.formats
import lit.util

config.name = "Clangd Unit Tests"
config.test_format = lit.formats.GoogleTest(".", "Tests")
config.test_source_root = config.clangd_binary_dir + "/unittests"
config.test_exec_root = config.clangd_binary_dir + "/unittests"

# Point the dynamic loader at dynamic libraries in 'lib'.
# FIXME: it seems every project has a copy of this logic. Move it somewhere.
import platform

# Clangd unittests uses ~4 threads per test. So make sure we don't over commit.
core_count = lit.util.usable_core_count()
# FIXME: Split unittests into groups that use threads, and groups that do not,
# and only limit multi-threaded tests.
lit_config.parallelism_groups["clangd"] = max(1, core_count // 4)
config.parallelism_group = "clangd"

if platform.system() == "Darwin":
    shlibpath_var = "DYLD_LIBRARY_PATH"
elif platform.system() == "Windows":
    shlibpath_var = "PATH"
else:
    shlibpath_var = "LD_LIBRARY_PATH"
config.environment[shlibpath_var] = os.path.pathsep.join(
    ("@SHLIBDIR@", "@LLVM_LIBS_DIR@", config.environment.get(shlibpath_var, ""))
)

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
