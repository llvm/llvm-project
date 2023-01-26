import lit.llvm

lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.use_default_substitutions()

config.name = 'ClangPseudo'
config.suffixes = ['.test', '.c', '.cpp']
config.excludes = ['Inputs']
config.test_format = lit.formats.ShTest(not lit.llvm.llvm_config.use_lit_shell)
config.test_source_root = config.clang_pseudo_source_dir + "/test"
config.test_exec_root = config.clang_pseudo_binary_dir + "/test"

config.environment['PATH'] = os.path.pathsep.join((
        config.clang_tools_dir,
        config.llvm_tools_dir,
        config.environment['PATH']))

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"
