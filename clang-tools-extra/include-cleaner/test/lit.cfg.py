import lit.llvm

lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.use_default_substitutions()

config.name = 'ClangIncludeCleaner'
config.suffixes = ['.test', '.c', '.cpp']
config.excludes = ['Inputs']
config.test_format = lit.formats.ShTest(not lit.llvm.llvm_config.use_lit_shell)
config.test_source_root = config.clang_include_cleaner_source_dir + "/test"
config.test_exec_root = config.clang_include_cleaner_binary_dir + "/test"

config.environment['PATH'] = os.path.pathsep.join((
        config.clang_tools_dir,
        config.llvm_tools_dir,
        config.environment['PATH']))
