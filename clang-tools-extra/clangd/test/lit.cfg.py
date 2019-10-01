import lit.llvm

lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.use_clang()

config.name = 'Clangd'
config.suffixes = ['.test']
config.excludes = ['Inputs']
config.test_format = lit.formats.ShTest(not lit.llvm.llvm_config.use_lit_shell)
config.test_source_root = config.clangd_source_dir + "/test"
config.test_exec_root = config.clangd_binary_dir + "/test"

# Clangd-specific lit environment.
config.substitutions.append(('%clangd-benchmark-dir',
                             config.clangd_binary_dir + "/benchmarks"))

if config.clangd_build_xpc:
  config.available_features.add('clangd-xpc-support')

