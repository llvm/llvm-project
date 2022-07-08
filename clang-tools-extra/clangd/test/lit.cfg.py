import lit.llvm

lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.use_clang([], [], required=False)
lit.llvm.llvm_config.use_default_substitutions()

config.name = 'Clangd'
config.suffixes = ['.test']
config.excludes = ['Inputs']
config.test_format = lit.formats.ShTest(not lit.llvm.llvm_config.use_lit_shell)
config.test_source_root = config.clangd_source_dir + "/test"
config.test_exec_root = config.clangd_binary_dir + "/test"


# Used to enable tests based on the required targets. Can be queried with e.g.
#    REQUIRES: x86-registered-target
def calculate_arch_features(arch_string):
  return [arch.lower() + '-registered-target' for arch in arch_string.split()]


lit.llvm.llvm_config.feature_config([('--targets-built',
                                      calculate_arch_features)])

# Clangd-specific lit environment.
config.substitutions.append(('%clangd-benchmark-dir',
                             config.clangd_binary_dir + "/benchmarks"))

if config.clangd_build_xpc:
  config.available_features.add('clangd-xpc-support')

if config.clangd_enable_remote:
  config.available_features.add('clangd-remote-index')

if config.clangd_tidy_checks:
  config.available_features.add('clangd-tidy-checks')

if config.have_zlib:
  config.available_features.add('zlib')

if config.have_zstd:
  config.available_features.add('zstd')
