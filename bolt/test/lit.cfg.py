# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'BOLT'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.cppm', '.m', '.mm', '.cu',
                   '.ll', '.cl', '.s', '.S', '.modulemap', '.test', '.rs']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.bolt_obj_root, 'test')

# checking if maxIndividualTestTime is available on the platform and sets
# it to 60sec if so, declares lit-max-individual-test-time feature for
# further checking by tests.
supported, errormsg = lit_config.maxIndividualTestTimeIsSupported
if supported:
    config.available_features.add("lit-max-individual-test-time")
    lit_config.maxIndividualTestTime = 60
else:
    lit_config.warning('Setting a timeout per test not supported. ' + errormsg
                       + ' Some tests will be skipped.')

if config.bolt_enable_runtime:
    config.available_features.add("bolt-runtime")

if config.gnu_ld:
    config.available_features.add("gnu_ld")

llvm_config.use_default_substitutions()

llvm_config.config.environment['CLANG'] = config.bolt_clang
llvm_config.use_clang()

llvm_config.config.environment['LD_LLD'] = config.bolt_lld
ld_lld = llvm_config.use_llvm_tool('ld.lld', required=True, search_env='LD_LLD')
llvm_config.config.available_features.add('ld.lld')
llvm_config.add_tool_substitutions([ToolSubst(r'ld\.lld', command=ld_lld)])

config.substitutions.append(('%cflags', ''))
config.substitutions.append(('%cxxflags', ''))

link_fdata_cmd = os.path.join(config.test_source_root, 'link_fdata.py')

tool_dirs = [config.llvm_tools_dir,
             config.test_source_root]

tools = [
    ToolSubst('llc', unresolved='fatal'),
    ToolSubst('llvm-dwarfdump', unresolved='fatal'),
    ToolSubst('llvm-bolt', unresolved='fatal'),
    ToolSubst('llvm-boltdiff', unresolved='fatal'),
    ToolSubst('llvm-bolt-heatmap', unresolved='fatal'),
    ToolSubst('llvm-bat-dump', unresolved='fatal'),
    ToolSubst('perf2bolt', unresolved='fatal'),
    ToolSubst('yaml2obj', unresolved='fatal'),
    ToolSubst('llvm-mc', unresolved='fatal'),
    ToolSubst('llvm-nm', unresolved='fatal'),
    ToolSubst('llvm-objdump', unresolved='fatal'),
    ToolSubst('llvm-objcopy', unresolved='fatal'),
    ToolSubst('llvm-strip', unresolved='fatal'),
    ToolSubst('llvm-readelf', unresolved='fatal'),
    ToolSubst('link_fdata', command=sys.executable, unresolved='fatal', extra_args=[link_fdata_cmd]),
    ToolSubst('merge-fdata', unresolved='fatal'),
    ToolSubst('llvm-readobj', unresolved='fatal'),
    ToolSubst('llvm-dwp', unresolved='fatal'),
    ToolSubst('split-file', unresolved='fatal'),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + '-registered-target')
    return features


llvm_config.feature_config(
    [('--assertion-mode', {'ON': 'asserts'}),
     ('--cxxflags', {r'-D_GLIBCXX_DEBUG\b': 'libstdcxx-safe-mode'}),
        ('--targets-built', calculate_arch_features)
     ])

config.targets = frozenset(config.targets_to_build.split())
