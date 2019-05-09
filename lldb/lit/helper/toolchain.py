import os
import itertools
import platform
import subprocess
import sys

import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

def use_lldb_substitutions(config):
    # Set up substitutions for primary tools.  These tools must come from config.lldb_tools_dir
    # which is basically the build output directory.  We do not want to find these in path or
    # anywhere else, since they are specifically the programs which are actually being tested.

    dsname = 'debugserver' if platform.system() in ['Darwin'] else 'lldb-server'
    dsargs = [] if platform.system() in ['Darwin'] else ['gdbserver']
    lldbmi = ToolSubst('%lldbmi',
                       command=FindTool('lldb-mi'),
                       extra_args=['--synchronous'],
                       unresolved='ignore')


    build_script = os.path.dirname(__file__)
    build_script = os.path.join(build_script, 'build.py')
    build_script_args = [build_script,
                        '--compiler=any', # Default to best compiler
                        '--arch=' + str(config.lldb_bitness)]
    if config.lldb_lit_tools_dir:
        build_script_args.append('--tools-dir={0}'.format(config.lldb_lit_tools_dir))
    if config.lldb_tools_dir:
        build_script_args.append('--tools-dir={0}'.format(config.lldb_tools_dir))
    if config.llvm_libs_dir:
        build_script_args.append('--libs-dir={0}'.format(config.llvm_libs_dir))

    primary_tools = [
        ToolSubst('%lldb',
                  command=FindTool('lldb'),
                  extra_args=['--no-lldbinit', '-S',
                              os.path.join(config.test_source_root,
                                           'lit-lldb-init')]),
        ToolSubst('%lldb-init',
                  command=FindTool('lldb'),
                  extra_args=['-S',
                              os.path.join(config.test_source_root,
                                           'lit-lldb-init')]),
        lldbmi,
        ToolSubst('%debugserver',
                  command=FindTool(dsname),
                  extra_args=dsargs,
                  unresolved='ignore'),
        ToolSubst('%platformserver',
                  command=FindTool('lldb-server'),
                  extra_args=['platform'],
                  unresolved='ignore'),
        'lldb-test',
        'lldb-instr',
        ToolSubst('%build',
                  command="'" + sys.executable + "'",
                  extra_args=build_script_args)
        ]

    llvm_config.add_tool_substitutions(primary_tools,
                                       [config.lldb_tools_dir])
    # lldb-mi always fails without Python support
    if lldbmi.was_resolved and not config.lldb_disable_python:
        config.available_features.add('lldb-mi')

def _use_msvc_substitutions(config):
    # If running from a Visual Studio Command prompt (e.g. vcvars), this will
    # detect the include and lib paths, and find cl.exe and link.exe and create
    # substitutions for each of them that explicitly specify /I and /L paths
    cl = lit.util.which('cl')
    link = lit.util.which('link')

    if not cl or not link:
        return

    cl = '"' + cl + '"'
    link = '"' + link + '"'
    includes = os.getenv('INCLUDE', '').split(';')
    libs = os.getenv('LIB', '').split(';')

    config.available_features.add('msvc')
    compiler_flags = ['"/I{}"'.format(x) for x in includes if os.path.exists(x)]
    linker_flags = ['"/LIBPATH:{}"'.format(x) for x in libs if os.path.exists(x)]

    tools = [
        ToolSubst('%msvc_cl', command=cl, extra_args=compiler_flags),
        ToolSubst('%msvc_link', command=link, extra_args=linker_flags)]
    llvm_config.add_tool_substitutions(tools)
    return

def use_support_substitutions(config):
    # Set up substitutions for support tools.  These tools can be overridden at the CMake
    # level (by specifying -DLLDB_LIT_TOOLS_DIR), installed, or as a last resort, we can use
    # the just-built version.
    flags = []
    if platform.system() in ['Darwin']:
        try:
            out = subprocess.check_output(['xcrun', '--show-sdk-path']).strip()
            res = 0
        except OSError:
            res = -1
        if res == 0 and out:
            sdk_path = lit.util.to_string(out)
            llvm_config.lit_config.note('using SDKROOT: %r' % sdk_path)
            flags = ['-isysroot', sdk_path]
    elif platform.system() in ['NetBSD', 'OpenBSD', 'Linux']:
        flags = ['-pthread']

    if sys.platform.startswith('netbsd'):
        # needed e.g. to use freshly built libc++
        flags += ['-L' + config.llvm_libs_dir,
                  '-Wl,-rpath,' + config.llvm_libs_dir]

    additional_tool_dirs=[]
    if config.lldb_lit_tools_dir:
        additional_tool_dirs.append(config.lldb_lit_tools_dir)

    llvm_config.use_clang(additional_flags=flags,
                          additional_tool_dirs=additional_tool_dirs,
                          required=True)

    if sys.platform == 'win32':
        _use_msvc_substitutions(config)

    have_lld = llvm_config.use_lld(additional_tool_dirs=additional_tool_dirs,
                                   required=False)
    if have_lld:
        config.available_features.add('lld')


    support_tools = ['yaml2obj', 'obj2yaml', 'llvm-pdbutil',
                     'llvm-mc', 'llvm-readobj', 'llvm-objdump',
                     'llvm-objcopy', 'lli']
    additional_tool_dirs += [config.lldb_tools_dir, config.llvm_tools_dir]
    llvm_config.add_tool_substitutions(support_tools, additional_tool_dirs)
