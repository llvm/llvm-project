# -*- Python -*-

import os
import subprocess

# Setup config name.
config.name = 'CRT' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)


def get_library_path(file):
    cmd = subprocess.Popen([config.clang.strip(),
                            config.target_cflags.strip(),
                            '-print-file-name=%s' % file],
                           stdout=subprocess.PIPE,
                           env=config.environment)
    if not cmd.stdout:
      lit_config.fatal("Couldn't find the library path for '%s'" % file)
    dir = cmd.stdout.read().strip()
    if sys.platform in ['win32'] and execute_external:
        # Don't pass dosish path separator to msys bash.exe.
        dir = dir.replace('\\', '/')
    # Ensure the result is an ascii string, across Python2.5+ - Python3.
    return str(dir.decode('ascii'))


def get_libgcc_file_name():
    cmd = subprocess.Popen([config.clang.strip(),
                            config.target_cflags.strip(),
                            '-print-libgcc-file-name'],
                           stdout=subprocess.PIPE,
                           env=config.environment)
    if not cmd.stdout:
      lit_config.fatal("Couldn't find the library path for '%s'" % file)
    dir = cmd.stdout.read().strip()
    if sys.platform in ['win32'] and execute_external:
        # Don't pass dosish path separator to msys bash.exe.
        dir = dir.replace('\\', '/')
    # Ensure the result is an ascii string, across Python2.5+ - Python3.
    return str(dir.decode('ascii'))


def build_invocation(compile_flags):
    return ' ' + ' '.join([config.clang] + compile_flags) + ' '


# Setup substitutions.
config.substitutions.append(
    ('%clang ', build_invocation([config.target_cflags])))
config.substitutions.append(
    ('%clangxx ',
     build_invocation(config.cxx_mode_flags + [config.target_cflags])))

base_lib = os.path.join(
    config.compiler_rt_libdir, "clang_rt.%%s%s.o" % config.target_suffix)
config.substitutions.append(('%crtbegin', base_lib % "crtbegin"))
config.substitutions.append(('%crtend', base_lib % "crtend"))

config.substitutions.append(
    ('%crt1', get_library_path('crt1.o')))
config.substitutions.append(
    ('%crti', get_library_path('crti.o')))
config.substitutions.append(
    ('%crtn', get_library_path('crtn.o')))

config.substitutions.append(
    ('%libgcc', get_libgcc_file_name()))

config.substitutions.append(
    ('%libstdcxx', '-l' + config.sanitizer_cxx_lib.lstrip('lib')))

# Default test suffixes.
config.suffixes = ['.c', '.cc', '.cpp']

if config.host_os not in ['Linux']:
    config.unsupported = True
