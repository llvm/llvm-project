#!/usr/bin/env python

import argparse
import multiprocessing
import platform
import shutil
import subprocess
import sys
from pathlib import Path, PurePath

import settings


def default_toolchain():
    system = platform.uname()[0]
    machine = platform.uname()[4]
    toolchain = Path(settings.THIS_DIR,
                             'cmake',
                             f'toolchain_{system.lower()}_{machine.lower()}.cmake')
    return toolchain if toolchain.is_file else None


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Help configure and build Flang')

    buildopt = parser.add_argument_group('general build options')
    buildopt.add_argument('-d', '--builddir', metavar='DIR', default='build',
                          help='Specify build directory (default: "projects_root_dir"/%(default)s) \
                          Note: The name of the build directory will be the same for flang and libpgmath')
    buildopt.add_argument('--clean', action='store_true', default=False,
                          help='Clean build')
    buildopt.add_argument('-b', '--build-type', metavar='TYPE', choices=['Release', 'Debug', "RelWithDebInfo", "MinSizeRel"],
                          default='Release', help='set build type (default: %(choices)s)')
    buildopt.add_argument('--cmake-param', metavar='OPT', action='append', default=[],
                          help='Add custom argument to CMake')
    buildopt.add_argument('-l', '--llvm-source-dir', metavar='DIR', default='',
                          help='Specify LLVM source directory, usually \
                                "/path/to/classic-flang-llvm-project/llvm" (default: %(default)s)')
    buildopt.add_argument('-p', '--install-prefix', metavar='PATH', nargs='?', default=None, const=False,
                          help='Install after build, also specify LLVM dir')
    buildopt.add_argument('--toolchain', metavar='FILE', default=default_toolchain().as_posix(),
                          help='Specify toolchain file (default: %(default)s)')
    buildopt.add_argument('-t', '--target', metavar='ARCH', choices=['X86', 'AArch64', 'PowerPC'], default='X86',
                          help='Control which targets are enabled (%(choices)s)')
    buildopt.add_argument('-j', '--jobs', metavar='N', type=int, default=multiprocessing.cpu_count(),
                          help='Number of parallel build jobs (default: %(default)s)')
    buildopt.add_argument('-c', '--use-ccache', action="store_true", default=False,
                          help='Using ccache during the build (default: %(default)s)')
    buildopt.add_argument('-v', '--verbose', action='store_true', default=False,
                          help='Verbose build (default: %(default)s)')

    arguments = parser.parse_args()
    return arguments


def generate_buildoptions(arguments):
    install_root = Path(arguments.install_prefix)

    base_cmake_args = [
        f'-DCMAKE_INSTALL_PREFIX={install_root.as_posix()}',
        f'-DCMAKE_BUILD_TYPE={arguments.build_type}',
        f'-DCMAKE_TOOLCHAIN_FILE={arguments.toolchain}'
    ]

    generator = 'Ninja' if sys.platform == 'win32' else 'Unix Makefiles'
    base_cmake_args.append(f'-G{generator}')

    if arguments.use_ccache:
        base_cmake_args.append('-DCMAKE_C_COMPILER_LAUNCHER=ccache')
        base_cmake_args.append('-DCMAKE_CXX_COMPILER_LAUNCHER=ccache')

    if arguments.cmake_param:
        base_cmake_args.extend(arguments.cmake_param)

    if arguments.llvm_source_dir:
        base_cmake_args.append(f'-DLLVM_MAIN_SRC_DIR={arguments.llvm_source_dir}')

    if arguments.verbose:
        base_cmake_args.append('-DCMAKE_VERBOSE_MAKEFILE=ON')

    return base_cmake_args


def normalize_builddir(project_srcdir, builddir, clean):
    build_path = ''
    if PurePath(builddir).is_absolute():
        build_path = Path(builddir, PurePath(project_srcdir).name)
    else:
        build_path = Path(project_srcdir, builddir)

    if clean and build_path.exists():
        shutil.rmtree(build_path)

    return build_path.as_posix()


def configure_libpgmath(arguments):
    build_path = normalize_builddir(
        settings.LIBPGMATH_DIR, arguments.builddir, arguments.clean)

    build_options = generate_buildoptions(arguments)

    cmake_cmd = ['cmake', '-B', build_path, '-S', settings.LIBPGMATH_DIR.as_posix()]

    cmake_cmd.extend(build_options)
    subprocess.run(cmake_cmd, check=True)

    return build_path


def configure_flang(arguments):
    build_path = normalize_builddir(
        settings.FLANG_DIR, arguments.builddir, arguments.clean)

    build_options = generate_buildoptions(arguments)
    install_root = Path(arguments.install_prefix)
    executable_suffix = '.exe' if sys.platform == 'win32' else ""
    flang = install_root / 'bin' / f'flang{executable_suffix}'

    additional_options = [
        f'-DCMAKE_Fortran_COMPILER={flang.as_posix()}',
        '-DCMAKE_Fortran_COMPILER_ID=Flang',
        '-DFLANG_INCLUDE_DOCS=ON',
        '-DFLANG_LLVM_EXTENSIONS=ON',
        f'-DLLVM_TARGETS_TO_BUILD={arguments.target}',
        '-DWITH_WERROR=ON'
    ]
    build_options.extend(additional_options)
    cmake_cmd = ['cmake', '-B', build_path, '-S', settings.FLANG_DIR.as_posix()]

    cmake_cmd.extend(build_options)
    subprocess.run(cmake_cmd, check=True)

    return build_path


def build_project(build_path, arguments):
    build_cmd = ['cmake', '--build', build_path, '--config',
                 arguments.build_type, '-j', str(arguments.jobs)]
    subprocess.run(build_cmd, check=True)


def install_project(build_path, arguments):
    install_cmd = ['cmake', '--build', build_path, '--config',
                   arguments.build_type, '--target', 'install']
    subprocess.run(install_cmd, check=True)


def print_success():
    print()
    print('=' * 30)
    print('Build succeeded!')
    print('=' * 30)
    print()


def print_header(title):
    print()
    print('*' * 30)
    print(f'{title}...')
    print('*' * 30)
    print()


def main():
    arguments = get_arguments()

    print_header('Building libpgmath')
    build_path = configure_libpgmath(arguments)
    build_project(build_path, arguments)
    install_project(build_path, arguments)

    print_header('Building flang')
    build_path = configure_flang(arguments)
    build_project(build_path, arguments)
    install_project(build_path, arguments)


if __name__ == "__main__":
    main()
    print_success()
