#!/usr/bin/env python

import errno
import hashlib
import fnmatch
import os
import platform
import re
import subprocess
import sys

from lldbbuild import *

#### SETTINGS ####


def LLVM_HASH_INCLUDES_DIFFS():
    return False

# The use of "x = "..."; return x" here is important because tooling looks for
# it with regexps.  Only change how this works if you know what you are doing.


def LLVM_REF():
    llvm_ref = "master"
    return llvm_ref


def CLANG_REF():
    clang_ref = "master"
    return clang_ref

# For use with Xcode-style builds


def XCODE_REPOSITORIES():
    return [
        {'name': "llvm",
         'vcs': VCS.git,
         'root': llvm_source_path(),
         'url': "http://llvm.org/git/llvm.git",
         'ref': LLVM_REF()},

        {'name': "clang",
         'vcs': VCS.git,
         'root': clang_source_path(),
         'url': "http://llvm.org/git/clang.git",
         'ref': CLANG_REF()},

        {'name': "ninja",
         'vcs': VCS.git,
         'root': ninja_source_path(),
         'url': "https://github.com/ninja-build/ninja.git",
         'ref': "master"}
    ]


def get_c_compiler():
    return subprocess.check_output([
        'xcrun',
        '--sdk', 'macosx',
        '-find', 'clang'
    ]).rstrip()


def get_cxx_compiler():
    return subprocess.check_output([
        'xcrun',
        '--sdk', 'macosx',
        '-find', 'clang++'
    ]).rstrip()

#                 CFLAGS="-isysroot $(xcrun --sdk macosx --show-sdk-path) -mmacosx-version-min=${DARWIN_DEPLOYMENT_VERSION_OSX}" \
#                        LDFLAGS="-mmacosx-version-min=${DARWIN_DEPLOYMENT_VERSION_OSX}" \


def get_deployment_target():
    return os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)


def get_c_flags():
    cflags = ''
    # sdk_path = subprocess.check_output([
    #     'xcrun',
    #     '--sdk', 'macosx',
    #     '--show-sdk-path']).rstrip()
    # cflags += '-isysroot {}'.format(sdk_path)

    deployment_target = get_deployment_target()
    if deployment_target:
        # cflags += ' -mmacosx-version-min={}'.format(deployment_target)
        pass

    return cflags


def get_cxx_flags():
    return get_c_flags()


def get_common_linker_flags():
    linker_flags = ""
    deployment_target = get_deployment_target()
    if deployment_target:
        # if len(linker_flags) > 0:
        #     linker_flags += ' '
        # linker_flags += '-mmacosx-version-min={}'.format(deployment_target)
        pass

    return linker_flags


def get_exe_linker_flags():
    return get_common_linker_flags()


def get_shared_linker_flags():
    return get_common_linker_flags()


def CMAKE_FLAGS():
    return {
        "Debug": [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
        ],
        "DebugClang": [
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
        ],
        "Release": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
        ],
        "BuildAndIntegration": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_ASSERTIONS=OFF",
        ],
    }


def CMAKE_ENVIRONMENT():
    return {
    }

#### COLLECTING ALL ARCHIVES ####


def collect_archives_in_path(path):
    files = os.listdir(path)
    # Only use libclang and libLLVM archives, and exclude libclang_rt
    regexp = "^lib(clang[^_]|LLVM|gtest).*$"
    return [
        os.path.join(
            path,
            file) for file in files if file.endswith(".a") and re.match(
            regexp,
            file)]


def archive_list():
    paths = library_paths()
    archive_lists = [collect_archives_in_path(path) for path in paths]
    return [archive for archive_list in archive_lists for archive in archive_list]


def write_archives_txt():
    f = open(archives_txt(), 'w')
    for archive in archive_list():
        f.write(archive + "\n")
    f.close()

#### COLLECTING REPOSITORY MD5S ####


def source_control_status(spec):
    vcs_for_spec = vcs(spec)
    if LLVM_HASH_INCLUDES_DIFFS():
        return vcs_for_spec.status() + vcs_for_spec.diff()
    else:
        return vcs_for_spec.status()


def source_control_status_for_specs(specs):
    statuses = [source_control_status(spec) for spec in specs]
    return "".join(statuses)


def all_source_control_status():
    return source_control_status_for_specs(XCODE_REPOSITORIES())


def md5(string):
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()


def all_source_control_status_md5():
    return md5(all_source_control_status())

#### CHECKING OUT AND BUILDING LLVM ####


def apply_patches(spec):
    files = os.listdir(os.path.join(lldb_source_path(), 'scripts'))
    patches = [
        f for f in files if fnmatch.fnmatch(
            f, spec['name'] + '.*.diff')]
    for p in patches:
        run_in_directory(["patch",
                          "-p0",
                          "-i",
                          os.path.join(lldb_source_path(),
                                       'scripts',
                                       p)],
                         spec['root'])


def check_out_if_needed(spec):
    if not os.path.isdir(spec['root']):
        vcs(spec).check_out()
        apply_patches(spec)


def all_check_out_if_needed():
    map(check_out_if_needed, XCODE_REPOSITORIES())


def should_build_llvm():
    if build_type() == BuildType.Xcode:
        # TODO use md5 sums
        return True


def do_symlink(source_path, link_path):
    print "Symlinking " + source_path + " to " + link_path
    if not os.path.exists(link_path):
        os.symlink(source_path, link_path)


def setup_source_symlink(repo):
    source_path = repo["root"]
    link_path = os.path.join(lldb_source_path(), os.path.basename(source_path))
    do_symlink(source_path, link_path)


def setup_source_symlinks():
    map(setup_source_symlink, XCODE_REPOSITORIES())


def setup_build_symlink():
    # We don't use the build symlinks in llvm.org Xcode-based builds.
    if build_type() != BuildType.Xcode:
        source_path = package_build_path()
        link_path = expected_package_build_path()
        do_symlink(source_path, link_path)


def should_run_cmake(cmake_build_dir):
    # We need to run cmake if our llvm build directory doesn't yet exist.
    if not os.path.exists(cmake_build_dir):
        return True

    # Wee also need to run cmake if for some reason we don't have a ninja
    # build file.  (Perhaps the cmake invocation failed, which this current
    # build may have fixed).
    ninja_path = os.path.join(cmake_build_dir, "build.ninja")
    return not os.path.exists(ninja_path)


def cmake_environment():
    cmake_env = join_dicts(os.environ, CMAKE_ENVIRONMENT())
    return cmake_env


def is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def find_executable_in_paths(program, paths_to_check):
    program_dir, program_name = os.path.split(program)
    if program_dir:
        if is_executable(program):
            return program
    else:
        for path_dir in paths_to_check:
            path_dir = path_dir.strip('"')
            executable_file = os.path.join(path_dir, program)
            if is_executable(executable_file):
                return executable_file
    return None


def find_cmake():
    # First check the system PATH env var for cmake
    cmake_binary = find_executable_in_paths(
        "cmake", os.environ["PATH"].split(os.pathsep))
    if cmake_binary:
        # We found it there, use it.
        return cmake_binary

    # Check a few more common spots.  Xcode launched from Finder
    # will have the default environment, and may not have
    # all the normal places present.
    extra_cmake_dirs = [
        "/usr/local/bin",
        "/opt/local/bin",
        os.path.join(os.path.expanduser("~"), "bin")
    ]

    if platform.system() == "Darwin":
        # Add locations where an official CMake.app package may be installed.
        extra_cmake_dirs.extend([
            os.path.join(
                os.path.expanduser("~"),
                "Applications",
                "CMake.app",
                "Contents",
                "bin"),
            os.path.join(
                os.sep,
                "Applications",
                "CMake.app",
                "Contents",
                "bin")])

    cmake_binary = find_executable_in_paths("cmake", extra_cmake_dirs)
    if cmake_binary:
        # We found it in one of the usual places.  Use that.
        return cmake_binary

    # We couldn't find cmake.  Tell the user what to do.
    raise Exception(
        "could not find cmake in PATH ({}) or in any of these locations ({}), "
        "please install cmake or add a link to it in one of those locations".format(
            os.environ["PATH"], extra_cmake_dirs))


def cmake_flags():
    cmake_flags = CMAKE_FLAGS()[lldb_configuration()]
    cmake_flags += ["-GNinja",
                    "-DCMAKE_C_COMPILER={}".format(get_c_compiler()),
                    "-DCMAKE_CXX_COMPILER={}".format(get_cxx_compiler()),
                    "-DCMAKE_INSTALL_PREFIX={}".format(expected_package_build_path_for("llvm")),
                    "-DCMAKE_C_FLAGS={}".format(get_c_flags()),
                    "-DCMAKE_CXX_FLAGS={}".format(get_cxx_flags()),
                    "-DCMAKE_EXE_LINKER_FLAGS={}".format(get_exe_linker_flags()),
                    "-DCMAKE_SHARED_LINKER_FLAGS={}".format(get_shared_linker_flags())]
    deployment_target = get_deployment_target()
    if deployment_target:
        cmake_flags.append(
            "-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(deployment_target))
    return cmake_flags


def run_cmake(cmake_build_dir, ninja_binary_path):
    cmake_binary = find_cmake()
    print "found cmake binary: using \"{}\"".format(cmake_binary)

    command_line = [cmake_binary] + cmake_flags() + [
        "-DCMAKE_MAKE_PROGRAM={}".format(ninja_binary_path),
        llvm_source_path()]
    print "running cmake like so: ({}) in dir ({})".format(command_line, cmake_build_dir)

    subprocess.check_call(
        command_line,
        cwd=cmake_build_dir,
        env=cmake_environment())


def create_directories_as_needed(path):
    try:
        os.makedirs(path)
    except OSError as error:
        # An error indicating that the directory exists already is fine.
        # Anything else should be passed along.
        if error.errno != errno.EEXIST:
            raise error


def run_cmake_if_needed(ninja_binary_path):
    cmake_build_dir = package_build_path()
    if should_run_cmake(cmake_build_dir):
        # Create the build directory as needed
        create_directories_as_needed(cmake_build_dir)
        run_cmake(cmake_build_dir, ninja_binary_path)


def build_ninja_if_needed():
    # First check if ninja is in our path.  If so, there's nothing to do.
    ninja_binary_path = find_executable_in_paths(
        "ninja", os.environ["PATH"].split(os.pathsep))
    if ninja_binary_path:
        # It's on the path.  cmake will find it.  We're good.
        print "found ninja here: \"{}\"".format(ninja_binary_path)
        return ninja_binary_path

    # Figure out if we need to build it.
    ninja_build_dir = ninja_source_path()
    ninja_binary_path = os.path.join(ninja_build_dir, "ninja")
    if not is_executable(ninja_binary_path):
        # Build ninja
        command_line = ["python", "configure.py", "--bootstrap"]
        print "building ninja like so: ({}) in dir ({})".format(command_line, ninja_build_dir)
        subprocess.check_call(
            command_line,
            cwd=ninja_build_dir,
            env=os.environ)

    return ninja_binary_path


def join_dicts(dict1, dict2):
    d = dict1.copy()
    d.update(dict2)
    return d


def build_llvm(ninja_binary_path):
    cmake_build_dir = package_build_path()
    subprocess.check_call(
        [ninja_binary_path],
        cwd=cmake_build_dir,
        env=cmake_environment())


def build_llvm_if_needed():
    if should_build_llvm():
        ninja_binary_path = build_ninja_if_needed()
        run_cmake_if_needed(ninja_binary_path)
        build_llvm(ninja_binary_path)
        setup_build_symlink()

#### MAIN LOGIC ####

all_check_out_if_needed()
build_llvm_if_needed()
write_archives_txt()

sys.exit(0)
