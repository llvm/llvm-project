import os
import subprocess

#### UTILITIES ####


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

#### SETTINGS ####


def LLVM_BUILD_DIRS():
    return {
        "Debug": "Ninja-RelWithDebInfoAssert",
        "DebugClang": "Ninja-DebugAssert",
        "Release": "Ninja-RelWithDebInfoAssert",
    }

#### INTERFACE TO THE XCODEPROJ ####


def lldb_source_path():
    return os.environ.get('SRCROOT')


def expected_llvm_build_path():
    if build_type() == BuildType.CustomSwift:
        return os.environ.get('LLDB_PATH_TO_LLVM_BUILD')
    else:
        return os.path.join(
            os.environ.get('LLDB_PATH_TO_LLVM_BUILD'),
            package_build_dir_name("llvm"))


def archives_txt():
    if build_type() == BuildType.CustomSwift:
        return os.path.join(expected_llvm_build_path(), "archives.txt")
    else:
        return os.path.join(expected_package_build_path(), "archives.txt")


def expected_package_build_path():
    return os.path.abspath(os.path.join(expected_llvm_build_path(), ".."))

def is_host_build():
    rc_project_name = os.environ.get('RC_ProjectName')
    if rc_project_name:
      if rc_project_name == 'lldb_host': return True
    return False

def rc_release_target():
  return os.environ.get('RC_RELEASE', '')

def architecture():
    if is_host_build(): return 'macosx'
    platform_name = os.environ.get('RC_PLATFORM_NAME')
    if not platform_name:
        platform_name = os.environ.get('PLATFORM_NAME')
    platform_arch = os.environ.get('ARCHS').split()[-1]
    return platform_name + "-" + platform_arch


def lldb_configuration():
    return os.environ.get('CONFIGURATION')


def llvm_configuration():
    return os.environ.get('LLVM_CONFIGURATION')


def llvm_build_dirtree():
    return os.environ.get('LLVM_BUILD_DIRTREE')

# Edit the code below when adding build styles.

BuildType = enum('CustomSwift',          # CustomSwift-(Debug,Release)
                 'Xcode')                # (Debug,DebugClang,Release)


def build_type():
    configuration = lldb_configuration()
    if "CustomSwift" in configuration:
        return BuildType.CustomSwift
    return BuildType.Xcode

#### VCS UTILITIES ####

VCS = enum('git',
           'svn')


def run_in_directory(args, path):
    return subprocess.check_output(args, cwd=path)


class Git:

    def __init__(self, spec):
        self.spec = spec

    def status(self):
        return run_in_directory(["git", "branch", "-v"], self.spec['root'])

    def diff(self):
        return run_in_directory(["git", "diff"], self.spec['root'])

    def check_out(self):
        run_in_directory(["git", "clone", self.spec['url'],
                          self.spec['root']], lldb_source_path())
        run_in_directory(["git", "fetch", "--all"], self.spec['root'])
        run_in_directory(["git", "checkout", self.spec[
                         'ref']], self.spec['root'])


class SVN:

    def __init__(self, spec):
        self.spec = spec

    def status(self):
        return run_in_directory(["svn", "info"], self.spec['root'])

    def diff(self):
        return run_in_directory(["svn", "diff"], self.spec['root'])
    # TODO implement check_out


def vcs(spec):
    if spec['vcs'] == VCS.git:
        return Git(spec)
    elif spec['vcs'] == VCS.svn:
        return SVN(spec)
    else:
        return None

#### SOURCE PATHS ####


def llvm_source_path():
    if build_type() == BuildType.CustomSwift:
        return os.path.join(lldb_source_path(), "..", "llvm")
    elif build_type() == BuildType.Xcode:
        return os.path.join(lldb_source_path(), "llvm")


def clang_source_path():
    if build_type() == BuildType.CustomSwift:
        return os.path.join(lldb_source_path(), "..", "clang")
    elif build_type() == BuildType.Xcode:
        return os.path.join(llvm_source_path(), "tools", "clang")


def swift_source_path():
    if build_type() == BuildType.CustomSwift:
        return os.path.join(lldb_source_path(), "..", "swift")
    elif build_type() == BuildType.Xcode:
        return os.path.join(llvm_source_path(), "tools", "swift")


def cmark_source_path():
    if build_type() == BuildType.CustomSwift:
        return os.path.join(lldb_source_path(), "..", "cmark")
    elif build_type() == BuildType.Xcode:
        return os.path.join(llvm_source_path(), "tools", "cmark")


def ninja_source_path():
    if build_type() == BuildType.CustomSwift:
        return os.path.join(lldb_source_path(), "..", "ninja")
    elif build_type() == BuildType.Xcode:
        return os.path.join(llvm_source_path(), "tools", "ninja")

#### BUILD PATHS ####


def packages():
    return ["llvm", "swift", "cmark"]


def package_build_dir_name(package):
    return package + "-" + architecture()


def expected_package_build_path_for(package):
    return os.path.join(
        expected_package_build_path(),
        package_build_dir_name(package))


def expected_package_build_paths():
    return [expected_package_build_path_for(package) for package in packages()]


def library_path(build_path):
    if "cmark" in build_path:
        return build_path + "/src"
    else:
        return build_path + "/lib"


def library_paths():
    return [library_path(build_path)
            for build_path in expected_package_build_paths()]


def package_build_path():
    return os.path.join(
        llvm_build_dirtree(),
        LLVM_BUILD_DIRS()[
            lldb_configuration()])
