import os
import subprocess
import sys

#### UTILITIES ####


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

#### SETTINGS ####

#### INTERFACE TO THE XCODEPROJ ####


def lldb_source_path():
    path = os.environ.get('SRCROOT')
    if path:
         return path
    else:
         return "./"


def expected_llvm_build_path():
    if build_type() == BuildType.Xcode:
        return package_build_path()
    else:
        return os.path.join(
            os.environ.get('LLDB_PATH_TO_LLVM_BUILD'),
            package_build_dir_name("llvm"))


def archives_txt():
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

def rc_platform_name():
  return os.environ.get('RC_PLATFORM_NAME', 'macOS')

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

BuildType = enum('Xcode')                # (Debug,DebugClang,Release)


def build_type():
    return BuildType.Xcode

#### VCS UTILITIES ####

VCS = enum('git',
           'svn')


def run_in_directory(args, path):
    return subprocess.check_output([str(arg) for arg in args], cwd=path)


class Git:

    def __init__(self, spec):
        self.spec = spec

    def status(self):
        return run_in_directory(["git", "branch", "-v"], self.spec['root'])

    def diff(self):
        return run_in_directory(["git", "diff"], self.spec['root'])

    def check_out(self):
        run_in_directory(["git",
                          "clone",
                          "--depth=1",
                          self.spec['url'],
                          self.spec['root']],
                         lldb_source_path())
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
    if build_type() == BuildType.Xcode:
        return os.path.join(lldb_source_path(), "llvm")


def clang_source_path():
    if build_type() == BuildType.Xcode:
        return os.path.join(llvm_source_path(), "tools", "clang")


def ninja_source_path():
    if build_type() == BuildType.Xcode:
        return os.path.join(lldb_source_path(), "ninja")

#### BUILD PATHS ####


def packages():
    return ["llvm"]


def package_build_dir_name(package):
    return package + "-" + architecture()


def expected_package_build_path_for(package):
    if build_type() == BuildType.Xcode:
        if package != "llvm":
            raise "On Xcode build, we only support the llvm package: requested {}"
        return package_build_path()
    return os.path.join(
        expected_package_build_path(),
        package_build_dir_name(package))


def expected_package_build_paths():
    return [expected_package_build_path_for(package) for package in packages()]


def library_path(build_path):
    return build_path + "/lib"


def library_paths():
    if build_type() == BuildType.Xcode:
        package_build_paths = [package_build_path()]
    else:
        package_build_paths = expected_package_build_paths()
    return [library_path(build_path) for build_path in package_build_paths]


def package_build_path():
    return os.path.join(
        llvm_build_dirtree(),
        os.environ["LLVM_CONFIGURATION"],
        "x86_64")
