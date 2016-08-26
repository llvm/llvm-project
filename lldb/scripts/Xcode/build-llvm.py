#!/usr/bin/env python

import hashlib
import fnmatch
import os
import platform
import re
import subprocess
import sys

from lldbbuild import *

#### SETTINGS ####

def LLVM_HASH_INCLUDES_DIFFS ():
    return False

# The use of "x = "..."; return x" here is important because tooling looks for
# it with regexps.  Only change how this works if you know what you are doing.

def LLVM_REF ():
    llvm_ref = "stable-next"
    return llvm_ref

def CLANG_REF ():
    clang_ref = "stable-next"
    return clang_ref

def SWIFT_REF ():
    swift_ref = "master-next"
    return swift_ref

# For use with Xcode-style builds

def XCODE_REPOSITORIES ():
    return [
        { 'name':   "llvm",
          'vcs':    VCS.git,
          'root':   llvm_source_path(),
          'url':    "ssh://git@github.com/apple/swift-llvm.git",
          'ref':    LLVM_REF() },

        { 'name':   "clang",
          'vcs':    VCS.git,
          'root':   clang_source_path(),
          'url':    "ssh://git@github.com/apple/swift-clang.git",
          'ref':    CLANG_REF() },

        { 'name':   "swift",
          'vcs':    VCS.git,
          'root':   swift_source_path(),
          'url':    "ssh://git@github.com/apple/swift.git",
          'ref':    SWIFT_REF() },

        { 'name':   "cmark",
          'vcs':    VCS.git,
          'root':   cmark_source_path(),
          'url':    "ssh://git@github.com/apple/swift-cmark.git",
          'ref':    "master" },

        { 'name':   "ninja",
          'vcs':    VCS.git,
          'root':   ninja_source_path(),
          'url':    "https://github.com/ninja-build/ninja.git",
          'ref':    "master" }
    ]

def BUILD_SCRIPT_FLAGS ():
    return {
        "Debug":                ["--preset=LLDB_Swift_ReleaseAssert"],
        "DebugClang":           ["--preset=LLDB_Swift_DebugAssert"],
        "Release":              ["--preset=LLDB_Swift_ReleaseAssert"],
    }

def BUILD_SCRIPT_ENVIRONMENT ():
    return {
        "SWIFT_SOURCE_ROOT":    lldb_source_path(),
        "SWIFT_BUILD_ROOT":     llvm_build_dirtree()
    }

#### COLLECTING ALL ARCHIVES ####

def collect_archives_in_path (path): 
    files = os.listdir(path)
    # Only use libclang and libLLVM archives (and gtests), and exclude libclang_rt.
    # Also include swigt and cmark.
    # This is not a very scalable solution.  Direct dependency determination would
    # be preferred.
    regexp = "^lib(clang[^_]|LLVM|gtest|swift|cmark).*$"
    return [os.path.join(path, file) for file in files if file.endswith(".a") and re.match(regexp, file)]

def archive_list ():
    paths = library_paths()
    archive_lists = [collect_archives_in_path(path) for path in paths]
    return [archive for archive_list in archive_lists for archive in archive_list]

def write_archives_txt ():
    f = open(archives_txt(), 'w')
    for archive in archive_list():
        f.write(archive + "\n")
    f.close()

#### COLLECTING REPOSITORY MD5S ####

def source_control_status (spec):
    vcs_for_spec = vcs(spec)
    if LLVM_HASH_INCLUDES_DIFFS():
        return vcs_for_spec.status() + vcs_for_spec.diff()
    else:
        return vcs_for_spec.status()

def source_control_status_for_specs (specs):
    statuses = [source_control_status(spec) for spec in specs]
    return "".join(statuses)

def all_source_control_status ():
    return source_control_status_for_specs(XCODE_REPOSITORIES())

def md5 (string):
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()

def all_source_control_status_md5 ():
    return md5(all_source_control_status())

#### CHECKING OUT AND BUILDING LLVM ####

def apply_patches(spec):
    files = os.listdir(os.path.join(lldb_source_path(), 'scripts'))
    patches = [f for f in files if fnmatch.fnmatch(f, spec['name'] + '.*.diff')]
    for p in patches:
        run_in_directory(["patch", "-p0", "-i", os.path.join(lldb_source_path(), 'scripts', p)], spec['root'])

def check_out_if_needed(spec):
    if (build_type() != BuildType.CustomSwift) and not (os.path.isdir(spec['root'])):
        vcs(spec).check_out()
        apply_patches(spec)

def all_check_out_if_needed ():
    map (check_out_if_needed, XCODE_REPOSITORIES())

def should_build_llvm ():
    if build_type() == BuildType.CustomSwift:
        return False
    if build_type() == BuildType.Xcode:
        # TODO use md5 sums
        return True 

def do_symlink (source_path, link_path):
    print "Symlinking " + source_path + " to " + link_path
    if os.path.islink(link_path):
        os.remove(link_path)
    if not os.path.exists(link_path):
        os.symlink(source_path, link_path)

def setup_source_symlink (repo):
    source_path = repo["root"]
    link_path = os.path.join(lldb_source_path(), os.path.basename(source_path))
    do_symlink(source_path, link_path)

def setup_source_symlinks ():
    map(setup_source_symlink, XCODE_REPOSITORIES())

def setup_build_symlink ():
    source_path = package_build_path()
    link_path = expected_package_build_path()
    do_symlink(source_path, link_path)
    
def build_script_flags ():
    return BUILD_SCRIPT_FLAGS()[lldb_configuration()] + ["swift_install_destdir=" + expected_package_build_path_for("swift")]

def join_dicts (dict1, dict2):
    d = dict1.copy()
    d.update(dict2)
    return d

def build_script_path ():
    return os.path.join(swift_source_path(), "utils", "build-script")

def build_script_environment():
    return join_dicts(os.environ, BUILD_SCRIPT_ENVIRONMENT())

def build_llvm ():
    subprocess.check_call(["python", build_script_path()] + build_script_flags(), cwd=lldb_source_path(), env=build_script_environment())

def build_llvm_if_needed ():
    if should_build_llvm():
        setup_source_symlinks()
        build_llvm()
        setup_build_symlink()

#### MAIN LOGIC ####

all_check_out_if_needed()
build_llvm_if_needed()
write_archives_txt()

sys.exit(0)
