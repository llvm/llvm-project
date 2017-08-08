#!/usr/bin/python

"""This script is for building and running lldb in the Jenkins buildbot."""
import os
import pwd
import subprocess

print "Running lldb build test as user: ", pwd.getpwuid(os.getuid()).pw_name

# We get everything from the workspace environment variable, so if it is
# not set we're toast:
workspace = os.getenv("WORKSPACE")
if workspace is None:
    print "WORKSPACE environment variable is not set.  Exiting."
    exit(-1)

if not os.path.isdir(workspace):
    print "WORKSPACE environment variable: ", workspace, " doesn't point to an extant directory.  Exiting."
    exit(-1)

llvm_dir = os.path.join(workspace, "llvm")
if not os.path.isdir(llvm_dir):
    print "workspace doesn't contain llvm directory.  Exiting."
    exit(-1)

# Check for clang.  It should be in llvm_dir/tools/clang.  If it is just
# in the workspace, then we'll have to make the link:

clang_dir = os.path.join(llvm_dir, "tools/clang")

if not os.path.isdir(clang_dir):
    orig_clang_dir = os.path.join(workspace, "clang")
    if not os.path.isdir(orig_clang_dir):
        print "No llvm/tools/clang or clang directories in the workspace.  Exiting."
        exit(-1)

    os.symlink(orig_clang_dir, clang_dir)
    if not os.path.isdir(clang_dir):
        print "Failed to make the symbolic link from the workspace clang to llvm/tools/clang.  Exiting."
        exit(-1)

# Do the same thing for swift:
swift_dir = os.path.join(llvm_dir, "tools/swift")

if not os.path.isdir(swift_dir):
    orig_swift_dir = os.path.join(workspace, "swift")
    if not os.path.isdir(orig_swift_dir):
        print "No llvm/tools/swift or swift directories in the workspace.  Exiting."
        exit(-1)

    os.symlink(orig_swift_dir, swift_dir)
    if not os.path.isdir(swift_dir):
        print "Failed to make the symbolic link from the workspace swift to llvm/tools/swift.  Exiting."
        exit(-1)

# Okay, everything should be set up now to run the build:
lldb_dir = os.path.join(workspace, "lldb")
if not os.path.isdir(lldb_dir):
    print "No lldb directory in workspace.  Exiting."
    exit(-1)

# Symlink the llvm we've made into the lldb directory:
llvm_in_lldb_dir = os.path.join(lldb_dir, "llvm")
if not os.path.isdir(llvm_in_lldb_dir):
    os.symlink(llvm_dir, llvm_in_lldb_dir)

lldb_configuration = os.getenv("LLDB_CONFIGURATION")
if lldb_configuration is None:
    lldb_configuration = "BuildAndIntegration"

lldb_arch = "x86_64"

build_args = [
    "xcodebuild",
    "-configuration",
    lldb_configuration,
    "-arch",
    lldb_arch]
return_val = subprocess.call(build_args, cwd=lldb_dir)
if return_val != 0:
    print "Build failed, return code: ", return_val, ".  Exiting."
    exit(-1)

# FIXME: The test suite is bringing down the machine by leaking PTYs.
# test_dir = os.path.join (lldb_dir, "test")
# test_binary = os.path.join(test_dir, "dotest.py")
# test_args = [test_binary, "-A", "x86_64"]
# return_val = subprocess.call(test_args, cwd=test_dir)

# We're not checking the return value of the testsuite yet.
# if return_val != 0:
#        print "Tests failed, return code: ", return_val, ".  Exiting."
#        exit (-1)

exit(0)
