// REQUIRES: shell
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: sed -e "s:TEST_DIR:%S:g" -e "s:OUT_DIR:%t:g" %S/Inputs/vfsroot.yaml > %t.yaml
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/cache -ivfsoverlay %t.yaml -F %S/Inputs -fsyntax-only /tests/vfsroot-module.m

// Test that a file missing from the VFS root is not found, even if it is
// discoverable through the real file system at location that is a part of
// the framework.
@import Broken;
