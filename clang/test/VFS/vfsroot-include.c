// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: sed -e "s@TEST_DIR@%{/S:regex_replacement}@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsroot.yaml > %t.yaml
// RUN: not %clang_cc1 -Werror -ivfsoverlay %t.yaml -I %S/Inputs -I /direct-vfs-root-files -fsyntax-only /tests/vfsroot-include.c 2>&1 | FileCheck %s
// The line above tests that the compiler input file is looked up through VFS.

// Test successful include through the VFS.
#include "not_real.h"

// Test that a file missing from the VFS root is not found, even if it is
// discoverable through the real file system. Fatal error should be the last
// in the file as it hides other errors.
#include "actual_header.h"
// CHECK: fatal error: 'actual_header.h' file not found
// CHECK: 1 error generated.
// CHECK-NOT: error:
