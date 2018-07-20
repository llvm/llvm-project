// REQUIRES: system-windows

// We used to require adding ".exe" suffix when cross-compiling on Windows.
// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/fuse_ld_windows -fuse-ld=foo 2>&1 \
// RUN:   | FileCheck %s

// Check that the old variant still works.
// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -B %S/Inputs/fuse_ld_windows -fuse-ld=foo.exe 2>&1 \
// RUN:   | FileCheck %s

// With the full path, the extension can be omitted, too,
// because Windows allows that.
// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -fuse-ld=%S/Inputs/fuse_ld_windows/ld.foo 2>&1 \
// RUN:   | FileCheck %s

// Check that the full path with the extension works too.
// RUN: %clang %s -### -o %t.o -target i386-unknown-linux \
// RUN:     -fuse-ld=%S/Inputs/fuse_ld_windows/ld.foo.exe 2>&1 \
// RUN:   | FileCheck %s

// CHECK-NOT: invalid linker name
// CHECK: /Inputs/fuse_ld_windows{{/|\\\\}}ld.foo
