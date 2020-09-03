// Check that we extract -MD from '-Wp,-MD,FOO', which is used by a number of
// major projects (e.g., FireFox and the Linux Kernel).

// RUN: %clang -target i386-pc-linux-gnu -### \
// RUN:   -Wp,-MD,FOO.d -fsyntax-only %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK: "-cc1"
// CHECK-NOT: -MD
// CHECK: "-dependency-file" "FOO.d"
// CHECK: "-MT"
//
// PR4062

// RUN: %clang -target i386-pc-linux-gnu -### \
// RUN:   -Wp,-MMD -fsyntax-only %s 2> %t
// RUN: FileCheck -check-prefix MMD < %t %s

// MMD: "-cc1"
// MMD-NOT: -MMD
// MMD: "-dependency-file" "Wp-args.d"

// Ensure response files are properly expanded with -Wp
// RUN: echo -DTEST > %t.rsp
// RUN: %clang -Wp,@%t.rsp -E %s | FileCheck -check-prefix RSP %s

#ifdef TEST
void foo();
#endif

// RSP: foo()
