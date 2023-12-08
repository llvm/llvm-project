// REQUIRES: x86-registered-target

// System directory and sysroot option causes warning.
// RUN: %clang -Wpoison-system-directories -target x86_64 -I/usr/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s 2> %t.1.stderr
// RUN: FileCheck -check-prefix=WARN < %t.1.stderr %s
// RUN: %clang -Wpoison-system-directories -target x86_64 -cxx-isystem/usr/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s 2> %t.1.stderr
// RUN: FileCheck -check-prefix=WARN < %t.1.stderr %s
// RUN: %clang -Wpoison-system-directories -target x86_64 -iquote/usr/local/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s 2> %t.1.stderr
// RUN: FileCheck -check-prefix=WARN < %t.1.stderr %s
// RUN: %clang -Wpoison-system-directories -target x86_64 -isystem/usr/local/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s 2> %t.1.stderr
// RUN: FileCheck -check-prefix=WARN < %t.1.stderr %s

// Missing target but included sysroot still causes the warning.
// RUN: %clang -Wpoison-system-directories -I/usr/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s 2> %t.2.stderr
// RUN: FileCheck -check-prefix=WARN < %t.2.stderr %s

// With -Werror the warning causes the failure.
// RUN: not %clang -Werror=poison-system-directories -target x86_64 -I/usr/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s 2> %t.3.stderr
// RUN: FileCheck -check-prefix=ERROR < %t.3.stderr %s

// Cros target without sysroot causes no warning.
// RUN: %clang -Wpoison-system-directories -Werror -target x86_64 -I/usr/include -c -o - %s

// By default the warning is off.
// RUN: %clang -Werror -target x86_64 -I/usr/include --sysroot %S/Inputs/sysroot_x86_64_cross_linux_tree -c -o - %s

// WARN: warning: include location {{[^ ]+}} is unsafe for cross-compilation [-Wpoison-system-directories]

// ERROR: error: include location {{[^ ]+}} is unsafe for cross-compilation [-Werror,-Wpoison-system-directories]
