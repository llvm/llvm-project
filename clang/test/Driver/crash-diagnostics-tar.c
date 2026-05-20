// UNSUPPORTED: system-windows
// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t.tar
// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=%t.tar -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: tar -xf %t.tar -C %t
// RUN: FileCheck %s --check-prefix=SH < %t/*/crash-diagnostics-tar-*.sh
// RUN: FileCheck %s --check-prefix=C < %t/*/crash-diagnostics-tar-*.c

#pragma clang __debug parser_crash

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK: Crash reproducer tarball created at:

// SH: # Crash reproducer for
// C: # 1 "
