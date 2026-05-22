// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t && mkdir %t
// RUN: cd %t
// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=repro.tar -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: mkdir extract
// RUN: tar -xf repro.tar -C extract
// RUN: FileCheck %s --check-prefix=SH < extract/*/crash-diagnostics-tar-*.sh
// RUN: FileCheck %s --check-prefix=C < extract/*/crash-diagnostics-tar-*.c

#pragma clang __debug parser_crash

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK: Crash reproducer tarball created at:

// SH: # Crash reproducer for
// C: # 1 "
