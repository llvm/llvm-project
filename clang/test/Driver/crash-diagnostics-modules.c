// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t && mkdir %t
// RUN: cd %t
// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=repro.tar -fmodules -fmodules-cache-path=cache -I %S/Inputs -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: tar -tf repro.tar | FileCheck %s --check-prefix=TAR

#include "empty.h"

#pragma clang __debug parser_crash

// CHECK: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
// CHECK: repro.tar

// TAR-DAG: .c
// TAR-DAG: .sh
// TAR-DAG: .cache/{{.*}}module.modulemap
