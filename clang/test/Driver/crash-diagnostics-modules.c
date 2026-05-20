// UNSUPPORTED: system-windows
// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=%t.tar -fmodules -fmodules-cache-path=%t/cache -I %S/Inputs -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: tar -tf %t.tar | FileCheck %s --check-prefix=TAR

#include "empty.h"

#pragma clang __debug parser_crash

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK: Crash reproducer tarball created at:

// TAR-DAG: {{.*}}.c
// TAR-DAG: {{.*}}.sh
// TAR-DAG: {{.*}}.cache/{{.*}}module.modulemap
