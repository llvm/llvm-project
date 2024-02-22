// REQUIRES: clang-cc1daemon, ansi-escape-sequences

// RUN: rm -rf %t && mkdir -p %t

// RUN: not %clang -cc1depscan -fdepscan=inline -cc1-args -cc1 -triple x86_64-apple-macos11 -x c %s -o %t/t.o -fcas-path %t/cas \
// RUN:   2>&1 | FileCheck %s -check-prefix=ERROR

// Using normal compilation as baseline.
// RUN: not %clang -target x86_64-apple-macos11 -c %s -o %t.o -Wl,-none --serialize-diagnostics %t/t1.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=ERROR -check-prefix=DRIVER
// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t.o -Wl,-none --serialize-diagnostics %t/t2.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=ERROR -check-prefix=DRIVER
// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t -cas-args -fcas-path %t/cas -- \
// RUN: %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t.o -Wl,-none --serialize-diagnostics %t/t3.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=ERROR -check-prefix=DRIVER

// RUN: diff %t/t1.diag %t/t2.diag
// RUN: diff %t/t1.diag %t/t3.diag

// DRIVER: warning: -Wl,-none: 'linker' input unused
// ERROR: error: 'non-existent.h' file not found
// ERROR: 1 error generated.

// Make sure successful compilation clears the diagnostic file.
// RUN: echo "int x;" > %t/a.c
// RUN: echo "int y;" > %t/b.c
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %t/a.c -o %t.o --serialize-diagnostics %t/t2.diag
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t -cas-args -fcas-path %t/cas -- \
// RUN: %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %t/b.c -o %t.o --serialize-diagnostics %t/t3.diag

// RUN: c-index-test -read-diagnostics %t/t2.diag 2>&1 | FileCheck %s -check-prefix=SERIAL
// RUN: c-index-test -read-diagnostics %t/t3.diag 2>&1 | FileCheck %s -check-prefix=SERIAL
// SERIAL: Number of diagnostics: 0

// Make sure warnings are still emitted for normal compilation.
// RUN: echo "#warning some warning" > %t/warn.c
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %t/warn.c -o %t.o \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// WARN: warning: some warning

// Make sure diagnostics emitted during CAS dep-scanning respect the color settings.
// RUN: not %clang -target x86_64-apple-macos11 -c %s -o %t.o -fdiagnostics-color=always -fansi-escape-codes \
// RUN:   2>&1 | FileCheck %s -check-prefix=COLOR-DIAG
// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t.o -fdiagnostics-color=always -fansi-escape-codes \
// RUN:   2>&1 | FileCheck %s -check-prefix=COLOR-DIAG
// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t -cas-args -fcas-path %t/cas -- \
// RUN: %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t.o -fdiagnostics-color=always -fansi-escape-codes \
// RUN:   2>&1 | FileCheck %s -check-prefix=COLOR-DIAG
// COLOR-DIAG: [[RED:.\[0;1;31m]]fatal error: [[RESET:.\[0m]]

#include "non-existent.h"
