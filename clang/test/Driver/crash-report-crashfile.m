// REQUIRES: crash-recovery, shell, system-darwin
// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only %s -I %S/Inputs/module -isysroot %/t/i/    \
// RUN: -fmodules -fmodules-cache-path=%t/m/ -DFOO=BAR 2>&1 | FileCheck %s

@import simple;
const int x = MODULE_MACRO;

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache
// CHECK-NEXT: note: diagnostic msg: {{.*}}.sh
// CHECK-NEXT: note: diagnostic msg: Crash backtrace is located in
// CHECK-NEXT: note: diagnostic msg: {{.*}}Library/Logs/DiagnosticReports{{.*}}
