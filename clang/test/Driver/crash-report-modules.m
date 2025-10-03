// RUN: rm -rf %t/crmdir
// RUN: mkdir -p %t/crmdir/i %t/crmdir/m

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t/crmdir TEMP=%t/crmdir TMP=%t/crmdir \
// RUN: not %clang -fsyntax-only %s -I %S/Inputs/module -isysroot %/t/i/                 \
// RUN: -fmodules -fmodules-cache-path=%t/crmdir/m/ -DFOO=BAR 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSRC %s -input-file %t/crmdir/crash-report-*.m
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crmdir/crash-report-*.sh
// REQUIRES: crash-recovery

// FIXME: This test creates excessively deep directory hierarchies that cause
// problems on Windows.
// UNSUPPORTED: system-windows

@import simple;
const int x = MODULE_MACRO;

// CHECK: PLEASE submit a bug report to {{.*}} and include the crash backtrace, preprocessed source, and associated run script.
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m
// CHECK-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKSRC: @import simple;
// CHECKSRC: const int x = 10;

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-SAME: "-D" "FOO=BAR"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-isysroot" "{{[^"]*}}/i/"
// CHECKSH: "-D" "FOO=BAR"
// CHECKSH-NOT: "-fmodules-cache-path="
// CHECKSH: "crash-report-modules-{{[^ ]*}}.m"
// CHECKSH: "-ivfsoverlay" "crash-report-modules-{{[^ ]*}}.cache{{(/|\\\\)}}vfs{{(/|\\\\)}}vfs.yaml"
