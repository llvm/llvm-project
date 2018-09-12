// FIXME: Instead of %T/crmdir, it would be nice to just use %t, but the
// filename ran into path length limits for the rm command on some Windows
// bots.
// RUN: rm -rf %T/crmdir
// RUN: mkdir -p %T/crmdir/i %T/crmdir/m

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%T/crmdir TEMP=%T/crmdir TMP=%T/crmdir \
// RUN: not %clang -fsyntax-only %s -I %S/Inputs/module -isysroot %/t/i/                 \
// RUN: -fmodules -fmodules-cache-path=%T/crmdir/m/ -DFOO=BAR 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=CHECKSRC %s -input-file %T/crmdir/crash-report-*.m
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %T/crmdir/crash-report-*.sh
// REQUIRES: crash-recovery

// FIXME: This test creates excessively deep directory hierarchies that cause
// problems on Windows.
// UNSUPPORTED: system-windows

@import simple;
const int x = MODULE_MACRO;

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
