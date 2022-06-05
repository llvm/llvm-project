// REQUIRES: crash-recovery, system-darwin
// RUN: rm -rf %t
// RUN: mkdir -p %t/i %t/m %t

// RUN: env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: not %clang -fsyntax-only %s \
// RUN:   -I %S/Inputs/module -isysroot %/t/i/ \
// RUN:   -fmodules -fmodules-cache-path=%t/m/ -DFOO=BAR 2>&1 | \
// RUN:   FileCheck -check-prefix=CRASH_ENV %s

// RUN: env TMPDIR=%t TEMP=%t TMP=%t \
// RUN: not %clang -gen-reproducer -fsyntax-only %s \
// RUN:   -I %S/Inputs/module -isysroot %/t/i/ \
// RUN:   -fmodules -fmodules-cache-path=%t/m/ -DFOO=BAR 2>&1 | \
// RUN:   FileCheck -check-prefix=CRASH_FLAG %s

@import simple;
const int x = MODULE_MACRO;

// CRASH_ENV: PLEASE submit a bug report to {{.*}} and include the crash backtrace, preprocessed source, and associated run script.
// CRASH_ENV: failing because environment variable 'FORCE_CLANG_DIAGNOSTICS_CRASH' is set
// CRASH_ENV: Preprocessed source(s) and associated run script(s) are located at:
// CRASH_ENV-NEXT: note: diagnostic msg: {{.*}}.m
// CRASH_ENV-NEXT: note: diagnostic msg: {{.*}}.cache
// CRASH_ENV-NEXT: note: diagnostic msg: {{.*}}.sh
// CRASH_ENV-NEXT: note: diagnostic msg: Crash backtrace is located in
// CRASH_ENV-NEXT: note: diagnostic msg: {{.*}}Library/Logs/DiagnosticReports{{.*}}

// CRASH_FLAG: failing because '-gen-reproducer' is used
// CRASH_FLAG: Preprocessed source(s) and associated run script(s) are located at:
// CRASH_FLAG-NEXT: note: diagnostic msg: {{.*}}.m
// CRASH_FLAG-NEXT: note: diagnostic msg: {{.*}}.cache
// CRASH_FLAG-NEXT: note: diagnostic msg: {{.*}}.sh
// CRASH_FLAG-NEXT: note: diagnostic msg: Crash backtrace is located in
// CRASH_FLAG-NEXT: note: diagnostic msg: {{.*}}Library/Logs/DiagnosticReports{{.*}}
