// REQUIRES: crash-recovery, shell, system-darwin

// RUN: rm -rf %t
// RUN: mkdir -p %t/m

// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only %s -index-store-path %t/index \
// RUN:     -fmodules -fmodules-cache-path=%t/m/ 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECKMOD %s
// RUN: FileCheck --check-prefix=CHECKMOD_SH %s -input-file %t/crash-index-*.sh

// RUN: rm -rf %t
// RUN: not env FORCE_CLANG_DIAGNOSTICS_CRASH= TMPDIR=%t TEMP=%t TMP=%t \
// RUN: %clang -fsyntax-only %s -index-store-path %t/index \
// RUN:     2>&1 | FileCheck --check-prefix=CHECK %s
// RUN: FileCheck --check-prefix=CHECKSH %s -input-file %t/crash-index-*.sh
int foo() { return 0; }

// CHECKMOD: Preprocessed source(s) and associated run script(s) are located at:
// CHECKMOD-NEXT: note: diagnostic msg: {{.*}}.m
// CHECKMOD-NEXT: note: diagnostic msg: {{.*}}.cache

// CHECKMOD_SH: # Crash reproducer
// CHECKMOD_SH-NEXT: # Driver args: "-fsyntax-only"
// CHECKMOD_SH-NEXT: # Original command: {{.*$}}
// CHECKMOD_SH-NEXT: "-cc1"
// CHECKMOD_SH: "crash-index-{{[^ ]*}}.m"
// CHECKMOD_SH: "-index-store-path" "crash-index-{{[^ ]*}}.cache/index-store"

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.m

// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: "-fsyntax-only"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "crash-index-{{[^ ]*}}.m"
// CHECKSH: "-index-store-path" "index-store"
