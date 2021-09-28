// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: env TMPDIR=%t TEMP=%t TMP=%t RC_DEBUG_OPTIONS=1           \
// RUN: not %clang_cl -fsyntax-only /Brepro /source-charset:utf-8 \
// RUN:     -- %s 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH %s

// REQUIRES: crash-recovery

#pragma clang __debug crash

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}crash-report-clang-cl-{{.*}}.c
// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: {{.*}}"-fsyntax-only"
// CHECKSH-SAME: /Brepro
// CHECKSH-SAME: /source-charset:utf-8
// CHECKSH-NOT: -mno-incremental-linker-compatible
// CHECKSH-NOT: -finput-charset=utf-8
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-main-file-name" "crash-report-clang-cl.c"
// CHECKSH: "crash-report-{{[^ ]*}}.c"
