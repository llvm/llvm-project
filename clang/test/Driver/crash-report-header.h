// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env TMPDIR="%t" TEMP="%t" TMP="%t" RC_DEBUG_OPTIONS=1 not %clang -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-header-*.h | FileCheck --check-prefix=CHECKSRC "%s"
// RUN: cat %t/crash-report-header-*.sh | FileCheck --check-prefix=CHECKSH "%s"
// REQUIRES: crash-recovery

#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.h
FOO
// CHECKSRC: FOO
// CHECKSH: "-cc1"
// CHECKSH: "-main-file-name" "crash-report-header.h"
// CHECKSH: "crash-report-header-{{[^ ]*}}.h"
