// RUN: rm -rf "%t"
// RUN: mkdir "%t"
// RUN: cp "%s" "%t/crash report spaces.c"
// RUN: env TMPDIR="%t" TEMP="%t" TMP="%t" RC_DEBUG_OPTIONS=1 not %clang -fsyntax-only "%t/crash report spaces.c" 2>&1 | FileCheck "%s"
// RUN: cat "%t/crash report spaces"-*.c | FileCheck --check-prefix=CHECKSRC "%s"
// RUN: cat "%t/crash report spaces"-*.sh | FileCheck --check-prefix=CHECKSH "%s"
// REQUIRES: crash-recovery

#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.c
FOO
// CHECKSRC: FOO
// CHECKSH: "-cc1"
// CHECKSH: "-main-file-name" "crash report spaces.c"
// CHECKSH: "crash report spaces-{{[^ ]*}}.c"
