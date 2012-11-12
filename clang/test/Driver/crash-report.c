// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env TMPDIR=%t TEMP=%t TMP=%t RC_DEBUG_OPTIONS=1 %clang -fsyntax-only %s \
// RUN:  -F/tmp/ -I /tmp/ -idirafter /tmp/ -iquote /tmp/ -isystem /tmp/ \
// RUN:  -iprefix /the/prefix -iwithprefix /tmp -iwithprefixbefore /tmp/ \
// RUN:  -internal-isystem /tmp/ -internal-externc-isystem /tmp/ \
// RUN:  -DFOO=BAR 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-*.c | FileCheck --check-prefix=CHECKSRC %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH %s
// REQUIRES: crash-recovery

#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}.c
FOO
// CHECKSRC: FOO
// CHECKSH: -D "FOO=BAR"
// CHECKSH-NOT: -F/tmp/
// CHECKSH-NOT: -I /tmp/
// CHECKSH-NOT: -idirafter /tmp/
// CHECKSH-NOT: -iquote /tmp/
// CHECKSH-NOT: -isystem /tmp/
// CHECKSH-NOT: -iprefix /the/prefix
// CHECKSH-NOT: -iwithprefix /tmp/
// CHECKSH-NOT: -iwithprefixbefore /tmp/
// CHECKSH-NOT: -internal-isystem /tmp/
// CHECKSH-NOT: -internal-externc-isystem /tmp/
// CHECKSH-NOT: -dwarf-debug-flags
