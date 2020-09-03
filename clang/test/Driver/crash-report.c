// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: echo '-fsyntax-only                                              \
// RUN:  -F/tmp/ -I /tmp/ -idirafter /tmp/ -iquote /tmp/ -isystem /tmp/  \
// RUN:  -iprefix /the/prefix -iwithprefix /tmp -iwithprefixbefore /tmp/ \
// RUN:  -Xclang -internal-isystem -Xclang /tmp/                         \
// RUN:  -Xclang -internal-externc-isystem -Xclang /tmp/                 \
// RUN:  -Xclang -main-file-name -Xclang foo.c                           \
// RUN:  -DFOO=BAR -DBAR="BAZ QUX"' > %t.rsp

// RUN: env TMPDIR=%t TEMP=%t TMP=%t RC_DEBUG_OPTIONS=1                  \
// RUN:  CC_PRINT_HEADERS=1 CC_LOG_DIAGNOSTICS=1                         \
// RUN:  not %clang %s @%t.rsp -DPARSER 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-*.c | FileCheck --check-prefix=CHECKSRC %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH %s

// RUN: env TMPDIR=%t TEMP=%t TMP=%t RC_DEBUG_OPTIONS=1                  \
// RUN:  CC_PRINT_HEADERS=1 CC_LOG_DIAGNOSTICS=1                         \
// RUN:  not %clang %s @%t.rsp -DCRASH 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-*.c | FileCheck --check-prefix=CHECKSRC %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH %s

// RUN: env TMPDIR=%t TEMP=%t TMP=%t RC_DEBUG_OPTIONS=1                  \
// RUN:  CC_PRINT_HEADERS=1 CC_LOG_DIAGNOSTICS=1                         \
// RUN:  not %clang %s @%t.rsp -DFATAL 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-*.c | FileCheck --check-prefix=CHECKSRC %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH %s

// REQUIRES: crash-recovery

#ifdef PARSER
#pragma clang __debug parser_crash
#elif CRASH
#pragma clang __debug crash
#elif FATAL
#pragma clang __debug llvm_fatal_error
#endif

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}crash-report-{{.*}}.c
FOO
// CHECKSRC: FOO
// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: {{.*}}"-fsyntax-only"
// CHECKSH-SAME: "-D" "FOO=BAR"
// CHECKSH-SAME: "-D" "BAR=BAZ QUX"
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-main-file-name" "crash-report.c"
// CHECKSH-NOT: "-header-include-file"
// CHECKSH-NOT: "-diagnostic-log-file"
// CHECKSH: "-D" "FOO=BAR"
// CHECKSH: "-D" "BAR=BAZ QUX"
// CHECKSH-NOT: "-F/tmp/"
// CHECKSH-NOT: "-I" "/tmp/"
// CHECKSH-NOT: "-idirafter" "/tmp/"
// CHECKSH-NOT: "-iquote" "/tmp/"
// CHECKSH-NOT: "-isystem" "/tmp/"
// CHECKSH-NOT: "-iprefix" "/the/prefix"
// CHECKSH-NOT: "-iwithprefix" "/tmp/"
// CHECKSH-NOT: "-iwithprefixbefore" "/tmp/"
// CHECKSH-NOT: "-internal-isystem" "/tmp/"
// CHECKSH-NOT: "-internal-externc-isystem" "/tmp/"
// CHECKSH-NOT: "-dwarf-debug-flags"
// CHECKSH: "crash-report-{{[^ ]*}}.c"
