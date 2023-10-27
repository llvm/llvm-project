// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: not %clang_cl -fsyntax-only /Brepro /source-charset:utf-8 \
// RUN:     -fcrash-diagnostics-dir=%t -- %s 2>&1 | FileCheck %s
// RUN: cat %t/crash-report-clang-cl-*.cpp | FileCheck --check-prefix=CHECKSRC %s
// RUN: cat %t/crash-report-clang-cl-*.sh | FileCheck --check-prefix=CHECKSH %s

// REQUIRES: crash-recovery

#pragma clang __debug crash

// CHECK: Preprocessed source(s) and associated run script(s) are located at:

// __has_feature(cxx_exceptions) is default-off in the cl-compatible driver.
FOO
#if __has_feature(cxx_exceptions)
int a = 1;
#else
int a = 0;
#endif
// CHECKSRC:      {{^}}FOO
// CHECKSRC-NEXT: {{^}}#if 0 /* disabled by -frewrite-includes */
// CHECKSRC-NEXT: {{^}}#if __has_feature(cxx_exceptions)
// CHECKSRC-NEXT: {{^}}#endif
// CHECKSRC-NEXT: {{^}}#endif /* disabled by -frewrite-includes */
// CHECKSRC-NEXT: {{^}}#if 0 /* evaluated by -frewrite-includes */
// CHECKSRC-NEXT: {{^}}#
// CHECKSRC-NEXT: {{^}}int a = 1;
// CHECKSRC-NEXT: {{^}}#else
// CHECKSRC-NEXT: {{^}}#
// CHECKSRC-NEXT: {{^}}int a = 0;
// CHECKSRC-NEXT: {{^}}#endif

// CHECK-NEXT: note: diagnostic msg: {{.*}}crash-report-clang-cl-{{.*}}.cpp
// CHECKSH: # Crash reproducer
// CHECKSH-NEXT: # Driver args: {{.*}}"-fsyntax-only"
// CHECKSH-SAME: /Brepro
// CHECKSH-SAME: /source-charset:utf-8
// CHECKSH-NOT: -mno-incremental-linker-compatible
// CHECKSH-NOT: -finput-charset=utf-8
// CHECKSH-NEXT: # Original command: {{.*$}}
// CHECKSH-NEXT: "-cc1"
// CHECKSH: "-main-file-name" "crash-report-clang-cl.cpp"
// CHECKSH: "crash-report-{{[^ ]*}}.cpp"
