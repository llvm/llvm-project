// RUN: rm -rf %t
// RUN: mkdir %t

// Verify that crash diagnostics are generated for just the crashing arch when
// multiple -arch options are present, rather than bailing out entirely.

// RUN: env CLANG_CRASH_DIAGNOSTICS_DIR=%t \
// RUN:   not %crash_opt %clang %s -arch arm64 -arch arm64e -arch x86_64 -fsyntax-only -DCRASH_x86_64 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH-x86_64 %s
// RUN: rm -f %t/crash-report-*

// RUN: env CLANG_CRASH_DIAGNOSTICS_DIR=%t \
// RUN:   not %crash_opt %clang %s -arch arm64 -arch arm64e -arch x86_64 -fsyntax-only -DCRASH_arm64 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH-arm64 %s
// RUN: rm -f %t/crash-report-*

// RUN: env CLANG_CRASH_DIAGNOSTICS_DIR=%t \
// RUN:   not %crash_opt %clang %s -arch arm64 -arch arm64e -arch x86_64 -fsyntax-only -DCRASH_arm64e 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH-arm64e %s
// RUN: rm -f %t/crash-report-*

// If multiple arches crash, capture the reproducer for the first one.
// RUN: env CLANG_CRASH_DIAGNOSTICS_DIR=%t \
// RUN:   not %crash_opt %clang %s -arch arm64 -arch arm64e -arch x86_64 -fsyntax-only -DCRASH_arm64 -DCRASH_arm64e 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH-arm64 %s
// RUN: rm -f %t/crash-report-*

// Likewise for -gen-reproducer, the reproducer script should target the first arch.
// RUN: env CLANG_CRASH_DIAGNOSTICS_DIR=%t \
// RUN:   not %clang %s -arch arm64 -arch arm64e -fsyntax-only -gen-reproducer 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK %s
// RUN: cat %t/crash-report-*.sh | FileCheck --check-prefix=CHECKSH-arm64 %s

// REQUIRES: crash-recovery, system-darwin
// REQUIRES: aarch64-registered-target, x86-registered-target

// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: note: diagnostic msg: {{.*}}crash-report-{{.*}}.c

// CHECKSH-x86_64: # Crash reproducer
// CHECKSH-x86_64: "-cc1"
// CHECKSH-x86_64: "-triple" "x86_64{{[^"]*}}"
// CHECKSH-x86_64-NOT: "-triple" "arm64{{[^"]*}}"
// CHECKSH-x86_64-NOT: "-arch" "arm64"
// CHECKSH-x86_64-NOT: "-arch" "arm64e"

// CHECKSH-arm64: # Crash reproducer
// CHECKSH-arm64: "-cc1"
// CHECKSH-arm64: "-triple" "arm64-{{[^"]*}}"
// CHECKSH-arm64-NOT: "-triple" "arm64e{{[^"]*}}"
// CHECKSH-arm64-NOT: "-triple" "x86_64{{[^"]*}}"
// CHECKSH-arm64-NOT: "-arch" "arm64e"
// CHECKSH-arm64-NOT: "-arch" "x86_64"

// CHECKSH-arm64e: # Crash reproducer
// CHECKSH-arm64e: "-cc1"
// CHECKSH-arm64e: "-triple" "arm64e{{[^"]*}}"
// CHECKSH-arm64e-NOT: "-triple" "arm64-{{[^"]*}}"
// CHECKSH-arm64e-NOT: "-triple" "x86_64{{[^"]*}}"
// CHECKSH-arm64e-NOT: "-arch" "arm64"
// CHECKSH-arm64e-NOT: "-arch" "x86_64"

#ifdef CRASH_x86_64
# ifdef __x86_64__
#  pragma clang __debug crash
# endif
#endif

#ifdef CRASH_arm64
# if defined(__aarch64__) && !defined(__arm64e__)
#  pragma clang __debug crash
# endif
#endif

#ifdef CRASH_arm64e
# ifdef __arm64e__
#  pragma clang __debug crash
# endif
#endif

int x;
