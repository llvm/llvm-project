// === Old PM ===
// No splitting at -O0.
// RUN: %clang_cc1 -O0 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-NO-SPLIT %s
//
// No splitting at -Oz.
// RUN: %clang_cc1 -Oz -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-NO-SPLIT %s
//
// No splitting by default.
// RUN: %clang_cc1 -O3 -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-NO-SPLIT %s
//
// No splitting when it's explicitly disabled.
// RUN: %clang_cc1 -O3 -fno-split-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-NO-SPLIT %s
//
// No splitting when LLVM passes are disabled.
// RUN: %clang_cc1 -O3 -fsplit-cold-code -disable-llvm-passes -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-NO-SPLIT %s
//
// Split at -O1.
// RUN: %clang_cc1 -O1 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-SPLIT %s
//
// Split at -Os.
// RUN: %clang_cc1 -Os -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-SPLIT %s
//
// Split at -O2.
// RUN: %clang_cc1 -O2 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-SPLIT %s
//
// Split at -O3.
// RUN: %clang_cc1 -O3 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=OLDPM-SPLIT %s

// OLDPM-NO-SPLIT-NOT: Hot Cold Split

// OLDPM-SPLIT: Hot Cold Split

// === New PM (ditto) ===
// No splitting at -O0.
// RUN: %clang_cc1 -O0 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-NO-SPLIT %s
//
// No splitting at -Oz.
// RUN: %clang_cc1 -Oz -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-NO-SPLIT %s
//
// No splitting by default.
// RUN: %clang_cc1 -O3 -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-NO-SPLIT %s
//
// No splitting when it's explicitly disabled.
// RUN: %clang_cc1 -O3 -fno-split-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-NO-SPLIT %s
//
// No splitting when LLVM passes are disabled.
// RUN: %clang_cc1 -O3 -fsplit-cold-code -disable-llvm-passes -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-NO-SPLIT %s
//
// Split at -O1.
// RUN: %clang_cc1 -O1 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-SPLIT %s
//
// Split at -Os.
// RUN: %clang_cc1 -Os -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-SPLIT %s
//
// Split at -O2.
// RUN: %clang_cc1 -O2 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-SPLIT %s
//
// Split at -O3.
// RUN: %clang_cc1 -O3 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=NEWPM-SPLIT %s

// NEWPM-NO-SPLIT-NOT: HotColdSplit

// NEWPM-SPLIT: HotColdSplit
