// === Old PM ===
// No splitting at -O0.
// RUN: %clang_cc1 -O0 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// No splitting at -Oz.
// RUN: %clang_cc1 -Oz -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// No splitting by default, even at -O3.
// RUN: %clang_cc1 -O3 -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// No splitting when it's explicitly disabled.
// RUN: %clang_cc1 -O3 -fno-split-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// Split at -O1.
// RUN: %clang_cc1 -O1 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s
//
// Split at -Os.
// RUN: %clang_cc1 -Os -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s
//
// Split at -O2.
// RUN: %clang_cc1 -O2 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s
//
// Split at -O3.
// RUN: %clang_cc1 -O3 -fsplit-cold-code -mllvm -debug-pass=Structure \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s

// === New PM (ditto) ===
// No splitting at -O0.
// RUN: %clang_cc1 -O0 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// No splitting at -Oz.
// RUN: %clang_cc1 -Oz -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// No splitting by default, even at -O3.
// RUN: %clang_cc1 -O3 -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// No splitting when it's explicitly disabled.
// RUN: %clang_cc1 -O3 -fno-split-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=NO-SPLIT %s
//
// Split at -O1.
// RUN: %clang_cc1 -O1 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s
//
// Split at -Os.
// RUN: %clang_cc1 -Os -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s
//
// Split at -O2.
// RUN: %clang_cc1 -O2 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s
//
// Split at -O3.
// RUN: %clang_cc1 -O3 -fsplit-cold-code -fexperimental-new-pass-manager -fdebug-pass-manager \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=SPLIT %s

// NO-SPLIT-NOT: "hot-cold-split"

// SPLIT: define {{.*}} @foo() [[ATTR:#[0-9]+]]
// SPLIT: attributes [[ATTR]] = { {{.*}} "hot-cold-split"

__attribute__((used))
void foo() {}
