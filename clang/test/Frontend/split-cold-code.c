// RUN: %clang_cc1 -O0 -fsplit-cold-code %s 2>&1 | FileCheck %s --check-prefix=O0
// O0: warning: '-fsplit-cold-code' has no effect when optimizations are disabled

// RUN: %clang_cc1 -Oz -fsplit-cold-code %s 2>&1 | FileCheck %s --check-prefix=Oz
// Oz: warning: '-fsplit-cold-code' has no effect when optimizing for minimum size
