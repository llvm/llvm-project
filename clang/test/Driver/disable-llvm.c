// We support a CC1 option for disabling LLVM's passes.
// RUN: %clang -O2 -Xclang -disable-llvm-passes -### %s 2>&1 \
// RUN:     | FileCheck --check-prefix=DISABLED %s

// Try -Xclang=.
// RUN: %clang -O2 -Xclang=-disable-llvm-passes -### %s 2>&1 \
// RUN:     | FileCheck --check-prefix=DISABLED %s
// DISABLED: -cc1
// DISABLED-NOT: "-mllvm" "-disable-llvm-passes"
// DISABLED: "-disable-llvm-passes"
//
// We also support two alternative spellings for historical reasons.
// RUN: %clang -O2 -Xclang -disable-llvm-optzns -### %s 2>&1 \
// RUN:     | FileCheck --check-prefix=DISABLED-LEGACY %s
// RUN: %clang -O2 -mllvm -disable-llvm-optzns -### %s 2>&1 \
// RUN:     | FileCheck --check-prefix=DISABLED-LEGACY %s
// DISABLED-LEGACY: -cc1
// DISABLED-LEGACY-NOT: "-mllvm" "-disable-llvm-optzns"
// DISABLED-LEGACY: "-disable-llvm-optzns"
//
// The main flag shouldn't be specially handled when used with '-mllvm'.
// RUN: %clang -O2 -mllvm -disable-llvm-passes -### %s 2>&1 | FileCheck --check-prefix=MLLVM %s
// MLLVM: -cc1
// MLLVM-NOT: -disable-llvm-passes
// MLLVM: "-mllvm" "-disable-llvm-passes"
// MLLVM-NOT: -disable-llvm-passes
