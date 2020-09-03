/// Test that -malign-branch* and -mbranches-within-32B-boundaries are handled for assembly files.

// RUN: %clang -target x86_64 -malign-branch-boundary=16 %s -c -### 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// BOUNDARY: "-mllvm" "-x86-align-branch-boundary=16"

// RUN: %clang -target x86_64 -malign-branch=fused,jcc,jmp %s -c -### %s 2>&1 | FileCheck %s --check-prefix=TYPE
// TYPE: "-mllvm" "-x86-align-branch=fused+jcc+jmp"

// RUN: %clang -target x86_64 -mpad-max-prefix-size=5 %s -c -### 2>&1 | FileCheck %s --check-prefix=PREFIX
// PREFIX: "-mllvm" "-x86-pad-max-prefix-size=5"

// RUN: %clang -target x86_64 -mbranches-within-32B-boundaries %s -c -### 2>&1 | FileCheck %s --check-prefix=32B
// 32B: "-mllvm" "-x86-branches-within-32B-boundaries"
