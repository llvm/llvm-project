/// Test that -malign-branch* and -mbranches-within-32B-boundaries are parsed and converted to MC options.

/// Test -malign-branch-boundary=
// RUN: %clang -target x86_64 -malign-branch-boundary=16 %s -c -### 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// BOUNDARY: "-mllvm" "-x86-align-branch-boundary=16"
// RUN: %clang -target x86_64-unknown-linux -malign-branch-boundary=16 -flto %s -### 2>&1 | FileCheck %s --check-prefix=BOUNDARY-LTO
// BOUNDARY-LTO: "-plugin-opt=-x86-align-branch-boundary=16"

// RUN: not %clang -target x86_64 -malign-branch-boundary=8 %s -c -### 2>&1 | FileCheck %s --check-prefix=BOUNDARY-ERR
// RUN: not %clang -target x86_64 -malign-branch-boundary=15 %s -c -### 2>&1 | FileCheck %s --check-prefix=BOUNDARY-ERR
// BOUNDARY-ERR: invalid argument {{.*}} to -malign-branch-boundary=

/// Test -malign-branch=
// RUN: %clang -target x86_64 -malign-branch=fused,jcc,jmp %s -c -### %s 2>&1 | FileCheck %s --check-prefix=TYPE0
// TYPE0: "-mllvm" "-x86-align-branch=fused+jcc+jmp"
// RUN: %clang -target x86_64 -malign-branch=fused,jcc,jmp,ret,call,indirect %s -c -### %s 2>&1 | FileCheck %s --check-prefix=TYPE1
// TYPE1: "-mllvm" "-x86-align-branch=fused+jcc+jmp+ret+call+indirect"
// RUN: %clang -target x86_64-unknown-linux -malign-branch=fused,jcc,jmp -flto %s -### %s 2>&1 | FileCheck %s --check-prefix=TYPE0-LTO
// TYPE0-LTO: "-plugin-opt=-x86-align-branch=fused+jcc+jmp"

// RUN: not %clang --target=x86_64 -malign-branch=fused,foo,bar %s -c -### %s 2>&1 | FileCheck %s --check-prefix=TYPE-ERR
// TYPE-ERR: invalid argument 'foo' to -malign-branch=; each element must be one of: fused, jcc, jmp, call, ret, indirect
// TYPE-ERR: invalid argument 'bar' to -malign-branch=; each element must be one of: fused, jcc, jmp, call, ret, indirect

/// Test -mpad-max-prefix-size=
// RUN: %clang -target x86_64 -mpad-max-prefix-size=0 %s -c -### 2>&1 | FileCheck %s --check-prefix=PREFIX-0
// PREFIX-0: "-mllvm" "-x86-pad-max-prefix-size=0"
// RUN: %clang -target x86_64 -mpad-max-prefix-size=15 %s -c -### 2>&1 | FileCheck %s --check-prefix=PREFIX-15
// PREFIX-15: "-mllvm" "-x86-pad-max-prefix-size=15"
// RUN: %clang -target x86_64-unknown-linux -mpad-max-prefix-size=0 -flto %s -### 2>&1 | FileCheck %s --check-prefix=PREFIX-0-LTO
// PREFIX-0-LTO: "-plugin-opt=-x86-pad-max-prefix-size=0"

/// Test -mbranches-within-32B-boundaries
// RUN: %clang -target x86_64 -mbranches-within-32B-boundaries %s -c -### 2>&1 | FileCheck %s --check-prefix=32B
// 32B: "-mllvm" "-x86-branches-within-32B-boundaries"
// RUN: %clang -target x86_64-unknown-linux -mbranches-within-32B-boundaries -flto %s -### 2>&1 | FileCheck %s --check-prefix=32B-LTO
// 32B-LTO: "-plugin-opt=-x86-branches-within-32B-boundaries"

/// Unsupported on other targets.
// RUN: not %clang -target aarch64 -malign-branch=jmp %s -c -### 2>&1 | FileCheck --check-prefix=UNUSED %s
// RUN: not %clang -target aarch64 -malign-branch-boundary=7 %s -c -### 2>&1 | FileCheck --check-prefix=UNUSED %s
// RUN: not %clang -target aarch64 -mpad-max-prefix-size=15 %s -c -### 2>&1 | FileCheck --check-prefix=UNUSED %s
// RUN: not %clang -target aarch64 -mbranches-within-32B-boundaries %s -c -### 2>&1 | FileCheck --check-prefix=UNUSED %s
// UNUSED: error: unsupported option '{{.*}}' for target '{{.*}}'
