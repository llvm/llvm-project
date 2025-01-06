// RUN: %clang --target=i386 %s -fpatchable-function-entry=1 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=x86_64 %s -fpatchable-function-entry=1 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64 %s -fpatchable-function-entry=1 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64 %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=loongarch32 %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=loongarch64 %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32 %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64 %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc-unknown-linux-gnu %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc64-unknown-linux-gnu %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// CHECK: "-fpatchable-function-entry=1"

// RUN: %clang --target=aarch64 -fsyntax-only %s -fpatchable-function-entry=1,1 -c -### 2>&1 | FileCheck --check-prefix=11 %s
// 11: "-fpatchable-function-entry=1" "-fpatchable-function-entry-offset=1"
// RUN: %clang --target=aarch64 -fsyntax-only %s -fpatchable-function-entry=2,1 -c -### 2>&1 | FileCheck --check-prefix=21 %s
// 21: "-fpatchable-function-entry=2" "-fpatchable-function-entry-offset=1"

// RUN: not %clang --target=powerpc64-ibm-aix-xcoff -fsyntax-only %s -fpatchable-function-entry=1 2>&1 | FileCheck --check-prefix=AIX64 %s
// AIX64: error: unsupported option '-fpatchable-function-entry=1' for target 'powerpc64-ibm-aix-xcoff'

// RUN: not %clang --target=powerpc-ibm-aix-xcoff -fsyntax-only %s -fpatchable-function-entry=1 2>&1 | FileCheck --check-prefix=AIX32 %s
// AIX32: error: unsupported option '-fpatchable-function-entry=1' for target 'powerpc-ibm-aix-xcoff'

// RUN: not %clang --target=x86_64 -fsyntax-only %s -fpatchable-function-entry=1,0, 2>&1 | FileCheck --check-prefix=EXCESS %s
// EXCESS: error: invalid argument '1,0,' to -fpatchable-function-entry=

// RUN: not %clang --target=aarch64-linux -fsyntax-only %s -fxray-instrument -fpatchable-function-entry=1 2>&1 | FileCheck --check-prefix=XRAY %s
// XRAY: error: invalid argument '-fxray-instrument' not allowed with '-fpatchable-function-entry='
