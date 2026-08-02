// Check LFI predefined macros.

// RUN: %clang --target=aarch64_lfi-linux -E -dM %s -o - | FileCheck %s --implicit-check-not=__LFI_NO
// RUN: %clang --target=aarch64_lfi-linux -mlfi=no-loads,no-stores -E -dM %s -o - | FileCheck %s --check-prefix=CONFIG
// RUN: %clang --target=aarch64-linux -E -dM %s -o - | FileCheck %s --check-prefix=OFF --implicit-check-not=__LFI

// CHECK: #define __LFI__ 1

// CONFIG-DAG: #define __LFI__ 1
// CONFIG-DAG: #define __LFI_NO_LOADS__ 1
// CONFIG-DAG: #define __LFI_NO_STORES__ 1

// OFF: #define __aarch64__ 1
