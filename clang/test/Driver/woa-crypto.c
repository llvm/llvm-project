// RUN: %clang -target arm-windows-msvc                       -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target arm-windows-msvc -march=armv8-a+crypto -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-CRYPTO

// CHECK-DEFAULT: "-target-cpu" "cortex-a9"
// CHECK-CRYPTO: "-target-cpu" "generic"
// CHECK-CRYPTO: "-target-feature" "+crypto"
