// RUN: %clang -target x86_64-apple-macosx -fembed-bitcode=all -c %s -o /dev/null -### 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-X64 %s

// CHECK-X64: "-cc1"

// CHECK-X64: "-cc1"
// CHECK-X64-NOT: "-fdebug-compilation-dir

// RUN: %clang -target armv7-apple-ios -fembed-bitcode=all -c %s -o /dev/null -### 2>&1 \
// RUN:    | FileCheck --check-prefix=CHECK-ARM %s

// CHECK-ARM: "-cc1"

// CHECK-ARM: "-cc1"
// CHECK-ARM: "-target-abi"
// CHECK-ARM: "apcs-gnu"
// CHECK-ARM-NOT: "-fdebug-compilation-dir

// RUN: %clang -target arm64-apple-ios -fembed-bitcode=all -c %s -o /dev/null -### 2>&1 \
// RUN:    | FileCheck --check-prefix=CHECK-AARCH64 %s

// CHECK-AARCH64: "-cc1"

// CHECK-AARCH64: "-cc1"
// CHECK-AARCH64: "-target-abi"
// CHECK-AARCH64: "darwinpcs"
// CHECK-AARCH64-NOT: "-fdebug-compilation-dir

// RUN: %clang --target=hexagon-unknown-elf -ffixed-r19 -fembed-bitcode=all -c %s -### 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-HEXAGON %s
// CHECK-HEXAGON: "-target-feature"
// CHECK-HEXAGON: "+reserved-r19"
//
// RUN: %clang --target=wasm32-unknown-unknown -fembed-bitcode=all -pthread -c %s -o /dev/null -### 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-WASM %s

// CHECK-WASM: "-cc1"
// CHECK-WASM: "-target-feature" "+atomics"

// CHECK-WASM: "-cc1"
// CHECK-WASM: "-target-feature" "+atomics"
