// Check that we get the right Android version.

// RUN: not %clang --target=aarch64-linux-androidS -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s

// CHECK-ERROR: error: version 'S' in target triple 'aarch64-unknown-linux-androidS' is invalid

// RUN: not %clang --target=armv7-linux-androideabiS -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR1 %s

// CHECK-ERROR1: error: version 'S' in target triple 'armv7-unknown-linux-androidS' is invalid

// RUN: %clang --target=aarch64-linux-android31 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-TARGET %s

// CHECK-TARGET: "aarch64-unknown-linux-android31"

// RUN: not %clang --target=armv7-linux-gnuS -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR2 %s

// CHECK-ERROR2: error: version 'S' in target triple 'armv7-unknown-linux-gnuS' is invalid

// RUN: %clang --target=wasm32-unknown-wasi-preview2 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-WASM %s

// CHECK-WASM: "-triple" "wasm32-unknown-wasi-preview2"

// RUN: %clang --target=wasm32-wasi-pthread -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-WASM1 %s

// CHECK-WASM1: "-triple" "wasm32-unknown-wasi-pthread"
