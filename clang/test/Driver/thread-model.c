// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -mthread-model posix -v 2>&1 | FileCheck %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -v 2>&1 | FileCheck %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -mthread-model single -v 2>&1 | FileCheck --check-prefix=SINGLE %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -mthread-model silly -v 2>&1 | FileCheck --check-prefix=INVALID %s

// CHECK: Thread model: posix
// CHECK-NOT: "-mthread-model"
// SINGLE: Thread model: single
// SINGLE: "-mthread-model" "single"
// INVALID: error: invalid thread model 'silly' in '-mthread-model silly' for this target
// WIN32: Thread model: win32

// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v 2>&1 | FileCheck %s
// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v -mthread-model single 2>&1 | FileCheck --check-prefix=SINGLE %s
// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v -mthread-model posix 2>&1 | FileCheck %s
// RUN: not %clang -### --target=wasm32-unknown-linux-gnu -c %s -v -mthread-model silly 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: %clang -### -target wasm64-unknown-linux-gnu -c %s -v 2>&1 | FileCheck %s

// RUN: %clang -### -target x86_64-pc-windows-gnu -c %s -v 2>&1 | FileCheck --check-prefix=WIN32 %s
// RUN: %clang -### -target x86_64-pc-windows-gnu -c %s -v -mthread-model posix 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-pc-windows-gnu -c %s -v -mthread-model win32 2>&1 | FileCheck --check-prefix=WIN32 %s
// RUN: %clang -### -target x86_64-pc-windows-msvc -c %s -v 2>&1 | FileCheck --check-prefix=WIN32 %s
