// RUN: %clangxx --target=aarch64-unknown-linux -fno-rtti -Xclang -fexperimental-omit-vtable-rtti -c %s -### 2>&1 | FileCheck %s --check-prefix=OMIT
// RUN: %clangxx --target=aarch64-unknown-linux -fno-rtti -Xclang -fno-experimental-omit-vtable-rtti -c %s -### 2>&1 | FileCheck %s --check-prefix=NO-OMIT

// OMIT: "-fexperimental-omit-vtable-rtti"
// NO-OMIT-NOT: "-fexperimental-omit-vtable-rtti"
