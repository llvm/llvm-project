// RUN: %clang --target=i386-linux-gnu -fsanitize=undefined -fsanitize-trap-loop %s -### 2>&1 | FileCheck --check-prefix=CHECK-SUPPORTED %s
// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap-loop %s -### 2>&1 | FileCheck --check-prefix=CHECK-SUPPORTED %s
// RUN: %clang --target=aarch64-linux-gnu -fsanitize=undefined -fsanitize-trap-loop %s -### 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED %s

// CHECK-SUPPORTED: "-fsanitize-trap-loop"
// CHECK-UNSUPPORTED-NOT: "-fsanitize-trap-loop"

