// RUN: %clang --target=arm64-apple-darwin -fsanitize-stable-abi %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-ASAN-STABLE-WARN
// CHECK-ASAN-STABLE-WARN: warning: argument unused during compilation: '-fsanitize-stable-abi'
// RUN: %clang --target=arm64-apple-darwin -fsanitize=address -fsanitize-stable-abi %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-ASAN-STABLE-OK
// RUN: %clang --target=arm64-apple-darwin -fsanitize=address -fno-sanitize-stable-abi -fsanitize-stable-abi %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-ASAN-STABLE-OK
// CHECK-ASAN-STABLE-OK: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// CHECK-ASAN-STABLE-OK: "-mllvm" "-asan-max-inline-poisoning-size=0"
// RUN: %clang --target=arm64-apple-darwin -fsanitize=address -fno-sanitize-stable-abi %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-NO-ASAN-STABLE-OK
// RUN: %clang --target=arm64-apple-darwin -fsanitize=address -fsanitize-stable-abi -fno-sanitize-stable-abi %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK-NO-ASAN-STABLE-OK
// CHECK-NO-ASAN-STABLE-OK-NOT: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// CHECK-NO-ASAN-STABLE-OK-NOT: "-mllvm" "-asan-max-inline-poisoning-size=0"
