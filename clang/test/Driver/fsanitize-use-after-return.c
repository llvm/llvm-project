// Option should not be passed to the frontend by default.
// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address %s \
// RUN:   -### 2>&1 | \
// RUN:   FileCheck %s
// CHECK-NOT: -fsanitize-address-use-after-return

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-use-after-return=never %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-NEVER-ARG %s
// CHECK-NEVER-ARG: "-fsanitize-address-use-after-return=never"

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-use-after-return=runtime %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-RUNTIME %s
// CHECK-RUNTIME: "-fsanitize-address-use-after-return=runtime"

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-use-after-return=always %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-ALWAYS-ARG %s

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-use-after-return=never \
// RUN:   -fsanitize-address-use-after-return=always %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-ALWAYS-ARG %s
// CHECK-ALWAYS-ARG: "-fsanitize-address-use-after-return=always"

// RUN: not %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-use-after-return=bad_arg %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-INVALID-ARG %s
// CHECK-INVALID-ARG: error: unsupported argument 'bad_arg' to option '-fsanitize-address-use-after-return='
