// Option should not be passed to the frontend by default.
// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address %s \
// RUN:   -### 2>&1 | \
// RUN:   FileCheck %s
// CHECK-NOT: -fsanitize-address-destructor-kind

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=none %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-NONE-ARG %s
// CHECK-NONE-ARG: "-fsanitize-address-destructor-kind=none"

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=global %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-GLOBAL-ARG %s
// CHECK-GLOBAL-ARG: "-fsanitize-address-destructor-kind=global"

// RUN: %clang -target x86_64-apple-macosx10.15-gnu -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=bad_arg %s -### 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-INVALID-ARG %s
// CHECK-INVALID-ARG: error: unsupported argument 'bad_arg' to option 'fsanitize-address-destructor-kind='
