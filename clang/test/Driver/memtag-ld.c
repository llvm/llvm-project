// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-SYNC,CHECK-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag-stack %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-SYNC,CHECK-NO-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag-heap %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-SYNC,CHECK-HEAP,CHECK-NO-STACK

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag -fsanitize-memtag-mode=async %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-ASYNC,CHECK-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag-stack -fsanitize-memtag-mode=async %s 2>&1 \
// RUN:   | FileCheck %s \
// RUN:   --check-prefixes=CHECK-ASYNC,CHECK-NO-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag-heap -fsanitize-memtag-mode=async %s 2>&1 \
// RUN:   | FileCheck %s \
// RUN:   --check-prefixes=CHECK-ASYNC,CHECK-HEAP,CHECK-NO-STACK

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag-heap -fsanitize-memtag-mode=asymm %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-INVALID-MODE

// RUN: %clang -### --target=aarch64-linux-android -march=armv8+memtag \
// RUN:   -fsanitize=memtag-stack -fsanitize=memtag-heap \
// RUN:   -fsanitize-memtag-mode=asymm -fno-sanitize=memtag %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-NONE

// CHECK-ASYNC:             ld{{.*}} "--android-memtag-mode=async"
// CHECK-SYNC:              ld{{.*}} "--android-memtag-mode=sync"
// CHECK-HEAP:              "--android-memtag-heap"
// CHECK-NO-HEAP-NOT:       "--android-memtag-heap"
// CHECK-STACK:             "--android-memtag-stack"
// CHECK-NO-STACK-NOT:      "--android-memtag-stack"
// CHECK-INVALID-MODE:      invalid value 'asymm' in '-fsanitize-memtag-mode=',
// CHECK-INVALID-MODE-SAME: expected one of: {async, sync}
// CHECK-NONE-NOT:          ld{{.*}} "--android-memtag

void f() {}
