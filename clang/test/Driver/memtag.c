// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-CC1-ALL,CHECK-SYNC,CHECK-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag-stack %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-CC1-STACK,CHECK-SYNC,CHECK-NO-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag-heap %s 2>&1 | FileCheck %s \
// RUN:   --check-prefixes=CHECK-CC1-HEAP,CHECK-SYNC,CHECK-HEAP,CHECK-NO-STACK

// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag -fsanitize-memtag-mode=async %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK-CC1-ALL,CHECK-ASYNC,CHECK-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag-stack -fsanitize-memtag-mode=async %s 2>&1 \
// RUN:   | FileCheck %s \
// RUN:   --check-prefixes=CHECK-CC1-STACK,CHECK-ASYNC,CHECK-NO-HEAP,CHECK-STACK

// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag-heap -fsanitize-memtag-mode=async %s 2>&1 \
// RUN:   | FileCheck %s \
// RUN:   --check-prefixes=CHECK-CC1-HEAP,CHECK-ASYNC,CHECK-HEAP,CHECK-NO-STACK

// RUN: not %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag-heap -fsanitize-memtag-mode=asymm %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-INVALID-MODE

// RUN: not %clang -### --target=aarch64-linux-gnu -march=armv8+memtag \
// RUN:   -fsanitize=memtag-stack -fsanitize=memtag-heap \
// RUN:   -fsanitize-memtag-mode=asymm -fno-sanitize=memtag %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-NONE

// CHECK-CC1-ALL:           "-fsanitize=memtag-stack,memtag-heap,memtag-globals"
// CHECK-CC1-STACK:         "-fsanitize=memtag-stack"
// CHECK-CC1-HEAP:          "-fsanitize=memtag-heap"

// CHECK-ASYNC:             ld{{.*}} "-z" "memtag-mode=async"
// CHECK-SYNC:              ld{{.*}} "-z" "memtag-mode=sync"
// CHECK-HEAP:              "-z" "memtag-heap"
// CHECK-NO-HEAP-NOT:       "-z" "memtag-heap"
// CHECK-STACK:             "-z" "memtag-stack"
// CHECK-NO-STACK-NOT:      "-z" "memtag-stack"

// CHECK-INVALID-MODE:      invalid value 'asymm' in '-fsanitize-memtag-mode=',
// CHECK-INVALID-MODE-SAME: expected one of: {async, sync}

// CHECK-NONE-NOT:          "-fsanitize=memtag-stack"
// CHECK-NONE-NOT:          "-fsanitize=memtag-heap"
// CHECK-NONE-NOT:          ld{{.*}} "-z" "memtag

void f() {}
