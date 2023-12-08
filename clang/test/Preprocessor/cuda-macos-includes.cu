// RUN: %clang -cc1 -fcuda-is-device -isysroot /var/empty \
// RUN:   -triple nvptx-nvidia-cuda -aux-triple i386-apple-macosx \
// RUN:   -E -fcuda-is-device -v -o /dev/null -x cuda %s 2>&1 | FileCheck %s

// RUN: %clang -cc1 -isysroot /var/empty \
// RUN:   -triple i386-apple-macosx -aux-triple nvptx-nvidia-cuda \
// RUN:   -E -fcuda-is-device -v -o /dev/null -x cuda %s 2>&1 | FileCheck %s

// Check that when we do CUDA host and device compiles on MacOS, we check for
// includes in /System/Library/Frameworks and /Library/Frameworks.

// CHECK-DAG: ignoring nonexistent directory "/var/empty/System/Library/Frameworks"
// CHECK-DAG: ignoring nonexistent directory "/var/empty/Library/Frameworks"
