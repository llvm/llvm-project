// RUN: %clang --target=i386-apple-ios13.1-macabi -c -### %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: 32-bit targets are not supported when building for Mac Catalyst
