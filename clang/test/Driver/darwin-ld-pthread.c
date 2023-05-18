// RUN: %clang -Wunused-command-line-argument -pthread -target x86_64-apple-darwin -### /dev/null -o /dev/null 2>&1 | FileCheck %s

// There is nothing to do at link time to get pthread support. But do not warn.
// CHECK-NOT: argument unused during compilation: '-pthread'
