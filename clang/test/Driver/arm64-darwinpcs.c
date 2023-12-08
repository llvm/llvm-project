// RUN: %clang -target arm64-apple-ios7.0 -### %s 2>&1 | FileCheck %s

// CHECK: "-target-abi" "darwinpcs"
