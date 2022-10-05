// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: touch %t/f.o
// RUN: mkdir -p %t/MacOSX12.0.sdk

// RUN: %clang -fuse-ld= -arch arm64 -mlinker-version=520 -isysroot %t/MacOSX12.sdk/does-not-exist -### %t/f.o 2>&1 | FileCheck %s

// CHECK: "-platform_version" "macos" "[[NUMBER:[0-9]+]].0.0" "[[NUMBER]].0"

// REQUIRES: system-darwin
