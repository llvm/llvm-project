// REQUIRES: default-pie-on-linux
/// Test -DCLANG_DEFAULT_PIE_ON_LINUX=on.

// RUN: %clang -### --target=aarch64-linux-gnu %s 2>&1 | FileCheck %s --check-prefix=PIE2

// PIE2: "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"
// PIE2: "-pie"
