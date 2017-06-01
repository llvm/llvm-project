// RUN: not %clang_cc1 -triple le32-unknown-nacl %s -S -o - 2>&1 | FileCheck %s

// CHECK: error: unable to create target: 'No available targets are compatible with this triple.
