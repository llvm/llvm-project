// REQUIRES: x86-registered-target
// RUN: not %clang -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=bar-format %s 2>&1 | FileCheck %s

// CHECK: error: invalid value
// CHECK: '-interface-stub-version=<experimental-tapi-elf-v1 |
// CHECK: experimental-yaml-elf-v1>' in 'Must specify a valid interface
// CHECK: stub format type using
