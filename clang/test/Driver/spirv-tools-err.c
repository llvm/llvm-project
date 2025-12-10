// UNSUPPORTED: spirv-tools
// REQUIRES: spirv-registered-target

// RUN: env PATH='' not %clang -o /dev/null --target=spirv64 --save-temps -v %s 2>&1 | FileCheck %s

// CHECK: error: cannot find SPIR-V Tools binary 'spirv-as'
// CHECK: error: cannot find SPIR-V Tools binary 'spirv-link'
