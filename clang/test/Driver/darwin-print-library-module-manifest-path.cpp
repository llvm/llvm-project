// Test that -print-library-module-manifest-path finds the correct file.
//
// Note this file is currently not available on Apple platforms

// RUN: %clang --print-library-module-manifest-path \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --target=x86_64-unknown-linux-gnu 2>&1 \
// RUN:   | FileCheck %s
// CHECK: <NOT PRESENT>
