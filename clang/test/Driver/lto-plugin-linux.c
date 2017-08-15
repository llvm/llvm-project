// Check that non-Windows, non-Darwin OSs use LLVMgold.so.
// REQUIRES: !system-darwin && !system-windows
// RUN: %clang -### %s -flto 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LTO-PLUGIN %s
//
// CHECK-LTO-PLUGIN: "-plugin" "{{.*}}/LLVMgold.so"
