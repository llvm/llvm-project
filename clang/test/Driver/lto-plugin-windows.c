// Check that Windows uses LLVMgold.dll.
// REQUIRES: system-windows
// RUN: %clang -### %s -flto 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LTO-PLUGIN %s
//
// CHECK-LTO-PLUGIN: "-plugin" "{{.*}}/LLVMgold.dll"
