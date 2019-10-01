// Run cc1as with debug on empty file. Needs a known name so we can check it.
// REQUIRES: x86-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp %s %t/comment.s
// RUN: %clang -cc1as -triple x86_64-linux-gnu -filetype asm -debug-info-kind=limited -dwarf-version=4 %t/comment.s | FileCheck %s
// RUN: %clang -cc1as -triple x86_64-linux-gnu -filetype asm -debug-info-kind=limited -dwarf-version=5 %t/comment.s | FileCheck %s
// Asm output actually emits the .section directives twice.
// CHECK: {{\.}}section .debug_info
// CHECK: {{\.}}section .debug_info
// CHECK-NOT: {{\.}}section
// Look for this as a relative path.
// CHECK: .ascii "{{[^\\/].*}}comment.s"
