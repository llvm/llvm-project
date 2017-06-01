// RUN: %clang_cc1 -code-completion-at=%s:3:1 %s | FileCheck %s
void func() {

}
// CHECK: COMPLETION: __auto_type
