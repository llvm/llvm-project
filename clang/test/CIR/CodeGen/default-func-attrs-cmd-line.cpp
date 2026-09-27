// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -default-function-attr "key=value" -default-function-attr "just_key" -default-function-attr "key-2=1" -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -default-function-attr "key=value" -default-function-attr "just_key" -default-function-attr "key-2=1" -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -default-function-attr "key=value" -default-function-attr "just_key" -default-function-attr "key-2=1"  -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" {
// CIR: cir.func{{.*}}@func() attributes {
// CIR-SAME: default_func_attrs = {just_key, key = "value", "key-2" = "1"}
// LLVM: define{{.*}}@func() #[[FUNC_ATTRS:.*]] {
void func() {}

void caller() {
  func();
  // CIR: cir.call @func()
  // CIR-SAME: default_func_attrs = {just_key, key = "value", "key-2" = "1"}
  // LLVM: call void{{.*}}@func() #[[FUNC_CALL_ATTRS:.*]]
}
}

// LLVM: attributes #[[FUNC_ATTRS]] =
// LLVM-SAME: "just_key"
// LLVM-SAME: "key"="value"
// LLVM-SAME: "key-2"="1"
// LLVM: attributes #[[FUNC_CALL_ATTRS]] =
// LLVM-SAME: "just_key"
// LLVM-SAME: "key"="value"
// LLVM-SAME: "key-2"="1"
