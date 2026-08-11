// TODO(cir): drop -fno-clangir-call-conv-lowering once CallConvLowering
// supports the builtin i32 in the Itanium EH personality signature.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -default-function-attr "key=value" -default-function-attr "just_key" -fcxx-exceptions -fexceptions -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -default-function-attr "key=value" -default-function-attr "just_key" -fcxx-exceptions -fexceptions -fclangir -fno-clangir-call-conv-lowering -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -default-function-attr "key=value" -default-function-attr "just_key" -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" void may_throw();

// CIR-LABEL: cir.func {{.*}}@caller()
// CIR: cir.call @may_throw()
// LLVM-LABEL: define{{.*}}@caller()
// LLVM: invoke void @may_throw() #[[ATTRS:[0-9]+]]
extern "C" void caller() {
  try {
    may_throw();
  } catch (...) {
  }
}

// CIR-SAME: default_func_attrs
// CIR-SAME: just_key
// CIR-SAME: key = "value"
// LLVM: attributes #[[ATTRS]]
// LLVM-SAME: "just_key"
// LLVM-SAME: "key"="value"
