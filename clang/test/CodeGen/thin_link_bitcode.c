// REQUIRES: x86-registered-target
//
// RUN: %clang -flto=thin -fthin-link-bitcode=%t.bc -target x86_64-unknown-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefix=LINKBC
//
// RUN: %clang_cc1 -o %t -flto=thin -fthin-link-bitcode=%t.nodebug -triple x86_64-unknown-linux-gnu -emit-llvm-bc -debug-info-kind=limited %s
// RUN: llvm-bcanalyzer -dump %t | FileCheck %s
// RUN: llvm-bcanalyzer -dump %t.nodebug | FileCheck %s --check-prefix=NO_DEBUG
// RUN: %clang_cc1 -o %t.newpm -flto=thin -fexperimental-new-pass-manager -fthin-link-bitcode=%t.newpm.nodebug -triple x86_64-unknown-linux-gnu -emit-llvm-bc -debug-info-kind=limited  %s
// RUN: llvm-bcanalyzer -dump %t.newpm | FileCheck %s
// RUN: llvm-bcanalyzer -dump %t.newpm.nodebug | FileCheck %s --check-prefix=NO_DEBUG
int main (void) {
  return 0;
}

// LINKBC: -fthin-link-bitcode=

// CHECK: COMPILE_UNIT
// NO_DEBUG-NOT: COMPILE_UNIT
