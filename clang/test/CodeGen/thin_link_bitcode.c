// RUN: %clang_cc1 -o %t -flto=thin -fthin-link-bitcode=%t.nodebug -triple x86_64-unknown-linux-gnu -emit-llvm-bc -debug-info-kind=limited %s
// RUN: llvm-bcanalyzer -dump %t | FileCheck %s
// RUN: llvm-bcanalyzer -dump %t.nodebug | FileCheck %s --check-prefix=NO_DEBUG
int main (void) {
  return 0;
}

// CHECK: COMPILE_UNIT
// NO_DEBUG-NOT: COMPILE_UNIT
