// RUN: %clang_cc1 -triple arm-apple-darwin -ast-dump -ast-dump-filter Test %s | FileCheck --strict-whitespace %s

__attribute__((interrupt)) void Test(void);
// CHECK: FunctionDecl{{.*}}Test
// CHECK-NEXT: ARMInterruptAttr
