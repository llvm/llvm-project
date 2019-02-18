// RUN: %clang_cc1 -triple msp430-unknown-unknown -ast-dump -ast-dump-filter Test %s | FileCheck --strict-whitespace %s

__attribute__((interrupt(12))) void Test(void);
// CHECK: FunctionDecl{{.*}}Test
// CHECK-NEXT: MSP430InterruptAttr
