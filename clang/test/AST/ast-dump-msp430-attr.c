// Test without serialization:
// RUN: %clang_cc1 -triple msp430-unknown-unknown -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple msp430-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple msp430-unknown-unknown -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

__attribute__((interrupt(12))) void Test(void);
// CHECK: FunctionDecl{{.*}}Test
// CHECK-NEXT: MSP430InterruptAttr
