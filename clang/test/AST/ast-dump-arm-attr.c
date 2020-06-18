// RUN: %clang_cc1 -triple arm-apple-darwin -ast-dump -ast-dump-filter Test %s | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -triple armv8m.base-none-eabi -mcmse -ast-dump -ast-dump-filter Test %s | FileCheck --strict-whitespace %s --check-prefix=CHECK-CMSE

__attribute__((interrupt)) void Test(void);
// CHECK: FunctionDecl{{.*}}Test
// CHECK-NEXT: ARMInterruptAttr

typedef int (*CmseTest)(int a) __attribute__((cmse_nonsecure_call));
// CHECK-CMSE: TypedefDecl{{.*}}CmseTest{{.*}}__attribute__((cmse_nonsecure_call))
