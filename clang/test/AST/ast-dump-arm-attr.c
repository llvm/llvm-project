// Tests without serialization:
// RUN: %clang_cc1 -triple arm-apple-darwin -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// RUN: %clang_cc1 -triple armv8m.base-none-eabi -mcmse -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s --check-prefix=CHECK-CMSE
//
// Tests with serialization:
// RUN: %clang_cc1 -triple arm-apple-darwin -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple arm-apple-darwin -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s
//
// RUN: %clang_cc1 -triple armv8m.base-none-eabi -mcmse -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple armv8m.base-none-eabi -mcmse -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

__attribute__((interrupt)) void Test(void);
// CHECK: FunctionDecl{{.*}}Test
// CHECK-NEXT: ARMInterruptAttr

typedef int (*CmseTest)(int a) __attribute__((cmse_nonsecure_call));
// CHECK-CMSE: TypedefDecl{{.*}}CmseTest{{.*}}__attribute__((cmse_nonsecure_call))
