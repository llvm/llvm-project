// RUN: %clang_cc1 -mpacked-stack -triple s390x-ibm-linux -emit-llvm \
// RUN:   -o - %s 2>&1 | FileCheck  %s
// RUN: not %clang_cc1 -mpacked-stack -triple x86_64-linux-gnu \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck -check-prefix=X86 %s

int foo(void) {
  return 0;
}

//CHECK: attributes #0 = { {{.*}}"packed-stack" {{.*}} }
//X86: error: option '-mpacked-stack' cannot be specified on this target
