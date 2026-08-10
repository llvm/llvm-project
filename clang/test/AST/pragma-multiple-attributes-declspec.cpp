// RUN: %clang_cc1 -triple i386-pc-win32 -fms-extensions -fms-compatibility -ast-dump %s | FileCheck %s

#pragma clang attribute push (__declspec(dllexport, noinline), apply_to=function)
void func1();
#pragma clang attribute pop
// CHECK: FunctionDecl {{.*}} func1
// CHECK-NEXT: DLLExportAttr {{.*}}
// CHECK-NEXT: NoInlineAttr {{.*}}
