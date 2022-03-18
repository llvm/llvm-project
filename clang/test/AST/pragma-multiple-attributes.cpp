// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

#pragma clang attribute push (__attribute__((disable_sanitizer_instrumentation, annotate("test1"))), apply_to=variable(is_global))
int var1;
#pragma clang attribute pop
// CHECK: VarDecl {{.*}} var1
// CHECK-NEXT: DisableSanitizerInstrumentationAttr {{.*}}
// CHECK-NEXT: AnnotateAttr {{.*}} "test1"

#pragma clang attribute push ([[clang::disable_sanitizer_instrumentation, clang::annotate("test2")]], apply_to=variable(is_global))
int var2;
#pragma clang attribute pop
// CHECK: VarDecl {{.*}} var2
// CHECK-NEXT: DisableSanitizerInstrumentationAttr {{.*}}
// CHECK-NEXT: AnnotateAttr {{.*}} "test2"
