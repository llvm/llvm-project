// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -verify -Wunsafe-buffer-usage -Wno-unused-value
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -Wunused-value -ast-dump -ast-dump-filter unsafeFunc | FileCheck %s --check-prefixes=CHECK-UNSAFE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -Wunused-value -ast-dump -ast-dump-filter annotatedUnsafeFunc | FileCheck %s --check-prefixes=CHECK-ANNO
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -Wunused-value -ast-dump -ast-dump-filter falseAPINotesButAnnotatedUnsafeFunc | FileCheck %s --check-prefixes=CHECK-ANNO-FALSE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -Wunused-value -ast-dump -ast-dump-filter funcWithoutAnnotation | FileCheck %s --check-prefixes=CHECK-UNANNO

#include "UnsafeBufferUsage.h"

// CHECK-UNSAFE: FunctionDecl {{.+}} unsafeFunc
// CHECK-UNSAFE: UnsafeBufferUsageAttr

// If the function already has the attribute, the APINotes does
// nothing no matter how the attribute is specified in the .apinotes
// file:

// CHECK-ANNO: FunctionDecl {{.+}} annotatedUnsafeFunc
// CHECK-ANNO: UnsafeBufferUsageAttr
// CHECK-ANNO-NOT: UnsafeBufferUsageAttr
// CHECK-ANNO: FunctionDecl {{.+}} annotatedUnsafeFunc

// CHECK-ANNO-FALSE: FunctionDecl {{.+}} falseAPINotesButAnnotatedUnsafeFunc
// CHECK-ANNO-FALSE: UnsafeBufferUsageAttr

// CHECK-UNANNO: FunctionDecl {{.+}} funcWithoutAnnotation
// CHECK-UNANNO-NOT: UnsafeBufferUsageAttr

void unsafeFunc(int *p, int n) {
  p[n]; // no warning
}

void annotatedUnsafeFunc(int *p, int n) {
  p[n]; // no warning
}

void falseAPINotesButAnnotatedUnsafeFunc(int *p, int n) {
  p[n]; // no warning
}

void funcWithoutAnnotation(int *p, int n) {
  p[n]; // expected-warning{{unsafe buffer access}}
}

void caller(int *p, int n) {
  unsafeFunc(p, n); // expected-warning{{function introduces unsafe buffer manipulation}}
  annotatedUnsafeFunc(p, n); // expected-warning{{function introduces unsafe buffer manipulation}}
  falseAPINotesButAnnotatedUnsafeFunc(p, n); // expected-warning{{function introduces unsafe buffer manipulation}}
  funcWithoutAnnotation(p, n);   // no warning
}
