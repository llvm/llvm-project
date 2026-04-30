// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++17 -verify -Wunsafe-buffer-usage -Wno-unused-value
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++17 -Wunused-value -ast-dump -ast-dump-filter unsafeFunc | FileCheck %s --check-prefixes=CHECK-UNSAFE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++17 -Wunused-value -ast-dump -ast-dump-filter annotatedUnsafeFunc | FileCheck %s --check-prefixes=CHECK-ANNO
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++17 -Wunused-value -ast-dump -ast-dump-filter falseAPINotesButAnnotatedUnsafeFunc | FileCheck %s --check-prefixes=CHECK-ANNO-FALSE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++17 -Wunused-value -ast-dump -ast-dump-filter funcWithoutAnnotation | FileCheck %s --check-prefixes=CHECK-UNANNO

// C++20 mode
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++20 -verify=expected,cpp20 -Wunsafe-buffer-usage -Wno-unused-value
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++20 -Wunused-value -ast-dump -ast-dump-filter unsafeFunc | FileCheck %s --check-prefixes=CHECK-UNSAFE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++20 -Wunused-value -ast-dump -ast-dump-filter annotatedUnsafeFunc | FileCheck %s --check-prefixes=CHECK-ANNO
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++20 -Wunused-value -ast-dump -ast-dump-filter falseAPINotesButAnnotatedUnsafeFunc | FileCheck %s --check-prefixes=CHECK-ANNO-FALSE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules\
// RUN:             -I %S/Inputs/Headers %s -x c++ -std=c++20 -Wunused-value -ast-dump -ast-dump-filter funcWithoutAnnotation | FileCheck %s --check-prefixes=CHECK-UNANNO

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
        // cpp20-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
}

void caller(int *p, int n) {
  unsafeFunc(p, n); // expected-warning{{function introduces unsafe buffer manipulation}}
                    // cpp20-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  annotatedUnsafeFunc(p, n); // expected-warning{{function introduces unsafe buffer manipulation}}
                             // cpp20-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  falseAPINotesButAnnotatedUnsafeFunc(p, n); // expected-warning{{function introduces unsafe buffer manipulation}}
                                             // cpp20-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  funcWithoutAnnotation(p, n);   // no warning
}
