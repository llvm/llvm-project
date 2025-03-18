// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Lifetimebound -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Lifetimebound -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter funcToAnnotate -x c++ | FileCheck --check-prefix=CHECK-PARAM %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Lifetimebound -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter annotateThis -x c++ | FileCheck --check-prefix=CHECK-METHOD-THIS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Lifetimebound -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter methodToAnnotate -x c++ | FileCheck --check-prefix=CHECK-METHOD %s
#include "Lifetimebound.h"

// CHECK-PARAM: FunctionDecl {{.+}} funcToAnnotate 
// CHECK-PARAM-NEXT: ParmVarDecl {{.+}} p
// CHECK-PARAM-NEXT: LifetimeBoundAttr

// CHECK-METHOD: CXXMethodDecl {{.+}} methodToAnnotate 
// CHECK-METHOD-NEXT: ParmVarDecl {{.+}} p
// CHECK-METHOD-NEXT: LifetimeBoundAttr

// CHECK-METHOD-THIS: CXXMethodDecl {{.+}} annotateThis 'int *() {{\[\[}}clang::lifetimebound{{\]\]}}'
// CHECK-METHOD-THIS: CXXMethodDecl {{.+}} annotateThis2 'int *() {{\[\[}}clang::lifetimebound{{\]\]}}'
