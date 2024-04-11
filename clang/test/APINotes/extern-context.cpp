// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInExternC -x c++ | FileCheck -check-prefix=CHECK-EXTERN-C %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInExternCXX -x c++ | FileCheck -check-prefix=CHECK-EXTERN-CXX %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalFuncInExternC -x c++ | FileCheck -check-prefix=CHECK-FUNC-EXTERN-C %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalFuncInExternCXX -x c++ | FileCheck -check-prefix=CHECK-FUNC-EXTERN-CXX %s

#include "ExternCtx.h"

// CHECK-EXTERN-C: Dumping globalInExternC:
// CHECK-EXTERN-C: VarDecl {{.+}} imported in ExternCtx globalInExternC 'int'
// CHECK-EXTERN-C: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-EXTERN-CXX: Dumping globalInExternCXX:
// CHECK-EXTERN-CXX: VarDecl {{.+}} imported in ExternCtx globalInExternCXX 'int'
// CHECK-EXTERN-CXX: UnavailableAttr {{.+}} <<invalid sloc>> "oh no #2"

// CHECK-FUNC-EXTERN-C: Dumping globalFuncInExternC:
// CHECK-FUNC-EXTERN-C: FunctionDecl {{.+}} imported in ExternCtx globalFuncInExternC 'void ()'
// CHECK-FUNC-EXTERN-C: UnavailableAttr {{.+}} <<invalid sloc>> "oh no #3"

// CHECK-FUNC-EXTERN-CXX: Dumping globalFuncInExternCXX:
// CHECK-FUNC-EXTERN-CXX: FunctionDecl {{.+}} imported in ExternCtx globalFuncInExternCXX 'void ()'
// CHECK-FUNC-EXTERN-CXX: UnavailableAttr {{.+}} <<invalid sloc>> "oh no #4"
