// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInExternC -x c++ | FileCheck -check-prefixes=CHECK-EXTERN-C,CHECK-EXTERN-C17 %s
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInExternCXX -x c++ | FileCheck -check-prefixes=CHECK-EXTERN-CXX,CHECK-EXTERN-CXX17 %s
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalFuncInExternC -x c++ | FileCheck -check-prefixes=CHECK-FUNC-EXTERN-C,CHECK-FUNC-EXTERN-C17 %s
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalFuncInExternCXX -x c++ | FileCheck -check-prefixes=CHECK-FUNC-EXTERN-CXX,CHECK-FUNC-EXTERN-CXX17 %s


// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInExternC -x c++ | FileCheck -check-prefixes=CHECK-EXTERN-C,CHECK-EXTERN-C20 %s
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalInExternCXX -x c++ | FileCheck -check-prefixes=CHECK-EXTERN-CXX,CHECK-EXTERN-CXX20 %s
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalFuncInExternC -x c++ | FileCheck -check-prefixes=CHECK-FUNC-EXTERN-C,CHECK-FUNC-EXTERN-C20 %s
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter globalFuncInExternCXX -x c++ | FileCheck -check-prefixes=CHECK-FUNC-EXTERN-CXX,CHECK-FUNC-EXTERN-CXX20 %s


#include "ExternCtx.h"

// CHECK-EXTERN-C: Dumping globalInExternC:
// CHECK-EXTERN-C17: VarDecl {{.+}} imported in ExternCtx globalInExternC 'int'
// CHECK-EXTERN-C20: VarDecl {{.+}} imported in ExternCtx hidden globalInExternC 'int'
// CHECK-EXTERN-C: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-EXTERN-CXX: Dumping globalInExternCXX:
// CHECK-EXTERN-CXX17: VarDecl {{.+}} imported in ExternCtx globalInExternCXX 'int'
// CHECK-EXTERN-CXX20: VarDecl {{.+}} imported in ExternCtx hidden globalInExternCXX 'int'
// CHECK-EXTERN-CXX: UnavailableAttr {{.+}} <<invalid sloc>> "oh no #2"

// CHECK-FUNC-EXTERN-C: Dumping globalFuncInExternC:
// CHECK-FUNC-EXTERN-C17: FunctionDecl {{.+}} imported in ExternCtx globalFuncInExternC 'void ()'
// CHECK-FUNC-EXTERN-C20: FunctionDecl {{.+}} imported in ExternCtx hidden globalFuncInExternC 'void ()'
// CHECK-FUNC-EXTERN-C: UnavailableAttr {{.+}} <<invalid sloc>> "oh no #3"

// CHECK-FUNC-EXTERN-CXX: Dumping globalFuncInExternCXX:
// CHECK-FUNC-EXTERN-CXX17: FunctionDecl {{.+}} imported in ExternCtx globalFuncInExternCXX 'void ()'
// CHECK-FUNC-EXTERN-CXX20: FunctionDecl {{.+}} imported in ExternCtx hidden globalFuncInExternCXX 'void ()'
// CHECK-FUNC-EXTERN-CXX: UnavailableAttr {{.+}} <<invalid sloc>> "oh no #4"
