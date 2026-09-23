// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildRef -x c++ | FileCheck --check-prefix=REF %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildConst -x c++ | FileCheck --check-prefix=CONST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildVolatile -x c++ | FileCheck --check-prefix=VOLATILE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildNone -x c++ | FileCheck --check-prefix=NONE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildCombined -x c++ | FileCheck --check-prefix=COMBINED-CVREF %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildConstRValue -x c++ | FileCheck --check-prefix=COMBINED-CONST-RVALUE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildVolatileRValue -x c++ | FileCheck --check-prefix=COMBINED-VOLATILE-RVALUE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildObjectOnly -x c++ | FileCheck --check-prefix=OBJECT-ONLY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildTwoObjectNotes -x c++ | FileCheck --check-prefix=MULTI-OBJECT %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildObjectAndParameter -x c++ | FileCheck --check-prefix=OBJECT-PARAMETER %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::buildStatic -x c++ | FileCheck --check-prefix=STATIC %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiers -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter ObjectBuilder::consume -x c++ | FileCheck --check-prefix=PARAM-REF %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiersWarnings -fdisable-module-hash -fapinotes-modules -Wapinotes -fsyntax-only -I %S/Inputs/Headers %s -x c++ 2>&1 | FileCheck --check-prefix=UNMATCHED %s
// RUN: %clang_cc1 -std=c++23 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiersCXX23 -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter buildExplicit -x c++ | FileCheck --check-prefix=EXPLICIT-OBJECT %s
// RUN: %clang_cc1 -std=c++23 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereObjectQualifiersCXX23Warnings -fdisable-module-hash -fapinotes-modules -Wapinotes -fsyntax-only -I %S/Inputs/Headers %s -x c++ 2>&1 | FileCheck --check-prefix=EXPLICIT-OBJECT-UNMATCHED %s
// RUN: not %clang_cc1 -fsyntax-only -fapinotes %s -I %S/Inputs/WhereObjectQualifiersDiag 2>&1 | FileCheck --check-prefix=DIAG %s

#include "WhereObjectQualifiers.h"

// REF: CXXMethodDecl {{.+}} buildRef 'void () &{{.*}}'
// REF: SwiftNameAttr {{.+}} "buildFromLValue()"
// REF: CXXMethodDecl {{.+}} buildRef 'void () &&{{.*}}'
// REF: SwiftNameAttr {{.+}} "buildFromRValue()"

// CONST: CXXMethodDecl {{.+}} buildConst 'void () &{{.*}}'
// CONST: SwiftNameAttr {{.+}} "buildMutableLValue()"
// CONST: CXXMethodDecl {{.+}} buildConst 'void () const &{{.*}}'
// CONST: SwiftNameAttr {{.+}} "buildConstLValue()"

// VOLATILE: CXXMethodDecl {{.+}} buildVolatile 'void (){{.*}}'
// VOLATILE: SwiftNameAttr {{.+}} "buildNonVolatile()"
// VOLATILE: CXXMethodDecl {{.+}} buildVolatile 'void () volatile{{.*}}'
// VOLATILE: SwiftNameAttr {{.+}} "buildVolatile()"

// NONE: CXXMethodDecl {{.+}} buildNone 'void (){{.*}}'
// NONE: SwiftNameAttr {{.+}} "buildUnqualified()"

// COMBINED-CVREF: CXXMethodDecl {{.+}} buildCombined 'void () const volatile &{{.*}}'
// COMBINED-CVREF: SwiftNameAttr {{.+}} "buildConstVolatileLValue()"
// COMBINED-CONST-RVALUE: CXXMethodDecl {{.+}} buildConstRValue 'void () const &&{{.*}}'
// COMBINED-CONST-RVALUE: SwiftNameAttr {{.+}} "buildConstRValue()"
// COMBINED-VOLATILE-RVALUE: CXXMethodDecl {{.+}} buildVolatileRValue 'void () volatile &&{{.*}}'
// COMBINED-VOLATILE-RVALUE: SwiftNameAttr {{.+}} "buildVolatileRValue()"

// OBJECT-ONLY: CXXMethodDecl {{.+}} buildObjectOnly 'void (int) &{{.*}}'
// OBJECT-ONLY: SwiftNameAttr {{.+}} "buildObjectOnlyLValue(_:)"
// OBJECT-ONLY: CXXMethodDecl {{.+}} buildObjectOnly 'void (double) &{{.*}}'
// OBJECT-ONLY: SwiftNameAttr {{.+}} "buildObjectOnlyLValue(_:)"
// OBJECT-ONLY: CXXMethodDecl {{.+}} buildObjectOnly 'void (int) &&{{.*}}'
// OBJECT-ONLY-NOT: SwiftNameAttr

// MULTI-OBJECT: CXXMethodDecl {{.+}} buildTwoObjectNotes 'void () &{{.*}}'
// MULTI-OBJECT-DAG: SwiftPrivateAttr
// MULTI-OBJECT-DAG: SwiftNameAttr {{.+}} "buildTwoObjectNotesLValue()"

// OBJECT-PARAMETER: CXXMethodDecl {{.+}} buildObjectAndParameter 'void (int) const{{.*}}'
// OBJECT-PARAMETER-DAG: SwiftPrivateAttr
// OBJECT-PARAMETER-DAG: SwiftNameAttr {{.+}} "buildObjectAndParameterInt(_:)"

// STATIC: CXXMethodDecl {{.+}} buildStatic 'void ()' static
// STATIC-NOT: SwiftNameAttr

// UNMATCHED-DAG: warning: API notes entry for 'buildStatic' has unmatched Where.Parameters [] Object{Ref: none}
// UNMATCHED-DAG: warning: API notes entry for 'buildStaticObjectOnly' has unmatched Where.Object Object{Ref: none}

// EXPLICIT-OBJECT: CXXMethodDecl {{.+}} buildExplicitConst 'void (const ExplicitObjectBuilder &)'
// EXPLICIT-OBJECT-NOT: SwiftNameAttr
// EXPLICIT-OBJECT: CXXMethodDecl {{.+}} buildExplicitLValue 'void (ExplicitObjectBuilder &)'
// EXPLICIT-OBJECT-NOT: SwiftNameAttr

// EXPLICIT-OBJECT-UNMATCHED-DAG: warning: API notes entry for 'buildExplicitConst' has unmatched Where.Object Object{Const: false, Ref: none}
// EXPLICIT-OBJECT-UNMATCHED-DAG: warning: API notes entry for 'buildExplicitLValue' has unmatched Where.Object Object{Ref: lvalue}

// PARAM-REF: CXXMethodDecl {{.+}} consume 'void (ObjectBuffer &)'
// PARAM-REF: SwiftNameAttr {{.+}} "consumeBorrowed(_:)"
// PARAM-REF: CXXMethodDecl {{.+}} consume 'void (ObjectBuffer &&)'
// PARAM-REF: SwiftNameAttr {{.+}} "consumeOwned(_:)"

// DIAG-DAG: error: 'Object' is only supported on C++ methods
// DIAG-DAG: error: 'Object' requires at least one field
// DIAG-DAG: error: multiple API notes entries for C++ method 'duplicate' with Where.Parameters [] Object{Ref: lvalue}
