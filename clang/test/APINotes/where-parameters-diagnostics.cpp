// RUN: not %clang_cc1 -fsyntax-only -fapinotes %s -I %S/Inputs/WhereParametersDuplicateSelectorDiag 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wapinotes -fsyntax-only -I %S/Inputs/Headers %s -x c++ 2>&1 | FileCheck %s --check-prefix=UNMATCHED --implicit-check-not=diagnosticMatchedGlobal --implicit-check-not=diagnosticAliasMatchedGlobal --implicit-check-not=diagnosticMatchedMethod --implicit-check-not=diagnosticAliasMatchedMethod
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter diagnosticBroadGlobal -x c++ | FileCheck %s --check-prefix=BROAD-GLOBAL
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter DiagnosticWidget::diagnosticBroadMethod -x c++ | FileCheck %s --check-prefix=BROAD-METHOD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter diagnosticMatchedGlobal -x c++ | FileCheck %s --check-prefix=MATCHED-GLOBAL
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter diagnosticAliasMatchedGlobal -x c++ | FileCheck %s --check-prefix=ALIAS-MATCHED-GLOBAL
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter DiagnosticWidget::diagnosticMatchedMethod -x c++ | FileCheck %s --check-prefix=MATCHED-METHOD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter DiagnosticWidget::diagnosticAliasMatchedMethod -x c++ | FileCheck %s --check-prefix=ALIAS-MATCHED-METHOD

#include "WhereParametersDiagnostics.h"

// DUPLICATE: error: duplicate definition of global function 'duplicateGlobal' with Where.Parameters [int]
// DUPLICATE: error: duplicate definition of global function 'duplicateEmpty' with Where.Parameters []
// DUPLICATE: error: duplicate definition of C++ method 'duplicateMethod' with Where.Parameters [int]
// DUPLICATE: error: duplicate definition of C++ method 'duplicateEmpty' with Where.Parameters []

// UNMATCHED-DAG: warning: API notes entry for 'unmatchedGlobal' has unmatched Where.Parameters [int]
// UNMATCHED-DAG: warning: API notes entry for 'diagnosticBroadGlobal' has unmatched Where.Parameters [int]
// UNMATCHED-DAG: warning: API notes entry for 'unmatchedMethod' has unmatched Where.Parameters [int]
// UNMATCHED-DAG: warning: API notes entry for 'diagnosticBroadMethod' has unmatched Where.Parameters [int]

// BROAD-GLOBAL: FunctionDecl {{.+}} diagnosticBroadGlobal 'void (float)'
// BROAD-GLOBAL: SwiftPrivateAttr
// BROAD-GLOBAL-NOT: SwiftNameAttr

// BROAD-METHOD: CXXMethodDecl {{.+}} diagnosticBroadMethod 'void (float)'
// BROAD-METHOD: SwiftPrivateAttr
// BROAD-METHOD-NOT: SwiftNameAttr

// MATCHED-GLOBAL: FunctionDecl {{.+}} diagnosticMatchedGlobal 'void (int)'
// MATCHED-GLOBAL: SwiftNameAttr {{.+}} "diagnosticMatchedGlobal(_:)"

// ALIAS-MATCHED-GLOBAL: FunctionDecl {{.+}} diagnosticAliasMatchedGlobal 'void (DiagnosticAliasInt)'
// ALIAS-MATCHED-GLOBAL: SwiftNameAttr {{.+}} "diagnosticAliasMatchedGlobal(_:)"

// MATCHED-METHOD: CXXMethodDecl {{.+}} diagnosticMatchedMethod 'void (int)'
// MATCHED-METHOD: SwiftNameAttr {{.+}} "diagnosticMatchedMethod(_:)"

// ALIAS-MATCHED-METHOD: CXXMethodDecl {{.+}} diagnosticAliasMatchedMethod 'void (DiagnosticAliasInt)'
// ALIAS-MATCHED-METHOD: SwiftNameAttr {{.+}} "diagnosticAliasMatchedMethod(_:)"
