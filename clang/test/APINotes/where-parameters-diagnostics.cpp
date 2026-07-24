// RUN: rm -rf %t && split-file %s %t
// RUN: not %clang_cc1 -fsyntax-only -fapinotes %t/diagnostics.cpp -I %t/WhereParametersDuplicateSelectorDiag 2>&1 | FileCheck %t/WhereParametersDuplicateSelectorDiag/APINotes.apinotes --check-prefix=DUPLICATE
// RUN: rm -rf %t/ModulesCache && mkdir -p %t/ModulesCache
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wapinotes -fsyntax-only -I %S/Inputs/Headers %t/diagnostics.cpp -x c++ 2>&1 | FileCheck %s --check-prefix=UNMATCHED --implicit-check-not=diagnosticMatchedGlobal --implicit-check-not=diagnosticAliasMatchedGlobal --implicit-check-not=diagnosticMatchedMethod --implicit-check-not=diagnosticAliasMatchedMethod
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %t/diagnostics.cpp -ast-dump -ast-dump-filter diagnosticBroadGlobal -x c++ | FileCheck %s --check-prefix=BROAD-GLOBAL
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %t/diagnostics.cpp -ast-dump -ast-dump-filter DiagnosticWidget::diagnosticBroadMethod -x c++ | FileCheck %s --check-prefix=BROAD-METHOD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %t/diagnostics.cpp -ast-dump -ast-dump-filter diagnosticMatchedGlobal -x c++ | FileCheck %s --check-prefix=MATCHED-GLOBAL
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %t/diagnostics.cpp -ast-dump -ast-dump-filter diagnosticAliasMatchedGlobal -x c++ | FileCheck %s --check-prefix=ALIAS-MATCHED-GLOBAL
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %t/diagnostics.cpp -ast-dump -ast-dump-filter DiagnosticWidget::diagnosticMatchedMethod -x c++ | FileCheck %s --check-prefix=MATCHED-METHOD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersDiagnostics -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %t/diagnostics.cpp -ast-dump -ast-dump-filter DiagnosticWidget::diagnosticAliasMatchedMethod -x c++ | FileCheck %s --check-prefix=ALIAS-MATCHED-METHOD


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

//--- diagnostics.cpp
#include "WhereParametersDiagnostics.h"

//--- WhereParametersDuplicateSelectorDiag/WhereParametersDiagnostics.h
#ifndef WHERE_PARAMETERS_DIAGNOSTICS_H
#define WHERE_PARAMETERS_DIAGNOSTICS_H

void duplicateGlobal(int);
void duplicateEmpty();
void allowedGlobal(int);
void allowedGlobal(double);

struct DiagnosticWidget {
  void duplicateMethod(int);
  void duplicateEmpty();
  void allowed(int);
  void allowed(double);
};

#endif // WHERE_PARAMETERS_DIAGNOSTICS_H

//--- WhereParametersDuplicateSelectorDiag/APINotes.apinotes
---
Name: WhereParametersDiagnostics
Functions:
- Name: duplicateGlobal
  Where:
    Parameters:
    - int
  SwiftName: duplicateGlobalA(_:)
- Name: duplicateGlobal
  Where:
    Parameters:
    - int
  SwiftName: duplicateGlobalB(_:)
# DUPLICATE: error: multiple API notes entries for global function 'duplicateGlobal' with Where.Parameters [int]
- Name: duplicateEmpty
  Where:
    Parameters: []
  SwiftName: duplicateEmptyA()
- Name: duplicateEmpty
  Where:
    Parameters: []
  SwiftName: duplicateEmptyB()
# DUPLICATE: error: multiple API notes entries for global function 'duplicateEmpty' with Where.Parameters []
- Name: allowedGlobal
  SwiftPrivate: true
- Name: allowedGlobal
  Where:
    Parameters:
    - int
  SwiftName: allowedGlobalInt(_:)
- Name: allowedGlobal
  Where:
    Parameters:
    - double
  SwiftName: allowedGlobalDouble(_:)
Tags:
- Name: DiagnosticWidget
  Methods:
  - Name: duplicateMethod
    Where:
      Parameters:
      - int
    SwiftName: duplicateMethodA(_:)
  - Name: duplicateMethod
    Where:
      Parameters:
      - int
    SwiftName: duplicateMethodB(_:)
# DUPLICATE: error: multiple API notes entries for C++ method 'duplicateMethod' with Where.Parameters [int]
  - Name: duplicateEmpty
    Where:
      Parameters: []
    SwiftName: duplicateEmptyA()
  - Name: duplicateEmpty
    Where:
      Parameters: []
    SwiftName: duplicateEmptyB()
# DUPLICATE: error: multiple API notes entries for C++ method 'duplicateEmpty' with Where.Parameters []
  - Name: allowed
    SwiftPrivate: true
  - Name: allowed
    Where:
      Parameters:
      - int
    SwiftName: allowedInt(_:)
  - Name: allowed
    Where:
      Parameters:
      - double
    SwiftName: allowedDouble(_:)
