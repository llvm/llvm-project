// A parameter takes API notes from its function's entry, never from a lookup
// of its own. While a parameter's attributes are processed its DeclContext is
// still the translation unit, so a lookup would find a global variable of the
// same name and give the parameter that global's notes.

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -fdisable-module-hash -fapinotes-modules -I %t %t/use.c -ast-dump -ast-dump-filter takesShadow -x c | FileCheck --check-prefix=PARAM %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -fdisable-module-hash -fapinotes-modules -I %t %t/use.c -ast-dump -ast-dump-filter shadow -x c | FileCheck --check-prefix=GLOBAL %s

// PARAM: Dumping takesShadow:
// PARAM-NEXT: FunctionDecl {{.*}} takesShadow 'void (int *)'
// PARAM-NEXT: ParmVarDecl {{.*}} shadow 'int *'
// PARAM-EMPTY:

// The global keeps its own notes.
// GLOBAL: Dumping shadow:
// GLOBAL-NEXT: VarDecl {{.*}} shadow 'int *'
// GLOBAL-NEXT: SwiftPrivateAttr

//--- module.modulemap
module Shadow { header "Shadow.h" export * }

//--- Shadow.apinotes
Name: Shadow
Globals:
  - Name: shadow
    SwiftPrivate: true

//--- Shadow.h
extern int *shadow;
void takesShadow(int *shadow);

//--- use.c
#include "Shadow.h"
void caller(void) { takesShadow(shadow); }
