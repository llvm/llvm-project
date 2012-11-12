
#ifndef PCH_HEADER
#define PCH_HEADER

#include <DependsOnModule/DependsOnModule.h>
extern int pch_glob;

#else

int glob;

#endif

// RUN: rm -rf %t.cache
// RUN: c-index-test -write-pch %t.h.pch %s -fmodule-cache-path %t.cache -fmodules -F %S/../Modules/Inputs -Xclang -fdisable-module-hash
// RUN: c-index-test -index-file %s -include %t.h -fmodule-cache-path %t.cache -fmodules -F %S/../Modules/Inputs \
// RUN:      -Xclang -fdisable-module-hash | FileCheck %s

// CHECK-NOT: [indexDeclaration]
// CHECK:      [importedASTFile]: {{.*}}.h.pch
// CHECK-NEXT: [enteredMainFile]: {{.*[/\\]}}index-pch-with-module.m
// CHECK-NEXT: [startedTranslationUnit]
// CHECK-NEXT: [indexDeclaration]: kind: variable | name: glob | {{.*}} | loc: 10:5
// CHECK-NOT: [indexDeclaration]

// RUN: c-index-test -index-tu %t.h.pch | FileCheck %s -check-prefix=CHECK-PCH

// CHECK-PCH: [enteredMainFile]: {{.*[/\\]}}index-pch-with-module.m
// CHECK-PCH: [startedTranslationUnit]
// CHECK-PCH: [importedASTFile]: {{.*}}.cache{{[/\\]}}DependsOnModule.pcm | loc: 5:2 | name: "DependsOnModule" | isImplicit: 1
// CHECK-PCH: [indexDeclaration]: kind: variable | name: pch_glob | {{.*}} | loc: 6:12
