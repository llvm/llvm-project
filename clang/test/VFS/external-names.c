// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" -e "s@EXTERNAL_NAMES@true@" %S/Inputs/use-external-names.yaml > %t.external.yaml
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" -e "s@EXTERNAL_NAMES@false@" %S/Inputs/use-external-names.yaml > %t.yaml

#include "external-names.h"
#ifdef REINCLUDE
#include "external-names.h"
#endif

////
// Preprocessor (__FILE__ macro and # directives):

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.external.yaml -E %s | FileCheck -check-prefix=CHECK-PP-EXTERNAL %s
// CHECK-PP-EXTERNAL: # {{[0-9]*}} "[[NAME:.*Inputs(/|\\\\)external-names.h]]"
// CHECK-PP-EXTERNAL-NEXT: void foo(char **c) {
// CHECK-PP-EXTERNAL-NEXT: *c = "[[NAME]]";

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.yaml -E %s | FileCheck -check-prefix=CHECK-PP %s
// CHECK-PP-NOT: Inputs

////
// Diagnostics:

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.external.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-DIAG-EXTERNAL %s
// CHECK-DIAG-EXTERNAL: {{.*}}Inputs{{..?}}external-names.h:{{[0-9]*:[0-9]*}}: warning: initializing 'const char **'

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.yaml -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-DIAG %s
// CHECK-DIAG-NOT: Inputs

////
// Debug info

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.external.yaml -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-DEBUG-EXTERNAL %s
// CHECK-DEBUG-EXTERNAL: ![[Num:[0-9]+]] = !DIFile(filename: "{{[^"]*}}Inputs{{..?}}external-names.h"
// CHECK-DEBUG-EXTERNAL: !DISubprogram({{.*}}file: ![[Num]]

// RUN: %clang_cc1 -I %t -ivfsoverlay %t.yaml -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-DEBUG %s
// CHECK-DEBUG-NOT: Inputs

////
// Dependency file

// RUN: %clang_cc1 -D REINCLUDE -I %t -ivfsoverlay %t.external.yaml -Eonly %s -MTfoo -dependency-file %t.external.dep
// RUN: echo "EOF" >> %t.external.dep
// RUN: cat %t.external.dep | FileCheck --check-prefix=CHECK-DEP-EXTERNAL %s
// CHECK-DEP-EXTERNAL: Inputs{{..?}}external-names.h
// CHECK-DEP-EXTERNAL-NEXT: EOF

// RUN: %clang_cc1 -D REINCLUDE -I %t -ivfsoverlay %t.yaml -Eonly %s -MTfoo -dependency-file %t.dep
// RUN: cat %t.dep | FileCheck --check-prefix=CHECK-DEP %s
// CHECK-DEP: Inputs{{..?}}external-names.h
