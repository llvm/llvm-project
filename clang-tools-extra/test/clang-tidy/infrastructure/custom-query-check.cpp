// REQUIRES: shell
// REQUIRES: custom-check

// RUN: sed -e "s:INPUT_DIR:%S/Inputs/custom-query-check:g" -e "s:OUT_DIR:%t:g" -e "s:MAIN_FILE:%s:g" %S/Inputs/custom-query-check/vfsoverlay.yaml > %t.yaml
// RUN: clang-tidy --experimental-custom-checks %t/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml | FileCheck %s --check-prefix=CHECK-SAME-DIR
// RUN: clang-tidy --experimental-custom-checks %t/subdir/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml | FileCheck %s --check-prefix=CHECK-SUB-DIR-BASE
// RUN: clang-tidy --experimental-custom-checks %t/subdir-override/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml | FileCheck %s --check-prefix=CHECK-SUB-DIR-OVERRIDE
// RUN: clang-tidy --experimental-custom-checks %t/subdir-empty/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml --allow-no-checks | FileCheck %s --check-prefix=CHECK-SUB-DIR-EMPTY
// RUN: clang-tidy --experimental-custom-checks %t/subdir-append/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml | FileCheck %s --check-prefix=CHECK-SUB-DIR-APPEND
// RUN: clang-tidy --experimental-custom-checks %t/subdir-append/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml --list-checks | FileCheck %s --check-prefix=LIST-CHECK
// RUN: clang-tidy --experimental-custom-checks %t/subdir-append/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml --dump-config | FileCheck %s --check-prefix=DUMP-CONFIG


long V;
// CHECK-SAME-DIR: [[@LINE-1]]:1: warning: use 'int' instead of 'long' [custom-avoid-long-type]
// CHECK-SUB-DIR-BASE: [[@LINE-2]]:1: warning: use 'int' instead of 'long' [custom-avoid-long-type]
// CHECK-SUB-DIR-OVERRIDE: [[@LINE-3]]:1: warning: use 'int' instead of 'long' override [custom-avoid-long-type]
// CHECK-SUB-DIR-EMPTY: No checks enabled.
// CHECK-SUB-DIR-APPEND: [[@LINE-5]]:1: warning: use 'int' instead of 'long' [custom-avoid-long-type]

void f();
// CHECK-SUB-DIR-APPEND: [[@LINE-1]]:1: warning: find function decl [custom-function-decl]

// LIST-CHECK: Enabled checks:
// LIST-CHECK:   custom-avoid-long-type
// LIST-CHECK:   custom-function-decl

// DUMP-CONFIG: CustomChecks:
// DUMP-CONFIG: - Name: avoid-long-type
// DUMP-CONFIG:   Query: |
// DUMP-CONFIG:     match varDecl(
// DUMP-CONFIG:       hasType(asString("long")),
// DUMP-CONFIG:       hasTypeLoc(typeLoc().bind("long"))
// DUMP-CONFIG:     )
// DUMP-CONFIG:   Diagnostic:
// DUMP-CONFIG:    - BindName: long
// DUMP-CONFIG:      Message: |
// DUMP-CONFIG:           use 'int' instead of 'long'
// DUMP-CONFIG:      Level:           Warning
// DUMP-CONFIG: - Name: function-decl
// DUMP-CONFIG:   Query: |
// DUMP-CONFIG:     match functionDecl().bind("func")
// DUMP-CONFIG:   Diagnostic:
// DUMP-CONFIG:    - BindName: func
// DUMP-CONFIG:      Message: |
// DUMP-CONFIG:        find function decl
// DUMP-CONFIG:   Level: Warning
