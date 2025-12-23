// sed command does not work as-is on Windows.
// UNSUPPORTED: system-windows
// RUN: sed -e "s:INPUT_DIR:%S/Inputs/custom-query-check:g" -e "s:OUT_DIR:%t:g" -e "s:MAIN_FILE:%s:g" %S/Inputs/custom-query-check/vfsoverlay.yaml > %t.yaml
// RUN: clang-tidy --allow-no-checks %t/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml | FileCheck %s --check-prefix=CHECK
// RUN: clang-tidy --allow-no-checks %t/main.cpp -checks='-*,custom-*' -vfsoverlay %t.yaml --list-checks | FileCheck %s --check-prefix=LIST-CHECK


long V;
// CHECK: No checks enabled.

void f();
// CHECK-SUB-DIR-APPEND: [[@LINE-1]]:1: warning: find function decl [custom-function-decl]

// LIST-CHECK: Enabled checks:
// LIST-CHECK-EMPTY:
