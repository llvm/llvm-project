// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -fvisibility hidden  %s | FileCheck --check-prefix=CHECK-CMD-HIDDEN %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -fvisibility hidden %s | FileCheck --check-prefix=CHECK-CMD-HIDDEN %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck --check-prefix=CHECK-CMD %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck --check-prefix=CHECK-CMD %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck --check-prefix=CHECK-CMD2 %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck --check-prefix=CHECK-CMD2 %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | llvm-readelf -s - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s

// Always Be Hidden:
// CHECK-CMD-HIDDEN-NOT: _Z6hiddenv
// CHECK-CMD2-NOT: _Z6hiddenv
__attribute__((visibility("hidden"))) void hidden() {}

// Always Be Visible:
// CHECK-CMD-HIDDEN: _Z9nothiddenv
// CHECK-CMD-DAG: _Z9nothiddenv
__attribute__((visibility("default"))) void nothidden() {}

// Do Whatever -fvisibility says:
// CHECK-CMD-HIDDEN-NOT: _Z10cmdVisiblev
// CHECK-CMD-DAG: _Z10cmdVisiblev
void cmdVisible() {}

// CHECK-SYMBOLS-DAG: DEFAULT    {{.*}} _Z10cmdVisiblev
// CHECK-SYMBOLS-DAG: HIDDEN     {{.*}} _Z6hiddenv
// CHECK-SYMBOLS-DAG: DEFAULT    {{.*}} _Z9nothiddenv
