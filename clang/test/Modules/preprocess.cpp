// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x c++ -E %s | \
// RUN:   FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x objective-c -E %s | \
// RUN:   FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x c++ -E -frewrite-includes %s | \
// RUN:   FileCheck -strict-whitespace %s --check-prefix=REWRITE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x objective-c -E -frewrite-includes %s | \
// RUN:   FileCheck -strict-whitespace %s --check-prefix=REWRITE
#include "dummy.h"
#include "dummy.h"
foo bar baz

// EOF marker to ensure -frewrite-includes doesn't match its own CHECK lines.

// REWRITE: #if 0
// REWRITE: #include{{ }}"dummy.h"
// REWRITE: #endif

// CHECK: #pragma clang module import dummy /* clang {{.*}} implicit import

// REWRITE: #if 0
// REWRITE: #include{{ }}"dummy.h"
// REWRITE: #endif

// CHECK: #pragma clang module import dummy /* clang {{.*}} implicit import

// CHECK: foo bar baz

// REWRITE: // {{EOF}} marker
