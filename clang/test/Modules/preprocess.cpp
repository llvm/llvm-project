// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x c++ -E %s | \
// RUN:   FileCheck -strict-whitespace %s --check-prefix=CHECK --check-prefix=CXX --check-prefix=CXX-DASHE
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x objective-c -E %s | \
// RUN:   FileCheck -strict-whitespace %s --check-prefix=CHECK --check-prefix=OBJC
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x c++ -E -frewrite-includes %s | \
// RUN:   FileCheck -strict-whitespace %s --check-prefix=CHECK --check-prefix=CXX
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x objective-c -E -frewrite-includes %s | \
// RUN:   FileCheck -strict-whitespace %s --check-prefix=CHECK --check-prefix=OBJC
#include "dummy.h"
#include "dummy.h"
foo bar baz

// The weird {{ }} here is to prevent the -frewrite-includes test from matching its own CHECK lines.

// CXX: #include{{ }}"dummy.h"
// CXX-DASHE-SAME: /* clang -E: implicit import for module dummy */
// CXX: #include{{ }}"dummy.h"
// CXX-DASHE-SAME: /* clang -E: implicit import for module dummy */
// CXX: foo bar baz

// OBJC: @import{{ }}dummy; /* clang 
// OBJC: @import{{ }}dummy; /* clang 
// OBJC: foo bar baz
