// RUN: rm -rf %t
// RUN: %clang_cc1 -Wunused-local-typedef -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -fsyntax-only 2>&1 | FileCheck %s -check-prefix=CHECK_1
// RUN: %clang_cc1 -Wunused-local-typedef -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -fsyntax-only 2>&1 | FileCheck %s -check-prefix=CHECK_2 --allow-empty

// For modules, the warning should only fire the first time, when the module is
// built.
// CHECK_1: warning: unused typedef
// CHECK_2-NOT: warning: unused typedef
@import warn_unused_local_typedef;
