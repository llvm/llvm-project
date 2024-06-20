#ifdef NONEXISTENT
@import load_nonexistent;
#endif

#ifdef FAILURE
@import load_failure;
#endif

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c++ -fmodules-cache-path=%t -fdisable-module-hash -emit-module -fmodule-name=load_failure %S/Inputs/module.modulemap
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -I %S/Inputs -fdisable-module-hash %s -DNONEXISTENT 2>&1 | FileCheck -check-prefix=CHECK-NONEXISTENT %s
// CHECK-NONEXISTENT: load_failure.c:2:9: fatal error: module 'load_nonexistent' not found

// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -I %S/Inputs -fdisable-module-hash %s -DFAILURE 2> %t.out
// RUN: FileCheck -check-prefix=CHECK-FAILURE %s < %t.out

// FIXME: Clean up diagnostic text below and give it a location
// CHECK-FAILURE: error: C99 was disabled in PCH file but is currently enabled
// FIXME: When we have a syntax for modules in C, use that.


