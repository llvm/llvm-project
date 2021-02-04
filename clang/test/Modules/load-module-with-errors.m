// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/prebuilt

// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-name=error -o %t/prebuilt/error.pcm \
// RUN:   -x objective-c -emit-module %S/Inputs/error/module.modulemap

// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fprebuilt-module-path=%t/prebuilt -fmodules-cache-path=%t \
// RUN:   -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fprebuilt-module-path=%t/prebuilt -fmodules-cache-path=%t \
// RUN:   -verify=pcherror %s

// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-file=error=%t/prebuilt/error.pcm -fmodules-cache-path=%t \
// RUN:   -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodule-file=error=%t/prebuilt/error.pcm -fmodules-cache-path=%t \
// RUN:   -verify=pcherror %s

// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodule-file=%t/prebuilt/error.pcm -fmodules-cache-path=%t \
// RUN:   -ast-print %s | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodule-file=%t/prebuilt/error.pcm -fmodules-cache-path=%t \
// RUN:   -verify=pcherror %s

// Shouldn't build the cached module (that has errors) when not allowing errors
// RUN: not %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c %s
// RUN: find %t -name "error-*.pcm" | not grep error

// Should build the cached module when allowing errors
// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c -verify %s
// RUN: find %t -name "error-*.pcm" | grep error

// Make sure there is still an error after the module is already in the cache
// RUN: %clang_cc1 -fsyntax-only -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c -verify %s

// Should rebuild the cached module if it had an error (if it wasn't rebuilt
// the verify would fail as it would be the PCH error instead)
// RUN: %clang_cc1 -fsyntax-only -fmodules \
// RUN:   -fmodules-cache-path=%t -fimplicit-module-maps -I %S/Inputs/error \
// RUN:   -x objective-c -verify %s

// allow-pcm-with-compiler-errors should also allow errors in PCH
// RUN: %clang_cc1 -fallow-pcm-with-compiler-errors -x objective-c \
// RUN:   -o %t/check.pch -emit-pch %S/Inputs/error/error.h

// pcherror-error@* {{PCH file contains compiler errors}}
@import error; // expected-error {{could not build module 'error'}}

void test(Error *x) {
  [x method];
}

// CHECK: @interface Error
// CHECK-NEXT: - (int)method;
// CHECK-NEXT: - (id)method2;
// CHECK-NEXT: @end
// CHECK: void test(Error *x)
