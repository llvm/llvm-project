// RUN: rm -rf %t
// RUN: mkdir %t

// Write out a module with errors make sure it can be read
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodules-cache-path=%t -x objective-c -emit-module \
// RUN:   -fmodule-name=error %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fallow-pcm-with-compiler-errors \
// RUN:   -fmodules-cache-path=%t -x objective-c -I %S/Inputs \
// RUN:   -fimplicit-module-maps -ast-print %s | FileCheck %s

// allow-pcm-with-compiler-errors should also allow errors in PCH
// RUN: %clang_cc1 -fallow-pcm-with-compiler-errors -x c++ -emit-pch \
// RUN:   -o %t/check.pch %S/Inputs/error.h

@import error;

void test(id x) {
  [x method];
}

// CHECK: @interface Error
// CHECK-NEXT: - (int)method;
// CHECK-NEXT: @end
// CHECK: void test(id x)
