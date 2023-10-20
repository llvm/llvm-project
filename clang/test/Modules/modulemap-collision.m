// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t/sources %t/build
// RUN: echo "// A.h" > %t/sources/A.h
// RUN: echo "framework module A {}" > %t/sources/module.modulemap
// RUN: echo "framework module A.Private { umbrella header \"A.h\" }" > %t/sources/module.private.modulemap
// RUN: cp %t/sources/module.modulemap %t/build/module.modulemap
// RUN: cp %t/sources/module.private.modulemap %t/build/module.private.modulemap

// RUN: sed -e "s:DUMMY_DIR:%t:g" %S/Inputs/all-product-headers.yaml > %t/build/all-product-headers.yaml
// RUN: %clang_cc1 -fsyntax-only -ivfsoverlay %t/build/all-product-headers.yaml -F%t/build -fmodules -fimplicit-module-maps -Wno-private-module -fmodules-cache-path=%t/cache -x objective-c %s -verify

// expected-no-diagnostics
#import <A/A.h>
