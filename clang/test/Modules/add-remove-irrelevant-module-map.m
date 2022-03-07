// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- a.modulemap
module a {}

//--- b.modulemap
module b {}

//--- test-simple.m
// expected-no-diagnostics
@import a;

// Build without b.modulemap:
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -fdisable-module-hash \
// RUN:   -fmodule-map-file=%t/a.modulemap %t/test-simple.m -verify
// RUN: mv %t/cache %t/cache-without-b

// Build with b.modulemap:
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -fdisable-module-hash \
// RUN:   -fmodule-map-file=%t/a.modulemap -fmodule-map-file=%t/b.modulemap %t/test-simple.m -verify
// RUN: mv %t/cache %t/cache-with-b

// Neither PCM file considers 'b.modulemap' an input:
//
// RUN: %clang_cc1 -module-file-info %t/cache-without-b/a.pcm | FileCheck %s
// RUN: %clang_cc1 -module-file-info %t/cache-with-b/a.pcm | FileCheck %s
// CHECK-NOT: Input file: {{.*}}/b.modulemap
