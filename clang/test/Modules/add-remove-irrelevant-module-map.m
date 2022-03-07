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
// RUN: %clang_cc1 -module-file-info %t/cache-without-b/a.pcm | FileCheck %s --check-prefix=CHECK-B
// RUN: %clang_cc1 -module-file-info %t/cache-with-b/a.pcm | FileCheck %s --check-prefix=CHECK-B
// CHECK-B-NOT: Input file: {{.*}}b.modulemap

//--- c.modulemap
module c [no_undeclared_includes] { header "c.h" }

//--- c.h
#if __has_include("d.h") // This should use 'd.modulemap' in order to determine that 'd.h'
                         // doesn't exist for 'c' because of its '[no_undeclared_includes]'.
#endif

//--- d.modulemap
module d { header "d.h" }

//--- d.h
// empty

//--- test-no-undeclared-includes.m
// expected-no-diagnostics
@import c;

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fdisable-module-hash \
// RUN:   -fmodule-map-file=%t/c.modulemap -fmodule-map-file=%t/d.modulemap \
// RUN:   %t/test-no-undeclared-includes.m -verify

// The PCM file considers 'd.modulemap' an input because it affects the compilation,
// although it doesn't describe the built module or its imports.
//
// RUN: %clang_cc1 -module-file-info %t/cache/c.pcm | FileCheck %s --check-prefix=CHECK-D
// CHECK-D: Input file: {{.*}}d.modulemap
