// RUN: rm -rf %t
// RUN: split-file %s %t

// Test that we warn on upwards relative paths in implicitly discovered module maps.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/sub \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.m -fsyntax-only \
// RUN:   -Wmodule-map-path-outside-directory -verify

// Test that we don't warn when the module map is explicitly specified.
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/sub/module.modulemap \
// RUN:   -fmodules-cache-path=%t/cache2 %t/tu-no-warn.m -fsyntax-only \
// RUN:   -Wmodule-map-path-outside-directory -verify

// Test umbrella header with upwards relative path.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/umbrella-header/sub \
// RUN:   -fmodules-cache-path=%t/cache3 %t/umbrella-header/tu.m -fsyntax-only \
// RUN:   -Wmodule-map-path-outside-directory -verify

// Test umbrella directory with upwards relative path.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/umbrella-dir/sub \
// RUN:   -fmodules-cache-path=%t/cache4 %t/umbrella-dir/tu.m -fsyntax-only \
// RUN:   -Wmodule-map-path-outside-directory -verify

// Test that interior .. that escapes the directory is caught.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/interior/sub \
// RUN:   -fmodules-cache-path=%t/cache5 %t/interior/tu.m -fsyntax-only \
// RUN:   -Wmodule-map-path-outside-directory -verify

//--- a.h
void a(void);

//--- sub/module.modulemap
module A {
  header "../a.h"
}

//--- tu.m
@import A;
// expected-warning@*{{path refers outside of the module directory; such paths in implicitly discovered module maps are deprecated}}

//--- tu-no-warn.m
// expected-no-diagnostics
@import A;

//--- umbrella-header/inc/b.h
void b(void);

//--- umbrella-header/sub/module.modulemap
module B {
  umbrella header "../inc/b.h"
}

//--- umbrella-header/tu.m
@import B;
// expected-warning@*{{path refers outside of the module directory; such paths in implicitly discovered module maps are deprecated}}

//--- umbrella-dir/inc/c.h
void c(void);

//--- umbrella-dir/sub/module.modulemap
module C {
  umbrella "../inc"
}

//--- umbrella-dir/tu.m
@import C;
// expected-warning@*{{path refers outside of the module directory; such paths in implicitly discovered module maps are deprecated}}

//--- interior/d.h
void d(void);

//--- interior/sub/inner/placeholder.h

//--- interior/sub/module.modulemap
module D {
  header "inner/../../d.h"
}

//--- interior/tu.m
@import D;
// expected-warning@*{{path refers outside of the module directory; such paths in implicitly discovered module maps are deprecated}}
