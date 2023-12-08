// Ensure that when rebuilding a module we don't save its old modtime when
// building modules that depend on it.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// This could be replaced by diamond_*, except we want to modify the top header
// RUN: echo '@import l; @import r;' > %t/b.h
// RUN: echo '@import t; // fromt l' > %t/l.h
// RUN: echo '@import t; // fromt r' > %t/r.h

// RUN: echo '// top' > %t/t.h-1
// RUN: cat %t/t.h-1 > %t/t.h

// RUN: echo 'module b { header "b.h" } module l { header "l.h" }' > %t/module.map-1
// RUN: echo 'module r { header "r.h" } module t { header "t.h" }' > %t/module.map-2
// RUN: cat %t/module.map-1 %t/module.map-2 > %t/module.map

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %s -Rmodule-build 2>&1 \
// RUN: | FileCheck -check-prefix=REBUILD-ALL %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %s -Rmodule-build -verify

// Add an identifier to ensure everything depending on t is out of date
// RUN: echo 'extern int a;' > %t/t.h-2
// RUN: cat %t/t.h-1 %t/t.h-2 > %t/t.h

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %s -Rmodule-build 2>&1 \
// RUN: | FileCheck -check-prefix=REBUILD-ALL %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %s -Rmodule-build -verify

// REBUILD-ALL: building module 'b'
// REBUILD-ALL: building module 'l'
// REBUILD-ALL: building module 't'
// REBUILD-ALL: building module 'r'

// Use -verify when expecting no modules to be rebuilt.
// expected-no-diagnostics

@import b;
