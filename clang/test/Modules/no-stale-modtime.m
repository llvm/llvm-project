// Ensure that when rebuilding a module we don't save its old modtime when
// building modules that depend on it.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: cat %t/t.h-1 > %t/t.h

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %t/main.m -Rmodule-build 2>&1 \
// RUN: | FileCheck -check-prefix=REBUILD-ALL %t/main.m
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %t/main.m -Rmodule-build -verify

// Add an identifier to ensure everything depending on t is out of date
// RUN: cat %t/t.h-1 %t/t.h-2 > %t/t.h

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %t/main.m -Rmodule-build 2>&1 \
// RUN: | FileCheck -check-prefix=REBUILD-ALL %t/main.m
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -I %t -fsyntax-only %t/main.m -Rmodule-build -verify

//--- b.h
@import l; @import r;

//--- l.h
@import t; // fromt l

//--- r.h
@import t; // fromt r

//--- t.h-1
// top

//--- t.h-2
extern int a;

//--- module.modulemap
module b { header "b.h" } module l { header "l.h" }
module r { header "r.h" } module t { header "t.h" }

//--- main.m

// REBUILD-ALL: building module 'b'
// REBUILD-ALL: building module 'l'
// REBUILD-ALL: building module 't'
// REBUILD-ALL: building module 'r'

// Use -verify when expecting no modules to be rebuilt.
// expected-no-diagnostics
@import b;
