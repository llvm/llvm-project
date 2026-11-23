// RUN: rm -rf %t
// RUN: split-file %s %t

// Test that we warn when two modules own the same header (at include time).
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/headers \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

// Test that the warning is off by default.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/headers \
// RUN:   -fmodules-cache-path=%t/cache2 %t/tu-no-warn.c -fsyntax-only -verify

// Test that no warning fires if the conflicting header is not included.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/headers \
// RUN:   -fmodules-cache-path=%t/cache3 %t/tu-no-include.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

// Test umbrella header in different module conflicts with explicit header.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/umbrella \
// RUN:   -fmodules-cache-path=%t/cache4 %t/tu-umbrella.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

// Test umbrella dir in different module conflicts with explicit header.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/umbrella-dir \
// RUN:   -fmodules-cache-path=%t/cache5 %t/tu-umbrella-dir.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

// Test that explicit header + umbrella in the SAME module is fine.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/same-module \
// RUN:   -fmodules-cache-path=%t/cache6 %t/tu-same-module.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

// Test that excluded header under another module's umbrella is fine.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/excluded \
// RUN:   -fmodules-cache-path=%t/cache7 %t/tu-excluded.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

// Test umbrella header in subdirectory claiming parent dir conflicts with
// explicit header in parent dir.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/parent-umbrella \
// RUN:   -fmodules-cache-path=%t/cache8 %t/tu-parent-umbrella.c -fsyntax-only \
// RUN:   -Wduplicate-header-ownership -verify

//--- headers/a.h

//--- headers/b.h

//--- headers/module.modulemap
module A {
  header "a.h"
}

module B {
  header "a.h"
}

module C {
  header "b.h"
}

//--- tu.c
#include "a.h"
// expected-warning@-1 {{header 'a.h' is owned by multiple modules}}
// expected-note@* {{header owned by module 'A' here}}
// expected-note@* {{header owned by module 'B' here}}

//--- tu-no-warn.c
// expected-no-diagnostics
#include "a.h"

//--- tu-no-include.c
// expected-no-diagnostics
#include "b.h"

//--- umbrella/umbrella.h
#include "d.h"

//--- umbrella/d.h

//--- umbrella/module.modulemap
module D {
  umbrella header "umbrella.h"
}

module E {
  header "d.h"
}

//--- tu-umbrella.c
#include "d.h"
// expected-warning@-1 {{header 'd.h' is owned by multiple modules}}
// expected-note@* {{header owned by module 'E' here}}
// expected-note@* {{header covered by umbrella for module 'D' here}}

//--- umbrella-dir/sub/e.h

//--- umbrella-dir/module.modulemap
module F {
  umbrella "sub"
}

module G {
  header "sub/e.h"
}

//--- tu-umbrella-dir.c
#include "sub/e.h"
// expected-warning@-1 {{header 'sub/e.h' is owned by multiple modules}}
// expected-note@* {{header owned by module 'G' here}}
// expected-note@* {{header covered by umbrella for module 'F' here}}

//--- same-module/sub/f.h

//--- same-module/module.modulemap
module H {
  umbrella "sub"
  header "sub/f.h"
}

//--- tu-same-module.c
// expected-no-diagnostics
#include "sub/f.h"

//--- excluded/sub/g.h

//--- excluded/module.modulemap
module I {
  umbrella "sub"
  exclude header "sub/g.h"
}

module J {
  header "sub/g.h"
}

//--- tu-excluded.c
// expected-no-diagnostics
#include "sub/g.h"

//--- parent-umbrella/umb.h
//--- parent-umbrella/h.h
//--- parent-umbrella/sub/s.h

//--- parent-umbrella/sub/module.modulemap
module Umb {
  umbrella header "../umb.h"
}

//--- parent-umbrella/module.modulemap
module K {
  header "h.h"
}

//--- tu-parent-umbrella.c
#include "sub/s.h"
#include "h.h"
// expected-warning@-1 {{header 'h.h' is owned by multiple modules}}
// expected-note@* {{header owned by module 'K' here}}
// expected-note@* {{header covered by umbrella for module 'Umb' here}}
