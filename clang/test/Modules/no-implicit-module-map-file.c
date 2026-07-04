// Test -fno-implicit-module-map-file=: exclude a specific module map from
// *implicit* module-map discovery, without disabling implicit discovery
// globally, and without affecting explicit -fmodule-map-file= loading or other
// implicitly-discovered maps.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Baseline: the module map is discovered implicitly and the #include is
// translated into an import of module 'Foo'.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/include -Rmodule-include-translation -verify=translated %t/use-foo.c

// Excluded: the same map is skipped for implicit discovery, so the #include
// stays textual (no translation) and compiles as an ordinary header.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/include -Rmodule-include-translation \
// RUN:   -fno-implicit-module-map-file=%t/include/module.modulemap \
// RUN:   -verify=textual %t/use-foo.c

// Path robustness: a different spelling of the same file (extra "/./" and a
// "/../") still matches, because matching is by underlying file, not string.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/include -Rmodule-include-translation \
// RUN:   -fno-implicit-module-map-file=%t/include/./sub/../module.modulemap \
// RUN:   -verify=textual %t/use-foo.c

// Explicit still works: even though the map is on the exclusion list, loading
// it explicitly via -fmodule-map-file= makes 'Foo' modular again, so the
// include is translated.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/include -Rmodule-include-translation \
// RUN:   -fno-implicit-module-map-file=%t/include/module.modulemap \
// RUN:   -fmodule-map-file=%t/include/module.modulemap \
// RUN:   -verify=translated %t/use-foo.c

// Other maps unaffected: excluding Foo's map does not stop a different,
// non-excluded map from being discovered and used.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/include -I %t/other -Rmodule-include-translation \
// RUN:   -fno-implicit-module-map-file=%t/include/module.modulemap \
// RUN:   -verify=other %t/use-bar.c

//--- include/module.modulemap
module Foo { header "foo.h" export * }

//--- include/foo.h
void foo(void);

//--- include/sub/.keep

//--- other/module.modulemap
module Bar { header "bar.h" export * }

//--- other/bar.h
void bar(void);

//--- use-foo.c
// translated-remark@+2 {{treating #include as an import of module 'Foo'}}
// textual-no-diagnostics
#include "foo.h"
void test(void) { foo(); }

//--- use-bar.c
// other-remark@+1 {{treating #include as an import of module 'Bar'}}
#include "bar.h"
void test(void) { bar(); }
