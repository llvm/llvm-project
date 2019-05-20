// Sanity-check. Run without modules:
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp %S/Inputs/expand-modular-headers-ppcallbacks/* %t/
// RUN: %check_clang_tidy -std=c++11 %s readability-identifier-naming %t/without-modules -- \
// RUN:   -config="CheckOptions: [{ \
// RUN:      key: readability-identifier-naming.MacroDefinitionCase, value: UPPER_CASE }]" \
// RUN:   -header-filter=.* \
// RUN:   -- -I %t/
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp %S/Inputs/expand-modular-headers-ppcallbacks/* %t/
// RUN: %check_clang_tidy -std=c++17 %s readability-identifier-naming %t/without-modules -- \
// RUN:   -config="CheckOptions: [{ \
// RUN:      key: readability-identifier-naming.MacroDefinitionCase, value: UPPER_CASE }]" \
// RUN:   -header-filter=.* \
// RUN:   -- -I %t/
//
// Run clang-tidy on a file with modular includes:
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp %S/Inputs/expand-modular-headers-ppcallbacks/* %t/
// RUN: %check_clang_tidy -std=c++11 %s readability-identifier-naming %t/with-modules -- \
// RUN:   -config="CheckOptions: [{ \
// RUN:      key: readability-identifier-naming.MacroDefinitionCase, value: UPPER_CASE }]" \
// RUN:   -header-filter=.* \
// RUN:   -- -I %t/ \
// RUN:   -fmodules -fimplicit-modules -fno-implicit-module-maps \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t/module-cache/
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp %S/Inputs/expand-modular-headers-ppcallbacks/* %t/
// RUN: %check_clang_tidy -std=c++17 %s readability-identifier-naming %t/with-modules -- \
// RUN:   -config="CheckOptions: [{ \
// RUN:      key: readability-identifier-naming.MacroDefinitionCase, value: UPPER_CASE }]" \
// RUN:   -header-filter=.* \
// RUN:   -- -I %t/ \
// RUN:   -fmodules -fimplicit-modules -fno-implicit-module-maps \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t/module-cache/
// FIXME: Make the test work in all language modes.
#include "c.h"

// CHECK-MESSAGES: a.h:1:9: warning: invalid case style for macro definition 'a' [readability-identifier-naming]
// CHECK-MESSAGES: a.h:1:9: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES: b.h:2:9: warning: invalid case style for macro definition 'b'
// CHECK-MESSAGES: b.h:2:9: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES: c.h:2:9: warning: invalid case style for macro definition 'c'
// CHECK-MESSAGES: c.h:2:9: note: FIX-IT applied suggested code changes

#define m
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for macro definition 'm'
// CHECK-MESSAGES: :[[@LINE-2]]:9: note: FIX-IT applied suggested code changes
