// RUN: rm -rf %t && mkdir %t

// Check that search paths used by `#include` and `#include_next` are reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage   \
// RUN:   -I%S/Inputs/search-path-usage/a          \
// RUN:   -I%S/Inputs/search-path-usage/a_next     \
// RUN:   -I%S/Inputs/search-path-usage/b          \
// RUN:   -I%S/Inputs/search-path-usage/c          \
// RUN:   -I%S/Inputs/search-path-usage/d          \
// RUN:   -DINCLUDE -verify
#ifdef INCLUDE
#include "a.h" // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/a'}} \
// expected-remark-re@#a-include-next {{search path used: '{{.*}}/search-path-usage/a_next'}}
#include "d.h" // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/d'}}
#endif

// Check that framework search paths are reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -F%S/Inputs/search-path-usage/FwA      \
// RUN:   -F%S/Inputs/search-path-usage/FwB      \
// RUN:   -DFRAMEWORK -verify
#ifdef FRAMEWORK
#include "FrameworkA/FrameworkA.h" // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/FwA'}}
#endif

// Check that system search paths are reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -isystem %S/Inputs/search-path-usage/b \
// RUN:   -isystem %S/Inputs/search-path-usage/d \
// RUN:   -DSYSTEM -verify
#ifdef SYSTEM
#include "b.h" // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/b'}}
#endif

// Check that sysroot-based search paths are reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -isysroot %S/Inputs/search-path-usage  \
// RUN:   -iwithsysroot /b                       \
// RUN:   -iwithsysroot /d                       \
// RUN:   -DSYSROOT -verify
#ifdef SYSROOT
#include "d.h" // \
// expected-remark {{search path used: '/d'}}
#endif

// Check that search paths used by `__has_include()` are reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -I%S/Inputs/search-path-usage/b        \
// RUN:   -I%S/Inputs/search-path-usage/d        \
// RUN:   -DHAS_INCLUDE -verify
#ifdef HAS_INCLUDE
#if __has_include("b.h") // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/b'}}
#endif
#if __has_include("x.h")
#endif
#endif

// Check that search paths used by `#import` are reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -I%S/Inputs/search-path-usage/b        \
// RUN:   -I%S/Inputs/search-path-usage/d        \
// RUN:   -DIMPORT -verify
#ifdef IMPORT
#import "d.h" // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/d'}}
#endif

// Check that used header maps are reported when the target file exists.
//
// RUN: sed "s|DIR|%/S/Inputs/search-path-usage|g" \
// RUN:             %S/Inputs/search-path-usage/b.hmap.json.template > %t/b.hmap.json
// RUN: %hmaptool write %t/b.hmap.json %t/b.hmap
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -I %t/b.hmap                           \
// RUN:   -I b                                   \
// RUN:   -DHMAP -verify
#ifdef HMAP
#include "b.h" // \
// expected-remark-re {{search path used: '{{.*}}/b.hmap'}}
#endif

// Check that unused header map are not reported.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -I%t/b.hmap                            \
// RUN:   -I%S/Inputs/search-path-usage/d        \
// RUN:   -DHMAP_NO_MATCH -verify
#ifdef HMAP_NO_MATCH
#include "d.h" // \
// expected-remark-re {{search path used: '{{.*}}/search-path-usage/d'}}
#endif

// Check that used header map is reported even when the target file is missing.
//
// RUN: sed "s|DIR|%/S/Inputs/search-path-usage/missing-subdir|g" \
// RUN:             %S/Inputs/search-path-usage/b.hmap.json.template > %t/b-missing.hmap.json
// RUN: %hmaptool write %t/b-missing.hmap.json %t/b-missing.hmap
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -I %t/b-missing.hmap                   \
// RUN:   -I b                                   \
// RUN:   -DHMAP_MATCHED_BUT_MISSING -verify
#ifdef HMAP_MATCHED_BUT_MISSING
#include "b.h" // \
// expected-remark-re {{search path used: '{{.*}}/b-missing.hmap'}} \
// expected-error {{'b.h' file not found}}
#endif

// Check that used header map is reported even when the target file is missing
// and the lookup is initiated by __has_include.
//
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage \
// RUN:   -I %t/b-missing.hmap                   \
// RUN:   -I b                                   \
// RUN:   -DHMAP_MATCHED_BUT_MISSING_IN_HAS_INCLUDE -verify
#ifdef HMAP_MATCHED_BUT_MISSING_IN_HAS_INCLUDE
#if __has_include("b.h") // \
// expected-remark-re {{search path used: '{{.*}}/b-missing.hmap'}}
#endif
#endif

// Check that search paths with module maps are NOT reported.
//
// RUN: mkdir %t/modulemap_abs
// RUN: sed "s|DIR|%/S/Inputs/search-path-usage|g"                            \
// RUN:   %S/Inputs/search-path-usage/modulemap_abs/module.modulemap.template \
// RUN:     > %t/modulemap_abs/module.modulemap
// RUN: %clang_cc1 -Eonly %s -Rsearch-path-usage                           \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules \
// RUN:   -I %t/modulemap_abs                                              \
// RUN:   -I %S/Inputs/search-path-usage/a                                 \
// RUN:   -DMODMAP_ABS -verify
#ifdef MODMAP_ABS
@import b; // \
// expected-no-diagnostics
#endif
