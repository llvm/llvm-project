// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %hmaptool write %S/Inputs/framework-public-includes-private/a.hmap.json %t/a.hmap
// RUN: %hmaptool write %S/Inputs/framework-public-includes-private/z.hmap.json %t/z.hmap

// RUN: sed -e "s@TEST_DIR@%{/S:regex_replacement}/Inputs/framework-public-includes-private@g" \
// RUN:   %S/Inputs/framework-public-includes-private/z.yaml > %t/z.yaml

// The output with and without modules should be the same, without modules first.
// RUN: %clang_cc1 \
// RUN:   -iquote %t/z.hmap -iquote %t/a.hmap -ivfsoverlay %t/z.yaml \
// RUN:   -F%S/Inputs/framework-public-includes-private \
// RUN:   -fsyntax-only %s -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -iquote %t/z.hmap -iquote %t/a.hmap -ivfsoverlay %t/z.yaml \
// RUN:   -F%S/Inputs/framework-public-includes-private \
// RUN:   -fsyntax-only %s \
// RUN:   2>%t/stderr

// The same warnings show up when modules is on but -verify doesn't get it
// because they only show up under the module A building context.
// RUN: FileCheck --input-file=%t/stderr %s
// CHECK: public framework header includes private framework header 'A/APriv.h'
// CHECK: public framework header includes private framework header 'A/APriv2.h'
// CHECK: public framework header includes private framework header 'Z/ZPriv.h'

#import "A.h"

int bar(void) { return foo(); }

// expected-warning@Inputs/framework-public-includes-private/A.framework/Headers/A.h:1{{public framework header includes private framework header 'A/APriv.h'}}
// expected-warning@Inputs/framework-public-includes-private/A.framework/Headers/A.h:2{{public framework header includes private framework header 'A/APriv2.h'}}
// expected-warning@Inputs/framework-public-includes-private/flat-header-path/Z.h:2{{public framework header includes private framework header 'Z/ZPriv.h'}}
