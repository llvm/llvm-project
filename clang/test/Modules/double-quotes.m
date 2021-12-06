// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %hmaptool write %S/Inputs/double-quotes/a.hmap.json %t/a.hmap
// RUN: %hmaptool write %S/Inputs/double-quotes/x.hmap.json %t/x.hmap

// RUN: sed -e "s@TEST_DIR@%{/S:regex_replacement}/Inputs/double-quotes@g" \
// RUN:   %S/Inputs/double-quotes/z.yaml > %t/z.yaml

// The output with and without modules should be the same

// RUN: %clang_cc1 \
// RUN:   -I %t/x.hmap -iquote %t/a.hmap -ivfsoverlay %t/z.yaml \
// RUN:   -F%S/Inputs/double-quotes -I%S/Inputs/double-quotes \
// RUN:   -Wquoted-include-in-framework-header -fsyntax-only %s -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I %t/x.hmap -iquote %t/a.hmap -ivfsoverlay %t/z.yaml \
// RUN:   -F%S/Inputs/double-quotes -I%S/Inputs/double-quotes \
// RUN:   -Wquoted-include-in-framework-header -fsyntax-only %s \
// RUN:   2>%t/stderr

// The same warnings show up when modules is on but -verify doesn't get it
// because they only show up under the module A building context.
// RUN: FileCheck --input-file=%t/stderr %s
// CHECK: double-quoted include "A0.h" in framework header, expected angle-bracketed instead
// CHECK: #include "A0.h"
// CHECK:          ^~~~~~
// CHECK: <A/A0.h>
// CHECK: double-quoted include "B.h" in framework header, expected angle-bracketed instead
// CHECK: #include "B.h"
// CHECK:          ^~~~~
// CHECK: <B.h>
// CHECK: double-quoted include "B.h" in framework header, expected angle-bracketed instead
// CHECK: #import "B.h" // Included from Z.h & A.h
// CHECK:         ^~~~~
// CHECK: <B.h>

#import "A.h"
#import <Z/Z.h>

// Make sure we correctly handle paths that resemble frameworks, but aren't.
#import "NotAFramework/Headers/Headers/Thing1.h"

int bar() { return foo(); }

// expected-warning@Inputs/double-quotes/A.framework/Headers/A.h:1{{double-quoted include "A0.h" in framework header, expected angle-bracketed instead}}
// expected-warning@Inputs/double-quotes/A.framework/Headers/A.h:2{{double-quoted include "B.h" in framework header, expected angle-bracketed instead}}
// expected-warning@Inputs/double-quotes/flat-header-path/Z.h:1{{double-quoted include "B.h" in framework header, expected angle-bracketed instead}}
