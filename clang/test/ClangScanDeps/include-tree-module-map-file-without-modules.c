// Headers covered by a module map loaded with -fmodule-map-file belong to that
// module even when modules are disabled, so the include-tree records a
// submodule name for them. Check the module declaration is recorded too, so
// replaying the include-tree can resolve that name.

// REQUIRES: ondisk_cas
// RUN: rm -rf %t
// RUN: split-file --leading-lines %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas > %t/deps.json
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp

// RUN: FileCheck %s -input-file %t/tu.rsp -check-prefix=RSP
// RSP-NOT: -fmodule-map-file

// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid > %t/tree.txt
// RUN: FileCheck %s -input-file %t/tree.txt -DPREFIX=%/t

// CHECK: [[PREFIX]]{{[/\\]}}tu.c llvmcas://
// CHECK: [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}foo.h llvmcas://
// CHECK-NEXT: Submodule: Foo
// CHECK: Module Map:
// CHECK-NEXT: Foo

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu.c -IDIR/Inputs -fmodule-map-file=DIR/module.modulemap",
    "file": "DIR/tu.c"
  }
]

//--- module.modulemap
module Foo {
  umbrella "Inputs"
  export *
}

//--- Inputs/foo.h
void foo(void);

//--- tu.c
#import "foo.h"
#import "foo.h"
void tu(void) { foo(); }
