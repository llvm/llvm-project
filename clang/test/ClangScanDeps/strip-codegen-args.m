// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb1.json.template > %t/cdb1.json

// RUN: clang-scan-deps -compilation-database %t/cdb1.json -format experimental-full > %t/result1.txt
// RUN: FileCheck %s -input-file %t/result1.txt

// This tests that codegen option that do not affect the AST or generation of a
// module are removed. It also tests that the optimization options that affect
// the AST are not reset to -O0.

// CHECK:        "modules": [
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK-NOT:          "-O0"
// CHECK-NOT:          "-flto"
// CHECK-NOT:          "-fno-autolink"
// CHECK-NOT:          "-mrelax-relocations=no"
// CHECK:            ]
// CHECK:            "name": "A"
// CHECK:          }
// CHECK-NOT:        "name": "A"
// CHECK:        "translation-units"

//--- cdb1.json.template
[
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -O2 -flto -fno-autolink -Xclang -mrelax-relocations=no -fsyntax-only DIR/t1.m",
    "file": "DIR/t1.m"
  },
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -O2 -flto=thin -fautolink -fsyntax-only DIR/t2.m",
    "file": "DIR/t2.m"
  },
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -O2 -flto=full -fsyntax-only DIR/t3.m",
    "file": "DIR/t2.m"
  }
]

//--- modules/A/module.modulemap

module A {
  umbrella header "A.h"
}

//--- modules/A/A.h

typedef int A_t;

//--- t1.m
@import A;

//--- t2.m
@import A;

//--- t3.m
@import A;
