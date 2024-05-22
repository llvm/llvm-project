// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -mode preprocess-dependency-directives > %t/result.txt

// RUN: FileCheck %s -input-file %t/result.txt

// Verify that there's a single version of module A.

// CHECK:        "modules": [
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK-NOT:          "-fvisibility="
// CHECK-NOT:          "-ftype-visibility="
// CHECK:            ]
// CHECK:            "name": "A"
// CHECK:          }
// CHECK-NOT:        "name": "A"
// CHECK:        "translation-units"

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -fsyntax-only DIR/t1.c",
    "file": "DIR/t1.c"
  },
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -fvisibility=hidden -fsyntax-only DIR/t2.c",
    "file": "DIR/t2.c"
  },
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -fvisibility=hidden -fvisibility-ms-compat -fsyntax-only DIR/t3.c",
    "file": "DIR/t3.c"
  }
]

//--- modules/A/module.modulemap

module A {
  umbrella header "A.h"
}

//--- modules/A/A.h

typedef int A_t;
extern int a(void);

//--- t1.c
#include "A.h"

//--- t2.c
#include "A.h"

//--- t3.c
#include "A.h"
