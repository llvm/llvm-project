// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb1.json.template > %t/cdb1.json

// RUN: clang-scan-deps -compilation-database %t/cdb1.json -format experimental-full -mode preprocess-dependency-directives > %t/result1.txt

// RUN: FileCheck %s -input-file %t/result1.txt

// Verify that secondary actions get stripped, and that there's a single version
// of module A.

// CHECK:        "modules": [
// CHECK-NEXT:     {
// CHECK:            "name": "A"
// CHECK:          }
// CHECK-NOT:        "name": "A"
// CHECK:        "translation-units"

//--- cdb1.json.template
[
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -fsyntax-only DIR/t1.m",
    "file": "DIR/t1.m"
  },
  {
    "directory": "DIR",
    "command": "clang -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -fsyntax-only DIR/t2.m",
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
