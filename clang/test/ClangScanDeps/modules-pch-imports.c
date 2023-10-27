// Check that a module from -fmodule-name= does not accidentally pick up extra
// dependencies that come from a PCH.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json

// Scan PCH
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

// Build PCH
// RUN: %deps-to-rsp %t/deps_pch.json --module-name A > %t/A.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --module-name B > %t/B.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/B.rsp
// RUN: %clang @%t/pch.rsp

// Scan TU with PCH
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Verify that the only modular import in C is E and not the unrelated modules
// A or B that come from the PCH.

// CHECK:      {
// CHECK-NEXT:  "modules": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK:                "module-name": "E"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NOT:          "-fmodule-file=
// CHECK:              "-fmodule-file={{(E=)?}}[[PREFIX]]/{{.*}}/E-{{.*}}.pcm"
// CHECK-NOT:          "-fmodule-file=
// CHECK:            ]
// CHECK:            "name": "C"
// CHECK:          }


//--- cdb_pch.json.template
[{
  "file": "DIR/prefix.h",
  "directory": "DIR",
  "command": "clang -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -include DIR/prefix.h -fmodule-name=C -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module A { header "A.h" export * }
module B { header "B.h" export * }
module C { header "C.h" export * }
module D { header "D.h" export * }
module E { header "E.h" export * }

//--- A.h
#pragma once
struct A { int x; };

//--- B.h
#pragma once
#include "A.h"
struct B { struct A a; };

//--- C.h
#pragma once
#include "E.h"
struct C { struct E e; };

//--- D.h
#pragma once
#include "C.h"
struct D { struct C c; };

//--- E.h
#pragma once
struct E { int y; };

//--- prefix.h
#include "B.h"

//--- tu.c
// C.h is first included textually due to -fmodule-name=C.
#include "C.h"
// importing D pulls in a modular import of C; it's this build of C that we
// are verifying above
#include "D.h"

void tu(void) {
  struct A a;
  struct B b;
  struct C c;
}
