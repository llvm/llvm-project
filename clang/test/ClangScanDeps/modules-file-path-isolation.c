// Ensure that the spelling of a path seen outside a module (e.g. header via
// symlink) does not leak into the compilation of that module unnecessarily.
// Note: the spelling of the modulemap path still depends on the includer, since
// that is the only source of information about it.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: ln -s A.h %t/Z.h

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 -format experimental-full \
// RUN:   -mode preprocess-dependency-directives > %t/output
// RUN: FileCheck %s < %t/output

// CHECK:      "modules": [
// CHECK-NEXT:   {
// CHECK:          "file-deps": [
// CHECK-NEXT:       "{{.*}}module.modulemap",
// CHECK-NEXT:       "{{.*}}A.h"
// CHECK-NEXT:     ],
// CHECK-NEXT:     "link-libraries": [],
// CHECK-NEXT:     "name": "A"
// CHECK-NEXT:   }

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps",
  "file": "DIR/tu.c"
}]

//--- module.modulemap
module A { header "A.h" }
module B { header "B.h" }
module C { header "C.h" }

//--- A.h

//--- B.h
#include "Z.h"

//--- tu.c
#include "B.h"
