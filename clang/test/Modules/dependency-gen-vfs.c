// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module M { header "m.h" }

//--- m-real.h

//--- overlay.json.template
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "external-contents": "DIR/m-real.h",
      "name": "DIR/m.h",
      "type": "file"
    }
  ]
}

//--- tu.c
#include "m.h"

// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -ivfsoverlay %t/overlay.json -dependency-file %t/tu.d -MT %t/tu.o -fsyntax-only %t/tu.c
// RUN: FileCheck %s --input-file=%t/tu.d
// CHECK:      {{.*}}tu.o: \
// CHECK-NEXT: {{.*}}tu.c \
// CHECK-NEXT: {{.*}}module.modulemap \
// CHECK-NEXT: {{.*}}m-real.h
