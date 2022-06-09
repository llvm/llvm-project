// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/compile-commands.json.in > %t/compile-commands.json

// RUN: clang-scan-deps -compilation-database %t/compile-commands.json -j 1 -format experimental-full \
// RUN:   -mode preprocess-dependency-directives -generate-modules-path-args > %t/output
// RUN: FileCheck %s < %t/output

// CHECK: "-disable-free",

//--- compile-commands.json.in

[{
  "directory": "DIR",
  "command": "clang -c DIR/main.c -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps",
  "file": "DIR/main.c"
}]

//--- module.modulemap

module A {
  header "a.h"
}

//--- a.h

void a(void);

//--- main.c

#include "a.h"
void m() {
  a();
}
