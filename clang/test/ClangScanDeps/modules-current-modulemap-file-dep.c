// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: sed -e "s|DIR|%/t|g" %t/vfs.yaml.in > %t/vfs.yaml

// RUN: clang-scan-deps -format experimental-full -j 1 -- \
// RUN:   %clang -ivfsoverlay %t/vfs.yaml -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t/cache -fmodule-name=ModuleName \
// RUN:     -I %/t/remapped -c %t/header-impl.c -o %t/header-impl.o \
// RUN:     | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:            "command-line": [
// CHECK:              "-fmodule-map-file=[[PREFIX]]/remapped/module.modulemap"
// CHECK:            "file-deps": [
// CHECK:              "[[PREFIX]]/original/module.modulemap"

// Verify that "file-deps" references actual on-disk module map and not using the virtual path.

//--- vfs.yaml.in
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "name": "DIR/remapped",
      "type": "directory",
      "contents": [
        {
          "name": "module.modulemap",
          "type": "file",
          "external-contents": "DIR/original/module.modulemap"
        },
        {
          "name": "header.h",
          "type": "file", 
          "external-contents": "DIR/original/header.h"
        }
      ]
    }
  ]
}

//--- original/module.modulemap
module ModuleName {
  header "header.h"
  export *
}

//--- original/header.h
int foo_function(void);

//--- header-impl.c
#include <header.h>

int foo_function(void) {
  return 0;
}
