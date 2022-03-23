// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: sed -e "s|DIR|%/t.dir|g" %t.dir/build/compile-commands.json.in > %t.dir/build/compile-commands.json
// RUN: sed -e "s|DIR|%/t.dir|g" %t.dir/build/vfs.yaml.in > %t.dir/build/vfs.yaml
// RUN: clang-scan-deps -compilation-database %t.dir/build/compile-commands.json -j 1 -format experimental-full \
// RUN:   -mode preprocess-minimized-sources -generate-modules-path-args > %t.db
// RUN: %deps-to-rsp %t.db --module-name=A > %t.A.cc1.rsp
// RUN: cat %t.A.cc1.rsp | sed 's:\\\\\?:/:g' | FileCheck %s

// CHECK-NOT: build/module.modulemap
// CHECK: A/module.modulemap

//--- build/compile-commands.json.in

[
{
  "directory": "DIR",
  "command": "clang DIR/main.m -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps -ivfsoverlay build/vfs.yaml",
  "file": "DIR/main.m"
}
]

//--- build/module.modulemap

module A {
  umbrella header "A.h"
}

//--- modules/A/A.h

typedef int A_t;

//--- build/vfs.yaml.in

{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
  {
     "contents": [
     {
        "external-contents": "DIR/build/module.modulemap",
        "name": "module.modulemap",
        "type": "file"
     }],
     "name": "DIR/modules/A",
     "type": "directory"
  }
  ]
}

//--- main.m

@import A;

A_t a = 0;
