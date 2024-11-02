// Check that tracking of VFSs works with PCH.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-pch.json.in > %t/build/compile-commands-pch.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-tu.json.in > %t/build/compile-commands-tu.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-tu-no-vfs.json.in > %t/build/compile-commands-tu-no-vfs.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/pch-overlay.yaml.in > %t/build/pch-overlay.yaml

// RUN: clang-scan-deps -compilation-database %t/build/compile-commands-pch.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search > %t/pch-deps.db
// RUN: %deps-to-rsp %t/pch-deps.db --module-name=A > %t/A.rsp
// RUN: %deps-to-rsp %t/pch-deps.db --module-name=B > %t/B.rsp
// RUN: %deps-to-rsp %t/pch-deps.db --tu-index=0 > %t/pch.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/B.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -compilation-database %t/build/compile-commands-tu.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search > %t/tu-deps.db
// RUN: %deps-to-rsp %t/tu-deps.db --module-name=C > %t/C.rsp
// RUN: %deps-to-rsp %t/tu-deps.db --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/C.rsp
// RUN: %clang @%t/tu.rsp

// RUN: not clang-scan-deps -compilation-database %t/build/compile-commands-tu-no-vfs.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search 2>&1 | FileCheck %s

// CHECK: error: PCH was compiled with different VFS overlay files than are currently in use
// CHECK: note: current translation unit has no VFS overlays

//--- build/compile-commands-pch.json.in

[
{
  "directory": "DIR",
  "command": "clang -x objective-c-header DIR/pch.h -I DIR/modules/A -I DIR/modules/B -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -o DIR/pch.h.pch -ivfsoverlay DIR/build/pch-overlay.yaml",
  "file": "DIR/pch.h"
}
]

//--- build/compile-commands-tu.json.in

[
{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -I DIR/modules/A -I DIR/modules/B -I DIR/modules/C -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -include DIR/pch.h -o DIR/tu.o -ivfsoverlay DIR/build/pch-overlay.yaml",
  "file": "DIR/tu.m"
}
]

//--- build/compile-commands-tu-no-vfs.json.in

[
{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -I DIR/modules/A -I DIR/modules/B -I DIR/modules/C -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -include DIR/pch.h -o DIR/tu.o",
  "file": "DIR/tu.m"
}
]

//--- build/pch-overlay.yaml.in

{
   "version":0,
   "case-sensitive":"false",
   "roots":[
      {
         "contents":[
         {
            "external-contents":"DIR/build/module.modulemap",
            "name":"module.modulemap",
            "type":"file"
         },
         {
            "external-contents":"DIR/build/A.h",
            "name":"A.h",
            "type":"file"
         }
         ],
         "name":"DIR/modules/A",
         "type":"directory"
      }
   ]
}

//--- pch.h
#include <B.h>

//--- build/module.modulemap

module A {
  umbrella header "A.h"
}

//--- build/A.h

typedef int A_t;

//--- modules/B/module.modulemap

module B {
  umbrella header "B.h"
  export *
}

//--- modules/B/B.h
#include <A.h>

typedef int B_t;

//--- modules/C/module.modulemap

module C {
  umbrella header "C.h"
}

//--- modules/C/C.h
#include <B.h>

typedef int C_t;

//--- tu.m

#include <C.h>

A_t a = 0;
B_t b = 0;
C_t c = 0;
