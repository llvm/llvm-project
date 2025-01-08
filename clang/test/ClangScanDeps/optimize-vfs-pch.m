// Check that tracking of VFSs works with PCH.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-pch.json.in > %t/build/compile-commands-pch.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-tu.json.in > %t/build/compile-commands-tu.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-tu-no-vfs-error.json.in > %t/build/compile-commands-tu-no-vfs-error.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands-tu1.json.in > %t/build/compile-commands-tu1.json
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

// RUN: not clang-scan-deps -compilation-database %t/build/compile-commands-tu-no-vfs-error.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

// CHECK-ERROR: error: PCH was compiled with different VFS overlay files than are currently in use
// CHECK-ERROR: note: current translation unit has no VFS overlays

// Next test is to verify that a module that doesn't use the VFS, that depends
// on the PCH's A, which does use the VFS, still records that it needs the VFS.
// This avoids a fatal error when emitting diagnostics.

// RUN: clang-scan-deps -compilation-database %t/build/compile-commands-tu1.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search > %t/tu1-deps.db
// RUN: %deps-to-rsp %t/tu1-deps.db --tu-index=0 > %t/tu1.rsp
// Reuse existing B
// RUN: %deps-to-rsp %t/tu1-deps.db --module-name=E > %t/E.rsp
// RUN: %deps-to-rsp %t/tu1-deps.db --module-name=D > %t/D.rsp
// The build of D depends on B which depend on the prebuilt A. D will only build
// if it has A's VFS, as it needs to emit a diagnostic showing the content of A.
// RUN: %clang @%t/E.rsp
// RUN: %clang @%t/D.rsp -verify
// RUN: %clang @%t/tu1.rsp
// RUN: cat %t/tu1-deps.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Check that D has the overlay, but E doesn't.
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "E"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/D/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-ivfsoverlay"
// CHECK-NEXT:         "[[PREFIX]]/build/pch-overlay.yaml"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "{{.*}}"
// CHECK-NEXT:         "{{.*}}"
// CHECK-NEXT:         "{{.*}}"
// CHECK-NEXT:         "{{.*}}"
// CHECK-NEXT:       ],
// CHECK:            "name": "D"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/E/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-ivfsoverlay"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "{{.*}}"
// CHECK-NEXT:         "{{.*}}"
// CHECK-NEXT:       ],
// CHECK:            "name": "E"
// CHECK-NEXT:     }

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

//--- build/compile-commands-tu-no-vfs-error.json.in

[
{
  "directory": "DIR",
  "command": "clang -Wpch-vfs-diff -Werror=pch-vfs-diff -fsyntax-only DIR/tu.m -I DIR/modules/A -I DIR/modules/B -I DIR/modules/C -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -include DIR/pch.h -o DIR/tu.o",
  "file": "DIR/tu.m"
}
]

//--- build/compile-commands-tu1.json.in

[
{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu1.m -I DIR/modules/B -I DIR/modules/D -I DIR/modules/E -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -include DIR/pch.h -o DIR/tu1.o -ivfsoverlay DIR/build/pch-overlay.yaml",
  "file": "DIR/tu1.m"
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

typedef int A_t __attribute__((deprecated("yep, it's depr")));

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

//--- modules/D/module.modulemap

module D {
  umbrella header "D.h"
  export *
}

//--- modules/D/D.h
#include <B.h>
#include <E.h>

typedef A_t D_t; // expected-warning{{'A_t' is deprecated}}
// expected-note@*:* {{marked deprecated here}}

//--- modules/E/module.modulemap

module E {
  umbrella header "E.h"
}

//--- modules/E/E.h
typedef int E_t;

//--- tu1.m

#include <D.h>

D_t d = 0;
E_t e = 0;
