// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands.json.in > %t/build/compile-commands.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/vfs.yaml.in > %t/build/vfs.yaml
// RUN: sed -e "s|DIR|%/t|g" %t/build/unused-vfs.yaml.in > %t/build/unused-vfs.yaml
// RUN: sed -e "s|DIR|%/t|g" %t/build/unused-vfs.yaml.in > %t/build/unused2-vfs.yaml
// RUN: clang-scan-deps -compilation-database %t/build/compile-commands.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search > %t/deps.db
// RUN: cat %t/deps.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Check that unused -ivfsoverlay arguments are removed, and that used ones are
// not.

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/A/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "build/unused-vfs.yaml"
// CHECK:              "-ivfsoverlay"
// CHECK-NEXT:         "build/vfs.yaml"
// CHECK-NOT:          "build/unused-vfs.yaml"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/build/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/build/A.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "A"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/B/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-ivfsoverlay"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules/B/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/modules/B/B.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "B"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "B"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/C/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-ivfsoverlay"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules/C/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/modules/C/C.h",
// CHECK-NEXT:         "[[PREFIX]]/modules/B/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "C"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }

//--- build/compile-commands.json.in

[
{
  "directory": "DIR",
  "command": "clang -c DIR/0.m -Imodules/A -Imodules/B -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -ivfsoverlay build/unused-vfs.yaml -ivfsoverlay build/unused2-vfs.yaml -ivfsoverlay build/vfs.yaml",
  "file": "DIR/0.m"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/A.m -Imodules/A -Imodules/B -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -ivfsoverlay build/vfs.yaml -ivfsoverlay build/unused-vfs.yaml",
  "file": "DIR/A.m"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/B.m -Imodules/B -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -ivfsoverlay build/unused-vfs.yaml -ivfsoverlay build/vfs.yaml",
  "file": "DIR/B.m"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/C.m -Imodules/A -Imodules/B -Imodules/C -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -ivfsoverlay build/vfs.yaml -ivfsoverlay build/unused-vfs.yaml",
  "file": "DIR/C.m"
}
]

//--- build/vfs.yaml.in

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

//--- build/unused-vfs.yaml.in

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
            }
         ],
         "name":"DIR/modules/D",
         "type":"directory"
      }
   ]
}

//--- build/module.modulemap

module A {
  umbrella header "A.h"
}

//--- build/A.h

typedef int A_t;

//--- modules/B/module.modulemap

module B {
  umbrella header "B.h"
}

//--- modules/B/B.h

typedef int B_t;

//--- modules/C/module.modulemap

module C {
  umbrella header "C.h"
}

//--- modules/C/C.h

@import B;

typedef B_t C_t;

//--- 0.m

#include <A.h>

A_t a = 0;

//--- A.m

#include <A.h>

A_t a = 0;

//--- B.m

#include <B.h>

B_t b = 0;

//--- C.m

#include <C.h>

C_t b = 0;
