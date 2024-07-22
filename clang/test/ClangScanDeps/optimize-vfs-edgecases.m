// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands.json.in > %t/build/compile-commands.json
// RUN: sed -e "s|DIR|%/t|g" %t/build/vfsoverlay.yaml.in > %t/build/vfsoverlay.yaml
// RUN: sed -e "s|DIR|%/t|g" %t/build/vfs.notyaml.in > %t/build/vfs.notyaml
// RUN: clang-scan-deps -compilation-database %t/build/compile-commands.json \
// RUN:   -j 1 -format experimental-full --optimize-args=vfs,header-search > %t/deps.db

// RUN: %deps-to-rsp %t/deps.db --module-name=A > %t/A.rsp
// RUN: cd %t && %clang @%t/A.rsp

// Check that the following edge cases are handled by ivfsoverlay tracking
// * `-ivfsoverlay` args that depend on earlier `-ivfsoverlay` args.

//--- build/compile-commands.json.in

[
{
  "directory": "DIR",
  "command": "clang -c DIR/0.m -Imodules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -ivfsoverlay build/vfsoverlay.yaml -ivfsoverlay build/vfs.yaml",
  "file": "DIR/0.m"
}
]

//--- build/vfsoverlay.yaml.in

{
   "version":0,
   "case-sensitive":"false",
   "roots":[
      {
         "contents":[
            {
               "external-contents":"DIR/build/vfs.notyaml",
               "name":"vfs.yaml",
               "type":"file"
            }
         ],
         "name":"DIR/build",
         "type":"directory"
      }
   ]
}

//--- build/vfs.notyaml.in

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

//--- build/module.modulemap

module A {
  umbrella header "A.h"
}

//--- build/A.h

typedef int A_t;

//--- 0.m

#include <A.h>

A_t a = 0;
