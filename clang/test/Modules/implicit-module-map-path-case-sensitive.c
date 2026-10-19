// Ensure that module variant hashes in the PCM paths differ when module map paths 
// differ by case. This relies on the canonical path of each
// module map resolving to the path specified by the case-sensitive VFS
// overlay, which is not dependent on underlying file system's resolution.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json

// RUN: %clang_cc1 -fsyntax-only %t/tu0.c -ivfsoverlay %t/overlay.json \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -Rmodule-build 2> %t/remarks_tu0.txt
// RUN: find %t/cache -name "Mod-*.pcm" | count 1

// RUN: %clang_cc1 -fsyntax-only %t/tu1.c -ivfsoverlay %t/overlay.json \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -Rmodule-build 2> %t/remarks_tu1.txt
// RUN: cat %t/remarks_tu0.txt %t/remarks_tu1.txt | FileCheck %s
// RUN: find %t/cache -name "Mod-*.pcm" | count 2

// CHECK:     tu0.c:{{.*}} remark: building module 'Mod' as '{{.*}}Mod-[[HASH:[A-Z0-9]+]].pcm'
// CHECK-NOT: Mod-[[HASH]].pcm

//--- overlay.json.template
{
  "version": 0,
  "case-sensitive": true,
  "roots": [
  {
     "contents": [
     {
        "external-contents": "DIR/real/module.modulemap",
        "name": "module.modulemap",
        "type": "file"
     },
     {
        "external-contents": "DIR/real/Mod.h",
        "name": "Mod.h",
        "type": "file"
     }],
     "name": "DIR/Dir",
     "type": "directory"
  },
  {
     "contents": [
     {
        "external-contents": "DIR/real/module.modulemap",
        "name": "module.modulemap",
        "type": "file"
     },
     {
        "external-contents": "DIR/real/Mod.h",
        "name": "Mod.h",
        "type": "file"
     }],
     "name": "DIR/dir",
     "type": "directory"
  }
  ]
}

//--- real/module.modulemap
module Mod { header "Mod.h" }
//--- real/Mod.h

//--- tu0.c
#include "Dir/Mod.h"

//--- tu1.c
#include "dir/Mod.h"
