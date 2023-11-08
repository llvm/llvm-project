// Check that the path of an imported modulemap file is not influenced by
// modules outside that module's dependency graph. Specifically, the "Foo"
// module below does not transitively import Mod via a symlink, so it should not
// see the symlinked path.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: ln -s module %t/include/symlink-to-module

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full  -mode=preprocess-dependency-directives \
// RUN:   -optimize-args=all -module-files-dir %t/build > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK: "modules": [
// CHECK:   {
// CHECK:     "command-line": [
// CHECK-NOT: ]
// CHECK:       "-fmodule-map-file=[[PREFIX]]/include/module/module.modulemap"
// CHECK:     ]
// CHECK:     "name": "Foo"
// CHECK:   }

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/test.c -F DIR/Frameworks -I DIR/include -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache",
  "file": "DIR/test.c"
}]

//--- include/module/module.modulemap
module Mod { header "mod.h" export * }

//--- include/module/mod.h

//--- include/module.modulemap
module Other { header "other.h" export * }

//--- include/other.h
#include "symlink-to-module/mod.h"
#include "module/mod.h"

//--- Frameworks/Foo.framework/Modules/module.modulemap
framework module Foo { header "Foo.h" export * }
//--- Frameworks/Foo.framework/Modules/module.private.modulemap
framework module Foo_Private { header "Priv.h" export * }

//--- Frameworks/Foo.framework/Headers/Foo.h
#include "module/mod.h"

//--- Frameworks/Foo.framework/PrivateHeaders/Priv.h
#include <Foo/Foo.h>
#include "other.h"

//--- module.modulemap
module Test { header "test.h" export * }

//--- test.h
#include <Foo/Priv.h>
#include <Foo/Foo.h>

//--- test.c
#include "test.h"
