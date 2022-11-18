// Check that the path of an "affecting" modulemap file matches what is captured
// in the cas fs.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: ln -s module %t/include/symlink-to-module

// RUN: clang-scan-deps -cas-path %t/cas -action-cache-path %t/cache  -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full  -mode=preprocess-dependency-directives \
// RUN:   -optimize-args -module-files-dir %t/build > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name=Mod > %t/mod.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=Other > %t/other.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=Foo > %t/foo.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=Foo_Private > %t/foo_private.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=Test > %t/test.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.cc1.rsp

// RUN: %clang @%t/mod.cc1.rsp
// RUN: %clang @%t/other.cc1.rsp
// RUN: %clang @%t/foo.cc1.rsp
// RUN: %clang @%t/foo_private.cc1.rsp
// RUN: %clang @%t/test.cc1.rsp
// RUN: %clang @%t/tu.cc1.rsp

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
