// Check that symlink aliases to a module map directory produce the same PCM
// across incremental scans. This test was adapted from 
// modules-symlink-dir-from-module.c, where a transitive module dependency (Mod)
// is only discoverable through an included header and not from by-name lookup.

// The first scan builds Mod's PCM via the canonical path. 
// The second scan encounters Mod only through a symlinked directory, but
// should reuse the existing PCM rather than rebuilding it with the symlink path
// as the module map file. This is because the PCM filename hash is canonicalized, 
// and the module map is resolved through header search so Mod's module map is 
// always discoverable regardless of whether it was imported directly or transitively.

// REQUIRES: symlinks

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb_alternate_path.json.in > %t/cdb_alternate_path.json
// RUN: ln -s module %t/include/symlink-to-module

// RUN: touch %t/session.timestamp

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full  -mode=preprocess-dependency-directives \
// RUN:   -optimize-args=all -module-files-dir %t/build > %t/deps.json

// RUN: ls %t/module-cache/*/Mod-*.pcm | count 1
// RUN: %clang -cc1 -module-file-info %t/module-cache/*/Mod-*.pcm 2>&1 | grep "Module map file" | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// RUN: clang-scan-deps -compilation-database %t/cdb_alternate_path.json -j 1 \
// RUN:   -format experimental-full  -mode=preprocess-dependency-directives \
// RUN:   -optimize-args=all -module-files-dir %t/build > %t/deps_incremental.json

// RUN: ls %t/module-cache/*/Mod-*.pcm | count 1
// RUN: %clang -cc1 -module-file-info %t/module-cache/*/Mod-*.pcm 2>&1 | grep "Module map file" | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

/// Verify resulting explicit module remain the same.
// RUN: %deps-to-rsp %t/deps.json --module-name=Mod > %t/Mod.cc1.rsp
// RUN: %deps-to-rsp %t/deps_incremental.json --module-name=Mod > %t/Mod_incremental.cc1.rsp
// RUN: diff %t/Mod.cc1.rsp %t/Mod_incremental.cc1.rsp

// CHECK: Module map file: [[PREFIX]]/include/module/module.modulemap

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/test.c -F DIR/Frameworks -I DIR/include -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -fbuild-session-file=DIR/session.timestamp -fmodules-validate-once-per-build-session",
  "file": "DIR/test.c"
}]

//--- cdb_alternate_path.json.in
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/test_incremental.c -F DIR/Frameworks -I DIR/include -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -fbuild-session-file=DIR/session.timestamp -fmodules-validate-once-per-build-session",
  "file": "DIR/test_incremental.c"
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

//--- test_incremental.c
#include "symlink-to-module/mod.h"
