// Check that when scanning framework modules, changing the framework binary
// does not change the cache key.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_no_fw.json

// RUN: echo 'build 1' > %t/Foo.framework/Foo

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_fw1.json

// The existince of the framework is significant, since it affects autolinking.
// RUN: not diff -u %t/deps_fw1.json %t/deps_fw2.json

// RUN: echo 'build 2' > %t/Foo.framework/Foo

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_fw2.json

// But the contents of the binary are not.
// RUN: diff -u %t/deps_fw1.json %t/deps_fw2.json

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang_tool -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache -F DIR",
  "file" : "DIR/tu.c"
}]

//--- Foo.framework/Modules/module.modulemap
framework module Foo {
  umbrella header "Foo.h"
  export *
}

//--- Foo.framework/Headers/Foo.h

//--- tu.c
#include "Foo/Foo.h"
