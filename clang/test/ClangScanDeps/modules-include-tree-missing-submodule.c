// Ensure we fallback to textual inclusion for headers in incomplete umbrellas.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json

// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

// RUN: %deps-to-rsp %t/deps_pch.json --module-name Foo > %t/Foo.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/Foo.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Extract include-tree casids
// RUN: cat %t/Foo.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Foo.casid
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid

// RUN: echo "MODULE Foo" > %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Foo.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid >> %t/result.txt
// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t

// CHECK-LABEL: MODULE Foo
// CHECK: <module-includes> llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 [[PREFIX]]/Foo.framework/Headers/Foo.h llvmcas://
// CHECK:   Submodule: Foo
// CHECK-NOT: Bar
// CHECK: Module Map:
// CHECK: Foo (framework)
// CHECK-NOT: Bar
// CHECK:   module *
// CHECK-NOT: Bar

// CHECK-LABEL: TRANSLATION UNIT
// CHECK: (PCH) llvmcas://
// CHECK: [[PREFIX]]/tu.c llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 [[PREFIX]]/Foo.framework/Headers/Bar.h llvmcas://

// RUN: %clang @%t/tu.rsp

//--- cdb_pch.json.template
[{
  "file": "DIR/prefix.h",
  "directory": "DIR",
  "command": "clang -x c-header DIR/prefix.h -o DIR/prefix.h.pch -F DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache"
}]

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -include prefix.h -F DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache"
}]

//--- Foo.framework/Modules/module.modulemap
framework module Foo {
  umbrella header "Foo.h"
  module * { export * }
}

//--- Foo.framework/Headers/Foo.h
// Do not import Bar.h
void foo(void);

//--- Foo.framework/Headers/Bar.h
void bar(void);

//--- prefix.h
#include <Foo/Foo.h>

//--- tu.c
#include <Foo/Bar.h>
// FIXME: -Wincomplete-umbrella warning
void tu(void) {
  bar();
}
