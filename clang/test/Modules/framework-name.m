// REQUIRES: shell
// RUN: rm -rf %t.mcp %t
// RUN: mkdir -p %t
// RUN: ln -s %S/Inputs/NameInDir2.framework %t/NameInImport.framework
// RUN: ln -s %S/Inputs/NameInDirInferred.framework %t/NameInImportInferred.framework
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t.mcp -fimplicit-module-maps -I %S/Inputs -F %S/Inputs -F %t -Wauto-import -verify %s

// Verify that we won't somehow find non-canonical module names or modules where
// we shouldn't search the framework.
// RUN: echo '@import NameInModMap;' | not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -F %S/Inputs -F %t -Wauto-import -x objective-c - 2>&1 | FileCheck %s
// RUN: echo '@import NameInDir;' | not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -F %S/Inputs -F %t -Wauto-import -x objective-c - 2>&1 | FileCheck %s
// RUN: echo '@import NameInImport;' | not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -F %S/Inputs -F %t -Wauto-import -x objective-c - 2>&1 | FileCheck %s
// RUN: echo '@import NameInImportInferred;' | not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -F %S/Inputs -F %t -Wauto-import -x objective-c - 2>&1 | FileCheck %s
// CHECK: module '{{.*}}' not found

// FIXME: We might want to someday lock down framework modules so that these
// name mismatches are disallowed. However, as long as we *don't* prevent them
// it's important that they map correctly to module imports.

// The module map name doesn't match the directory name.
#import <NameInDir/NameInDir.h> // expected-warning {{import of module 'NameInModMap'}}

// The name in the import doesn't match the module name.
#import <NameInImport/NameInDir2.h> // expected-warning {{import of module 'NameInDir2'}}
@import NameInDir2;                 // OK

// The name in the import doesn't match the module name (inferred framework module).
#import <NameInImportInferred/NameInDirInferred.h> // expected-warning {{import of module 'NameInDirInferred'}}

@import ImportNameInDir;
#ifdef NAME_IN_DIR
#error NAME_IN_DIR should be undef'd
#endif
