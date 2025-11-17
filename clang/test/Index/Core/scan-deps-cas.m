// REQUIRES: ondisk_cas

// RUN: rm -rf %t

// RUN: c-index-test core --scan-deps -working-dir %S -output-dir=%t -cas-path %t/cas \
// RUN:  -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t/mcpit \
// RUN:     -o FoE.o -x objective-c %s > %t.result
// RUN: cat %t.result | FileCheck %s -DPREFIX=%S -DOUTPUTS=%/t -check-prefix=INCLUDE_TREE

// RUN: env CLANG_CACHE_USE_INCLUDE_TREE=1 c-index-test core --scan-deps -working-dir %S -output-dir=%t -cas-path %t/cas \
// RUN:  -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t/mcpit \
// RUN:     -o FoE.o -x objective-c %s > %t.includetree.result
// RUN: cat %t.includetree.result | FileCheck %s -DPREFIX=%S -DOUTPUTS=%/t -check-prefix=INCLUDE_TREE

// RUN: c-index-test core --scan-deps -working-dir %S -output-dir=%t \
// RUN:  -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t/mcp \
// RUN:     -o FoE.o -x objective-c %s | FileCheck %s -check-prefix=NO_CAS
// NO_CAS-NOT: fcas
// NO_CAS-NOT: faction-cache
// NO_CAS-NOT: fcache-compile-job

#include "ModA.h"

// CHECK:       modules:
// CHECK-NEXT:   module:
// CHECK-NEXT:     name: ModA
// CHECK-NEXT:     context-hash: [[HASH_MOD_A:[A-Z0-9]+]]
// CHECK-NEXT:     cwd-ignored: 0
// CHECK-NEXT:     module-map-path: [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}module.modulemap
// CHECK-NEXT:     casfs-root-id: [[CASFS_MODA_ROOT_ID:llvmcas://[[:xdigit:]]+]]
// CHECK-NEXT:     cache-key: [[CASFS_MODA_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}module.modulemap
// CHECK-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}ModA.h
// CHECK-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}SubModA.h
// CHECK-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}SubSubModA.h
// CHECK-NEXT:     link libraries:
// CHECK-NEXT:         libModA(framework)
// CHECK-NEXT:         libModB
// CHECK-NEXT:         /absolute/path/to/a/lib/file
// CHECK-NEXT:     build-args:
// CHECK-SAME:       -cc1
// CHECK-SAME:       -fcas-path
// CHECK-SAME:       -fcas-fs [[CASFS_MODA_ROOT_ID]]
// CHECK-SAME:       -fcache-compile-job
// CHECK-SAME:       -emit-module
// CHECK-SAME:       -fmodule-name=ModA
// CHECK-SAME:       -fno-implicit-modules

// CHECK-NEXT: dependencies:
// CHECK-NEXT:   command 0:
// CHECK-NEXT:     context-hash: [[HASH_TU:[A-Z0-9]+]]
// CHECK-NEXT:     casfs-root-id: [[CASFS_TU_ROOT_ID:llvmcas://[[:xdigit:]]+]]
// CHECK-NEXT:     cache-key: [[CASFS_TU_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:       ModA:[[HASH_MOD_A]]
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]{{[/\\]}}scan-deps-cas.m
// CHECK-NEXT:     build-args:
// CHECK-SAME:       -cc1
// CHECK-SAME:       -fcas-path
// CHECK-SAME:       -fcas-fs [[CASFS_TU_ROOT_ID]]
// CHECK-SAME:       -fcache-compile-job
// CHECK-SAME:       -fmodule-file-cache-key [[PCM:.*ModA_.*pcm]] llvmcas://{{[[:xdigit:]]+}}
// CHECK-SAME:       -fmodule-file={{(ModA=)?}}[[PCM]]


// INCLUDE_TREE:      modules:
// INCLUDE_TREE-NEXT:   module:
// INCLUDE_TREE-NEXT:     name: ModA
// INCLUDE_TREE-NEXT:     context-hash: [[HASH_MOD_A:[A-Z0-9]+]]
// INCLUDE_TREE-NEXT:     cwd-ignored: 0
// INCLUDE_TREE-NEXT:     module-map-path: [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}module.modulemap
// INCLUDE_TREE-NEXT:     include-tree-id: [[ModA_INCLUDE_TREE_ID:llvmcas://[[:xdigit:]]+]]
// INCLUDE_TREE-NEXT:     cache-key: [[ModA_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// INCLUDE_TREE-NEXT:     module-deps:
// INCLUDE_TREE-NEXT:     file-deps:
// INCLUDE_TREE-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}module.modulemap
// INCLUDE_TREE-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}ModA.h
// INCLUDE_TREE-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}SubModA.h
// INCLUDE_TREE-NEXT:       [[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module{{[/\\]}}SubSubModA.h
// INCLUDE_TREE-NEXT:     link libraries:
// INCLUDE_TREE-NEXT:         libModA(framework)
// INCLUDE_TREE-NEXT:         libModB
// INCLUDE_TREE-NEXT:         /absolute/path/to/a/lib/file
// INCLUDE_TREE-NEXT:     build-args:
// INCLUDE_TREE-SAME:       -cc1
// INCLUDE_TREE-SAME:       -fcas-path
// INCLUDE_TREE-SAME:       -fcas-include-tree [[ModA_INCLUDE_TREE_ID]]
// INCLUDE_TREE-SAME:       -fcache-compile-job

// INCLUDE_TREE:      dependencies:
// INCLUDE_TREE-NEXT:   command 0:
// INCLUDE_TREE-NEXT:     context-hash: [[HASH_TU:[A-Z0-9]+]]
// INCLUDE_TREE-NEXT:     include-tree-id: [[INC_TU_INCLUDE_TREE_ID:llvmcas://[[:xdigit:]]+]]
// INCLUDE_TREE-NEXT:     cache-key: [[INC_TU_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// INCLUDE_TREE-NEXT:     module-deps:
// INCLUDE_TREE-NEXT:       ModA:[[HASH_MOD_A]]
// INCLUDE_TREE-NEXT:     file-deps:
// INCLUDE_TREE-NEXT:       [[PREFIX]]{{[/\\]}}scan-deps-cas.m
// INCLUDE_TREE-NEXT:     build-args:
// INCLUDE_TREE-SAME:       -cc1
// INCLUDE_TREE-SAME:       -fcas-path
// INCLUDE_TREE-SAME:       -fcas-include-tree [[INC_TU_INCLUDE_TREE_ID]]
// INCLUDE_TREE-SAME:       -fcache-compile-job
// INCLUDE_TREE-SAME:       -fmodule-file-cache-key [[PCM:.*ModA_.*pcm]] [[ModA_CACHE_KEY]]
// INCLUDE_TREE-SAME:       -fmodule-file={{(ModA=)?}}[[PCM]]
