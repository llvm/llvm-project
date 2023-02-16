// RUN: rm -rf %t.mcp %t
// RUN: echo %S > %t.result
//
// RUN: c-index-test core --scan-deps %S -output-dir=%t -cas-path %t/cas \
// RUN:  -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c %s >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck %s -DOUTPUTS=%/t

// RUN: c-index-test core --scan-deps %S -output-dir=%t \
// RUN:  -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c %s | FileCheck %s -check-prefix=NO_CAS
// NO_CAS-NOT: fcas
// NO_CAS-NOT: faction-cache
// NO_CAS-NOT: fcache-compile-job

@import ModA;

// CHECK: [[PREFIX:.*]]
// CHECK-NEXT: modules:
// CHECK-NEXT:   module:
// CHECK-NEXT:     name: ModA
// CHECK-NEXT:     context-hash: [[HASH_MOD_A:[A-Z0-9]+]]
// CHECK-NEXT:     module-map-path: [[PREFIX]]/Inputs/module/module.modulemap
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/ModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/SubModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/SubSubModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/module.modulemap
// CHECK-NEXT:     build-args:
// CHECK-SAME:       -cc1
// CHECK-SAME:       -fcas-path
// CHECK-SAME:       -fcas-fs llvmcas://{{[[:xdigit:]]+}}
// CHECK-SAME:       -fcache-compile-job
// CHECK-SAME:       -emit-module
// CHECK-SAME:       -fmodule-name=ModA
// CHECK-SAME:       -fno-implicit-modules

// CHECK-NEXT: dependencies:
// CHECK-NEXT:   command 0:
// CHECK-NEXT:     context-hash: [[HASH_TU:[A-Z0-9]+]]
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:       ModA:[[HASH_MOD_A]]
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]/scan-deps-cas.m
// CHECK-NEXT:     build-args:
// CHECK-SAME:       -cc1
// CHECK-SAME:       -fcas-path
// CHECK-SAME:       -fcas-fs llvmcas://{{[[:xdigit:]]+}}
// CHECK-SAME:       -fcache-compile-job
// CHECK-SAME:       -fmodule-file-cache-key=[[PCM:.*ModA_.*pcm]]=llvmcas://{{[[:xdigit:]]+}}
// CHECK-SAME:       -fmodule-file={{(ModA=)?}}[[PCM]]
