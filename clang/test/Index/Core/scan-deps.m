// Use driver arguments.
// RUN: rm -rf %t.mcp
// RUN: echo %S > %t.result
// RUN: echo %S > %t_savetemps.result
// RUN: echo %S > %t_v3.result
//
// RUN: c-index-test core --scan-deps -working-dir %S -output-dir=%t -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c %s >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck %s -DOUTPUTS=%/t --check-prefixes=CHECK,CC1

// RUN: c-index-test core --scan-deps -working-dir %S -output-dir=%t -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp -save-temps=obj \
// RUN:     -o FoE.o -x objective-c %s >> %t_savetemps.result
// RUN: cat %t_savetemps.result | sed 's/\\/\//g' | FileCheck %s -DOUTPUTS=%/t --check-prefixes=CHECK,SAVETEMPS

@import ModA;

// CHECK: [[PREFIX:.*]]
// CHECK-NEXT: modules:
// CHECK-NEXT:   module:
// CHECK-NEXT:     name: ModA
// CHECK-NEXT:     context-hash: [[HASH_MOD_A:[A-Z0-9]+]]
// CHECK-NEXT:     cwd-ignored: 0
// CHECK-NEXT:     module-map-path: [[PREFIX]]/Inputs/module/module.modulemap
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/module.modulemap
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/ModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/SubModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/SubSubModA.h
// CHECK-NEXT:     build-args: {{.*}} -emit-module {{.*}} -fmodule-name=ModA {{.*}} -fno-implicit-modules {{.*}}

// CHECK-NEXT: dependencies:
// CHECK-NEXT:   command 0:
// CHECK-NEXT:     context-hash: [[HASH_TU:[A-Z0-9]+]]
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:       ModA:[[HASH_MOD_A]]
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]/scan-deps.m
// CC1-NEXT:       build-args: -cc1 {{.*}} -fmodule-file={{(ModA=)?}}{{.*}}ModA_{{.*}}.pcm
// SAVETEMPS-NEXT: build-args: -cc1 {{.*}} -E {{.*}} -fmodule-file={{(ModA=)?}}{{.*}}ModA_{{.*}}.pcm

// SAVETEMPS-NEXT: command 1:
// SAVETEMPS-NEXT:   context-hash: [[HASH_TU]]
// SAVETEMPS-NEXT:   module-deps:
// SAVETEMPS-NEXT:     ModA:[[HASH_MOD_A]]
// SAVETEMPS-NEXT:   file-deps:
// SAVETEMPS-NEXT:     [[PREFIX]]/scan-deps.m
// SAVETEMPS-NEXT:   build-args: -cc1 {{.*}} -emit-llvm-bc {{.*}} -fmodule-file={{(ModA=)?}}{{.*}}ModA_{{.*}}.pcm

// SAVETEMPS-NEXT: command 2:
// SAVETEMPS-NEXT:   context-hash: [[HASH_TU]]
// SAVETEMPS-NEXT:   module-deps:
// SAVETEMPS-NEXT:     ModA:[[HASH_MOD_A]]
// SAVETEMPS-NEXT:   file-deps:
// SAVETEMPS-NEXT:     [[PREFIX]]/scan-deps.m
// SAVETEMPS-NEXT:   build-args: -cc1 {{.*}} -S

// SAVETEMPS-NEXT: command 3:
// SAVETEMPS-NEXT:   context-hash: [[HASH_TU]]
// SAVETEMPS-NEXT:   module-deps:
// SAVETEMPS-NEXT:     ModA:[[HASH_MOD_A]]
// SAVETEMPS-NEXT:   file-deps:
// SAVETEMPS-NEXT:     [[PREFIX]]/scan-deps.m
// SAVETEMPS-NEXT:   build-args: -cc1as
