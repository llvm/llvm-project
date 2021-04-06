// RUN: rm -rf %t.mcp
// RUN: echo %S > %t.result
// RUN: c-index-test core --scan-deps %S -- %clang -cc1 -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp -fimplicit-module-maps \
// RUN:     -o FoE.o -x objective-c %s >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck %s

// Use driver arguments.
// RUN: rm -rf %t.mcp
// RUN: echo %S > %t.result
// RUN: c-index-test core --scan-deps %S -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c %s >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck %s

@import ModA;

// CHECK: [[PREFIX:.*]]
// CHECK-NEXT: modules:
// CHECK-NEXT:   module:
// CHECK-NEXT:     name: ModA
// CHECK-NEXT:     context-hash: [[CONTEXT_HASH:[A-Z0-9]+]]
// CHECK-NEXT:     module-map-path: [[PREFIX]]/Inputs/module/module.modulemap
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/ModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/SubModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/SubSubModA.h
// CHECK-NEXT:       [[PREFIX]]/Inputs/module/module.modulemap
// CHECK-NEXT:     build-args: -remove-preceeding-explicit-module-build-incompatible-options -fno-implicit-modules -emit-module -fmodule-name=ModA
// CHECK-NEXT: dependencies:
// CHECK-NEXT:   context-hash: [[CONTEXT_HASH]]
// CHECK-NEXT:   module-deps:
// CHECK-NEXT:     ModA:[[CONTEXT_HASH]]
// CHECK-NEXT:   file-deps:
// CHECK-NEXT:     [[PREFIX]]/scan-deps.m
// CHECK-NEXT:   additional-build-args: -fno-implicit-modules -fno-implicit-module-maps
