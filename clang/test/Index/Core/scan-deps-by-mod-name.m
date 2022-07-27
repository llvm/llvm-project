// Use driver arguments.
// RUN: rm -rf %t.mcp
// RUN: echo %S > %t.result
// RUN: echo %S > %t_v2.result
//
// RUN: c-index-test core --scan-deps-by-mod-name -output-dir %t -module-name=ModA %S -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck %s -check-prefixes=CHECK,CHECK_V3 -DOUTPUTS=%/t
//
// RUN: c-index-test core --scan-deps-by-mod-name -scandeps-v2 -module-name=ModA %S -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c >> %t_v2.result
// RUN: cat %t_v2.result | sed 's/\\/\//g' | FileCheck %s -check-prefixes=CHECK,CHECK_V2

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
// CHECK-NEXT:     build-args: {{.*}} -emit-module {{.*}} -fmodule-name=ModA {{.*}} -fno-implicit-modules {{.*}}
// CHECK-NEXT: dependencies:
// CHECK-NEXT:   context-hash:
// CHECK-NEXT:   module-deps:
// CHECK-NEXT:     ModA:[[HASH_MOD_A]]
// CHECK-NEXT:   file-deps:
// CHECK-NEXT:   build-args: {{.*}} -fno-implicit-modules -fno-implicit-module-maps
// CHECK_V3:     -fmodule-file={{.*}}ModA_{{.*}}.pcm
// CHECK_V2-NOT: -fmodule-file=
