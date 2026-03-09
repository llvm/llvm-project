// Use driver arguments.
// RUN: rm -rf %t.mcp
// RUN: echo %S > %t.result
// RUN: echo %S > %t_v2.result
//
// RUN: c-index-test core --scan-deps-by-mod-name -output-dir %t -module-name=ModA -working-dir %S -- %clang -c -I %S/Inputs/module \
// RUN:     -fmodules -fmodules-cache-path=%t.mcp \
// RUN:     -o FoE.o -x objective-c >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck %s -DOUTPUTS=%/t

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
// CHECK-NEXT:     link libraries:
// CHECK-NEXT:         libModA(framework)
// CHECK-NEXT:         libModB
// CHECK-NEXT:         /absolute/path/to/a/lib/file
// CHECK-NEXT:     build-args: {{.*}} -emit-module {{.*}} -fmodule-name=ModA {{.*}} -fno-implicit-modules {{.*}}
// CHECK-NEXT: dependencies:
// CHECK-NEXT:   command 0:
// CHECK-NEXT:     context-hash:
// CHECK-NEXT:     module-deps:
// CHECK-NEXT:       ModA:[[HASH_MOD_A]]
// CHECK-NEXT:     file-deps:
// CHECK-NEXT:       {{.*}}ScanningByName-{{.*}}.input
// CHECK-NEXT:     build-args: -cc1 {{.*}} -fmodule-file={{(ModA=)?}}{{.*}}ModA_{{.*}}.pcm
