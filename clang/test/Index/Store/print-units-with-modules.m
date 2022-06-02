// RUN: rm -rf %t.idx %t.mcp
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t.o -index-store-path %t.idx -fmodules -fmodules-cache-path=%t.mcp -Xclang -fdisable-module-hash -I %S/Inputs/module
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s --check-prefixes=ALL,MODULES

// RUN: rm -rf %t.idx %t.mcp
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t.o -index-store-path %t.idx -index-ignore-pcms -fmodules -fmodules-cache-path=%t.mcp -Xclang -fdisable-module-hash -I %S/Inputs/module
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s --check-prefixes=ALL,IGNORE

@import ModDep;
@import ModSystem;

// IGNORE-NOT: ModDep.pcm
// MODULES: ModDep.pcm
// MODULES: provider: clang-
// MODULES: is-system: 0
// MODULES: is-module: 1
// MODULES: module-name: ModDep
// MODULES: has-main: 0
// MODULES: main-path: {{$}}
// MODULES: out-file: {{.*}}{{/|\\}}ModDep.pcm
// MODULES: DEPEND START
// MODULES: Unit | user | ModTop | {{.*}}{{/|\\}}ModTop.pcm | ModTop.pcm
// MODULES: Record | user | ModDep | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModDep.h | ModDep.h
// MODULES: DEPEND END (2)

// IGNORE-NOT: ModSystem.pcm
// MODULES: ModSystem.pcm
// MODULES: is-system: 1
// MODULES: is-module: 1
// MODULES: module-name: ModSystem
// MODULES: has-main: 0
// MODULES: main-path: {{$}}
// MODULES: out-file: {{.*}}{{/|\\}}ModSystem.pcm
// MODULES: DEPEND START
// MODULES: Record | system | ModSystem | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModSystem.h | ModSystem.h
// MODULES: DEPEND END (1)

// IGNORE-NOT: ModTop.pcm
// MODULES: ModTop.pcm
// MODULES: is-system: 0
// MODULES: is-module: 1
// MODULES: module-name: ModTop
// MODULES: has-main: 0
// MODULES: main-path: {{$}}
// MODULES: out-file: {{.*}}{{/|\\}}ModTop.pcm
// MODULES: DEPEND START
// MODULES: Record | user | ModTop | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModTop.h | ModTop.h
// MODULES: Record | user | ModTop.Sub1 | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModTopSub1.h | ModTopSub1.h
// MODULES: File | user | ModTop.Sub2 | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModTopSub2.h{{$}}
// MODULES: DEPEND END (3)

// ALL: print-units-with-modules.m.tmp.o
// ALL: is-system: 0
// ALL: is-module: 0
// ALL: module-name: <none>
// ALL: has-main: 1
// ALL: main-path: {{.*}}{{/|\\}}print-units-with-modules.m
// ALL: out-file: {{.*}}{{/|\\}}print-units-with-modules.m.tmp.o
// ALL: DEPEND START
// MODULES: Unit | user | ModDep | {{.*}}{{/|\\}}ModDep.pcm | ModDep.pcm
// MODULES: Unit | system | ModSystem | {{.*}}{{/|\\}}ModSystem.pcm | ModSystem.pcm
// IGNORE: Unit | user | ModDep | {{.*}}{{/|\\}}ModDep.pcm
// IGNORE: Unit | system | ModSystem | {{.*}}{{/|\\}}ModSystem.pcm
// ALL: File | user | {{.*}}{{/|\\}}print-units-with-modules.m{{$}}
// ALL: File | user | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}module.modulemap{{$}}
// ALL: DEPEND END (4)
