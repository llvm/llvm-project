// RUN: rm -rf %t.idx %t.mcp
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.7 -c %s -o %t.o -index-store-path %t.idx -fmodules -fmodules-cache-path=%t.mcp -Xclang -fdisable-module-hash -I %S/Inputs/module
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s

@import ModDep;
@import ModSystem;

// CHECK: ModDep.pcm
// CHECK: provider: clang-
// CHECK: is-system: 0
// CHECK: is-module: 1
// CHECK: module-name: ModDep
// CHECK: has-main: 0
// CHECK: main-path: {{$}}
// CHECK: out-file: {{.*}}{{/|\\}}ModDep.pcm
// CHECK: DEPEND START
// CHECK: Unit | user | ModTop | {{.*}}{{/|\\}}ModTop.pcm | ModTop.pcm
// CHECK: Record | user | ModDep | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModDep.h | ModDep.h
// CHECK: DEPEND END (2)

// CHECK: ModSystem.pcm
// CHECK: is-system: 1
// CHECK: is-module: 1
// CHECK: module-name: ModSystem
// CHECK: has-main: 0
// CHECK: main-path: {{$}}
// CHECK: out-file: {{.*}}{{/|\\}}ModSystem.pcm
// CHECK: DEPEND START
// CHECK: Record | system | ModSystem | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModSystem.h | ModSystem.h
// CHECK: DEPEND END (1)

// CHECK: ModTop.pcm
// CHECK: is-system: 0
// CHECK: is-module: 1
// CHECK: module-name: ModTop
// CHECK: has-main: 0
// CHECK: main-path: {{$}}
// CHECK: out-file: {{.*}}{{/|\\}}ModTop.pcm
// CHECK: DEPEND START
// CHECK: Record | user | ModTop | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModTop.h | ModTop.h
// CHECK: Record | user | ModTop.Sub1 | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModTopSub1.h | ModTopSub1.h
// CHECK: File | user | ModTop.Sub2 | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}ModTopSub2.h{{$}}
// CHECK: DEPEND END (3)

// CHECK: print-units-with-modules.m.tmp.o
// CHECK: is-system: 0
// CHECK: is-module: 0
// CHECK: module-name: <none>
// CHECK: has-main: 1
// CHECK: main-path: {{.*}}{{/|\\}}print-units-with-modules.m
// CHECK: out-file: {{.*}}{{/|\\}}print-units-with-modules.m.tmp.o
// CHECK: DEPEND START
// CHECK: Unit | user | ModDep | {{.*}}{{/|\\}}ModDep.pcm | ModDep.pcm
// CHECK: Unit | system | ModSystem | {{.*}}{{/|\\}}ModSystem.pcm | ModSystem.pcm
// CHECK: File | user | {{.*}}{{/|\\}}print-units-with-modules.m{{$}}
// CHECK: File | user | {{.*}}{{/|\\}}Inputs{{/|\\}}module{{/|\\}}module.modulemap{{$}}
// CHECK: DEPEND END (4)
