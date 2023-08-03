// RUN: rm -rf %t %t.idxdep %t.idxignore %t.idx %t.mcp %t.o

// ------------------ Build explicit PCM DependencyA, indexing PCM flag does not disable indexing

// RUN: %clang_cc1 -x objective-c -std=gnu11 -triple x86_64-apple-macosx10.8 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash \
// RUN:            -fmodule-name=DependencyA -emit-module %S/Inputs/explicit-modules/module.modulemap -o %t/DependencyA.pcm \
// RUN:            -index-store-path %t.idxdep -index-ignore-pcms
// RUN: c-index-test core -print-unit %t.idxdep | FileCheck %s --check-prefixes=DEPA

// ------------------ Build without indexing imported PCMs

// RUN: %clang_cc1 -std=gnu11 -triple x86_64-apple-macosx10.8 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash \
// RUN:            -fmodule-file=%t/DependencyA.pcm %s -o %t.o \
// RUN:            -index-store-path %t.idxignore -index-ignore-pcms
// RUN: c-index-test core -print-unit %t.idxignore | FileCheck %s --check-prefixes=MAIN,IGNORE

// ------------------ Build with indexing of imported PCMs

// RUN: %clang_cc1 -std=gnu11 -triple x86_64-apple-macosx10.8 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash \
// RUN:            -fmodule-file=%t/DependencyA.pcm %s -o %t.o \
// RUN:            -index-store-path %t.idx
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s --check-prefixes=DEPA,MAIN,INDEXMAIN

@import DependencyA;

int fetchDependencyAVersion() {
  return dependencyAVersion();
}

// IGNORE-NOT: DependencyA.pcm
// DEPA: DependencyA.pcm
// DEPA: module-name: DependencyA

// MAIN: print-units-explicit-modules-ignore-pcms.m.tmp.o
// MAIN: DEPEND START
// INDEXMAIN: Unit | user | DependencyA | {{.*}}{{/|\\}}DependencyA.pcm | DependencyA.pcm
// IGNORE-NOT: Unit | user | DependencyA | {{.*}}{{/|\\}}DependencyA.pcm | DependencyA.pcm
// IGNORE: Unit | user | DependencyA | {{.*}}{{/|\\}}DependencyA.pcm
