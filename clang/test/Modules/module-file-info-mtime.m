// Check that mtime from a input file of a pcm is emitted, when it was built from an implicit module invocation.

// RUN: rm -fr %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fdisable-module-hash -fimplicit-module-maps %t/client.m -fsyntax-only -I%t/BuildDir
// RUN: %clang_cc1 -module-file-info %t/cache/A.pcm | FileCheck %s

// CHECK: Module name: A
// CHECK: Module map file: {{.*}}module.modulemap
// CHECK: Input file: {{.*}}A.h
// CHECK-NEXT: MTime: {{[0-9]+}} 


//--- BuildDir/A/module.modulemap
module A [system] {
  umbrella "."
}

//--- BuildDir/A/A.h
typedef int local_t;

//--- client.m
#import <A/A.h>
