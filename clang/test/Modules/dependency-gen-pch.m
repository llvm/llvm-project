// RUN: rm -rf %t-mcp
// RUN: mkdir -p %t-mcp
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 -module-file-deps -dependency-file %t.d -MT %s.o -I %S/Inputs -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t-mcp -emit-pch -o %t.pch %s
// RUN: FileCheck %s < %t.d
// CHECK: dependency-gen-pch.m.o
// CHECK-NEXT: dependency-gen-pch.m
// CHECK-NEXT: Inputs{{.}}module.modulemap
// CHECK-NEXT: diamond_top.pcm
// CHECK-NEXT: Inputs{{.}}diamond_top.h

#import "diamond_top.h"
