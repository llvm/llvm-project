// Test that the virtual file "__inferred_module.map" doesn't show up as dependency.

// REQUIRES: x86-registered-target
// RUN: rm -rf %t-mcp
// RUN: %clang_cc1 -isysroot %S/Inputs/System -triple x86_64-apple-darwin10 -dependency-file %t.d -MT %s.o -F %S/Inputs -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-mcp %s
// RUN: FileCheck %s < %t.d
// CHECK-NOT: __inferred_module

@import Module;
