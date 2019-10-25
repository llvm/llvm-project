// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t %s -Rmodule-build 2> %t1
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -internal-isystem %S -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t %s -Rmodule-build 2> \
// RUN:   %t2
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -internal-isystem %S -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t %s \
// RUN:   -fmodules-strict-context-hash -Rmodule-build 2> %t3
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -Weverything -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fmodules-strict-context-hash \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t %s -Rmodule-build 2> \
// RUN:   %t4
// RUN: echo %t > %t.path
// RUN: cat %t.path %t1 %t2 %t3 %t4 | FileCheck %s

// This test verifies that only strict hashing includes search paths and
// diagnostics in the module context hash.

#include <stdio.h>

// CHECK: [[PREFIX:(.*[/\\])+[a-zA-Z0-9.-]+]]
// CHECK: building module 'cstd' as '[[PREFIX]]{{[/\\]}}[[CONTEXT_HASH:[A-Z0-9]+]]{{[/\\]}}cstd-[[AST_HASH:[A-Z0-9]+]].pcm'
// CHECK: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}cstd-[[AST_HASH]].pcm'
// CHECK-NOT: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}
// CHECK: cstd-[[AST_HASH]].pcm'
// CHECK-NOT: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}
// CHECK: cstd-[[AST_HASH]].pcm'
