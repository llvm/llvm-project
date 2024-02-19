// This test verifies that only strict hashing includes search paths and
// diagnostics in the module context hash.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fimplicit-module-maps \
// RUN:   -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t %s \
// RUN:   -Rmodule-build 2> %t1
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -internal-isystem %S -fmodules \
// RUN:   -fbuiltin-headers-in-system-modules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t %s -Rmodule-build 2> %t2
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -internal-isystem %S -fmodules \
// RUN:   -fbuiltin-headers-in-system-modules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t %s -fmodules-strict-context-hash \
// RUN:   -Rmodule-build 2> %t3
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -Weverything -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fmodules-strict-context-hash \
// RUN:   -fbuiltin-headers-in-system-modules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t %s -Rmodule-build 2> %t4
// RUN: echo %t > %t.path
// RUN: cat %t.path %t1 %t2 %t3 %t4 | FileCheck %s

// This tests things verified by ASTReader::checkLanguageOptions that are not
// part of LangOpts.def.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fimplicit-module-maps \
// RUN:   -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t \
// RUN:   -x objective-c %s -Rmodule-build 2> %t1
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fimplicit-module-maps \
// RUN:   -fbuiltin-headers-in-system-modules -fobjc-runtime=macosx-1.0.0.0 \
// RUN:   -fmodules-cache-path=%t -x objective-c %s -Rmodule-build 2> %t2
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem \
// RUN:   %S/Inputs/System/usr/include -fmodules -fimplicit-module-maps \
// RUN:   -fbuiltin-headers-in-system-modules -fcomment-block-commands=lp,bj \
// RUN:   -fmodules-cache-path=%t -x objective-c %s -Rmodule-build 2> %t3
// RUN: echo %t > %t.path
// RUN: cat %t.path %t1 %t2 %t3 | FileCheck --check-prefix=LANGOPTS %s

#include <stdio.h>

// CHECK: [[PREFIX:(.*[/\\])+[a-zA-Z0-9.-]+]]
// CHECK: building module 'cstd' as '[[PREFIX]]{{[/\\]}}[[CONTEXT_HASH:[A-Z0-9]+]]{{[/\\]}}cstd-[[AST_HASH:[A-Z0-9]+]].pcm'
// CHECK: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}cstd-[[AST_HASH]].pcm'
// CHECK-NOT: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}
// CHECK: cstd-[[AST_HASH]].pcm'
// CHECK-NOT: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}
// CHECK: cstd-[[AST_HASH]].pcm'

// LANGOPTS: [[PREFIX:(.*[/\\])+[a-zA-Z0-9.-]+]]
// LANGOPTS: building module 'cstd' as '[[PREFIX]]{{[/\\]}}[[CONTEXT_HASH:[A-Z0-9]+]]{{[/\\]}}cstd-[[AST_HASH:[A-Z0-9]+]].pcm'
// LANGOPTS-NOT: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}
// LANGOPTS: cstd-[[AST_HASH]].pcm'
// LANGOPTS-NOT: building module 'cstd' as '{{.*[/\\]}}[[CONTEXT_HASH]]{{[/\\]}}
// LANGOPTS: cstd-[[AST_HASH]].pcm'
