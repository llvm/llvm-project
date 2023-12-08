// RUN: rm -rf %t
// RUN: cp -r %S/Inputs/relative-import-path %t
// RUN: cp %s %t/t.c

// Use FileCheck, which is more flexible.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:     -fdisable-module-hash -fsyntax-only \
// RUN:     -I%S/Inputs/relative-import-path \
// RUN:     -working-directory=%t \
// RUN:     -Rmodule-build -Rmodule-import t.c 2>&1 |\
// RUN: FileCheck %s -implicit-check-not "remark:" -DWORKDIR=%t

#include "A.h" // \
// CHECK: remark: building module 'A'
// CHECK: remark: building module 'B'
// CHECK: remark: building module 'C'
// CHECK: remark: finished building module 'C'
// CHECK: remark: importing module 'C' from '[[WORKDIR]]{{[/\\]cache[/\\]}}C.pcm'
// CHECK: remark: finished building module 'B'
// CHECK: remark: importing module 'B' from '[[WORKDIR]]{{[/\\]cache[/\\]}}B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '[[WORKDIR]]{{[/\\]cache[/\\]}}C.pcm'
// CHECK: remark: finished building module 'A'
// CHECK: remark: importing module 'A' from '[[WORKDIR]]{{[/\\]cache[/\\]}}A.pcm'
// CHECK: remark: importing module 'B' into 'A' from '[[WORKDIR]]{{[/\\]cache[/\\]}}B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '[[WORKDIR]]{{[/\\]cache[/\\]}}C.pcm'
