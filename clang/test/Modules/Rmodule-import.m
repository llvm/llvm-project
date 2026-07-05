// RUN: rm -rf %t1 %t2

// Run with -verify, which onliy gets remarks from the main TU.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t1 \
// RUN:     -fdisable-module-hash -fsyntax-only -I%S/Inputs/Rmodule-import \
// RUN:     -Rmodule-build -Rmodule-import -verify %s

// Run again, using FileCheck to check remarks from the module builds.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t2 \
// RUN:     -fdisable-module-hash -fsyntax-only -I%S/Inputs/Rmodule-import \
// RUN:     -Rmodule-build -Rmodule-import %s 2>&1 |\
// RUN: FileCheck %s -implicit-check-not "remark:"

#include "A.h" // \
   expected-remark-re{{building module 'A' as '{{.*[/\\]}}A.pcm'}} \
   expected-remark{{finished building module 'A'}} \
   expected-remark-re{{importing module 'A' from '{{.*[/\\]}}A.pcm'}} \
   expected-remark-re{{importing module 'B' into 'A' from '{{.*[/\\]}}B.pcm'}} \
   expected-remark-re{{importing module 'C' into 'B' from '{{.*[/\\]}}C.pcm'}}
// CHECK: remark: building module 'A'
// CHECK: remark: building module 'B'
// CHECK: remark: building module 'C'
// CHECK: remark: finished building module 'C'
// CHECK: remark: importing module 'C' from '{{.*[/\\]}}C.pcm'
// CHECK: remark: finished building module 'B'
// CHECK: remark: importing module 'B' from '{{.*[/\\]}}B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '{{.*[/\\]}}C.pcm'
// CHECK: remark: finished building module 'A'
// CHECK: remark: importing module 'A' from '{{.*[/\\]}}A.pcm'
// CHECK: remark: importing module 'B' into 'A' from '{{.*[/\\]}}B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '{{.*[/\\]}}C.pcm'
#include "B.h" // \
   expected-remark-re{{importing module 'B' from '{{.*[/\\]}}B.pcm'}}
// CHECK: remark: importing module 'B' from '{{.*[/\\]}}B.pcm'
#include "C.h" // \
   expected-remark-re{{importing module 'C' from '{{.*[/\\]}}C.pcm'}}
// CHECK: remark: importing module 'C' from '{{.*[/\\]}}C.pcm'
@import D; // \
   expected-remark-re{{building module 'D' as '{{.*[/\\]}}D.pcm'}} \
   expected-remark{{finished building module 'D'}} \
   expected-remark-re{{importing module 'D' from '{{.*[/\\]}}D.pcm'}}
// CHECK: remark: building module 'D'
// CHECK: remark: finished building module 'D'
// CHECK: remark: importing module 'D' from '{{.*[/\\]}}D.pcm'
