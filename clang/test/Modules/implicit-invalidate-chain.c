// RUN: rm -rf %t1 %t2 %t-include
// RUN: mkdir %t-include
// RUN: echo 'module D { header "D.h" }' >> %t-include/module.modulemap

// Run with -verify, which onliy gets remarks from the main TU.
//
// RUN: echo '#define D 0' > %t-include/D.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t1 \
// RUN:     -fdisable-module-hash -fsyntax-only \
// RUN:     -I%S/Inputs/implicit-invalidate-chain -I%t-include \
// RUN:     -Rmodule-build -Rmodule-import %s
// RUN: echo '#define D 11' > %t-include/D.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t1 \
// RUN:     -fdisable-module-hash -fsyntax-only \
// RUN:     -I%S/Inputs/implicit-invalidate-chain -I%t-include \
// RUN:     -Rmodule-build -Rmodule-import -verify %s

// Run again, using FileCheck to check remarks from the module builds.  This is
// the key test: after the first attempt to import an out-of-date 'D', all the
// modules have been invalidated and are not imported again until they are
// rebuilt.
//
// RUN: echo '#define D 0' > %t-include/D.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t2 \
// RUN:     -fdisable-module-hash -fsyntax-only \
// RUN:     -I%S/Inputs/implicit-invalidate-chain -I%t-include \
// RUN:     -Rmodule-build -Rmodule-import %s
// RUN: echo '#define D 11' > %t-include/D.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t2 \
// RUN:     -fdisable-module-hash -fsyntax-only \
// RUN:     -I%S/Inputs/implicit-invalidate-chain -I%t-include \
// RUN:     -Rmodule-build -Rmodule-import %s 2>&1 |\
// RUN: FileCheck %s -implicit-check-not "remark:"

#include "A.h" // \
   expected-remark-re{{importing module 'A' from '{{.*}}/A.pcm'}} \
   expected-remark-re{{importing module 'B' into 'A' from '{{.*}}/B.pcm'}} \
   expected-remark-re{{importing module 'C' into 'B' from '{{.*}}/C.pcm'}} \
   expected-remark-re{{importing module 'D' into 'C' from '{{.*}}/D.pcm'}} \
   expected-remark-re{{building module 'A' as '{{.*}}/A.pcm'}} \
   expected-remark{{finished building module 'A'}} \
   expected-remark-re{{importing module 'A' from '{{.*}}/A.pcm'}} \
   expected-remark-re{{importing module 'B' into 'A' from '{{.*}}/B.pcm'}} \
   expected-remark-re{{importing module 'C' into 'B' from '{{.*}}/C.pcm'}} \
   expected-remark-re{{importing module 'D' into 'C' from '{{.*}}/D.pcm'}}
// CHECK: remark: importing module 'A' from '{{.*}}/A.pcm'
// CHECK: remark: importing module 'B' into 'A' from '{{.*}}/B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '{{.*}}/C.pcm'
// CHECK: remark: importing module 'D' into 'C' from '{{.*}}/D.pcm'
// CHECK: remark: building module 'A'
// CHECK: remark: building module 'B'
// CHECK: remark: building module 'C'
// CHECK: remark: building module 'D'
// CHECK: remark: finished building module 'D'
// CHECK: remark: importing module 'D' from '{{.*}}/D.pcm'
// CHECK: remark: finished building module 'C'
// CHECK: remark: importing module 'C' from '{{.*}}/C.pcm'
// CHECK: remark: importing module 'D' into 'C' from '{{.*}}/D.pcm'
// CHECK: remark: finished building module 'B'
// CHECK: remark: importing module 'B' from '{{.*}}/B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '{{.*}}/C.pcm'
// CHECK: remark: importing module 'D' into 'C' from '{{.*}}/D.pcm'
// CHECK: remark: finished building module 'A'
// CHECK: remark: importing module 'A' from '{{.*}}/A.pcm'
// CHECK: remark: importing module 'B' into 'A' from '{{.*}}/B.pcm'
// CHECK: remark: importing module 'C' into 'B' from '{{.*}}/C.pcm'
// CHECK: remark: importing module 'D' into 'C' from '{{.*}}/D.pcm'
