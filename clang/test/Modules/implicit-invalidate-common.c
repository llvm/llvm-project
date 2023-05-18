// RUN: rm -rf %t
// RUN: mkdir -p %t/implicit-invalidate-common
// RUN: cp -r %S/Inputs/implicit-invalidate-common %t/
// RUN: echo '#include "A.h"' > %t/A.c
// RUN: echo '#include "B.h"' > %t/B.c

// Build with an empty module cache. Module 'Common' should be built only once.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/Cache \
// RUN:     -fsyntax-only -I %t/implicit-invalidate-common -Rmodule-build \
// RUN:     %t/A.c 2> %t/initial_build.txt
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/Cache \
// RUN:     -fsyntax-only -I %t/implicit-invalidate-common -Rmodule-build \
// RUN:     %t/B.c 2>> %t/initial_build.txt
// RUN: FileCheck %s --implicit-check-not "remark:" --input-file %t/initial_build.txt

// Update module 'Common' and build with the populated module cache. Module
// 'Common' still should be built only once. Note that we are using the same
// flags for A.c and B.c to avoid building Common.pcm at different paths.
//
// RUN: echo ' // ' >> %t/implicit-invalidate-common/Common.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/Cache \
// RUN:     -fsyntax-only -I %t/implicit-invalidate-common -Rmodule-build \
// RUN:     %t/A.c 2> %t/incremental_build.txt
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/Cache \
// RUN:     -fsyntax-only -I %t/implicit-invalidate-common -Rmodule-build \
// RUN:     %t/B.c 2>> %t/incremental_build.txt
// RUN: FileCheck %s --implicit-check-not "remark:" --input-file %t/incremental_build.txt

// CHECK: remark: building module 'A'
// CHECK: remark: building module 'Common'
// CHECK: remark: finished building module 'Common'
// CHECK: remark: finished building module 'A'
// CHECK: remark: building module 'B'
// CHECK: remark: finished building module 'B'
