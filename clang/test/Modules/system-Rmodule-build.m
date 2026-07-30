// RUN: rm -rf %t
// RUN: mkdir -p %t/sys
// RUN: echo '#include <B.h>' > %t/sys/A.h
// RUN: echo '' > %t/sys/B.h
// RUN: echo 'module A { header "A.h" }' > %t/sys/module.modulemap
// RUN: echo 'module B { header "B.h" }' >> %t/sys/module.modulemap

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fsyntax-only %s \
// RUN:            -isystem %t/sys -Rmodule-build 2>&1 | FileCheck %s

@import A;

// CHECK: building module 'A' as
// CHECK: building module 'B' as
// CHECK: finished building module 'B'
// CHECK: finished building module 'A'
