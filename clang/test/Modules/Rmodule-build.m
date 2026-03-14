// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '// A' > %t/A.h
// RUN: echo '#include "C.h"' > %t/B.h
// RUN: echo '// C' > %t/C.h
// RUN: echo 'module A { header "A.h" }' > %t/module.modulemap
// RUN: echo 'module B { header "B.h" }' >> %t/module.modulemap
// RUN: echo 'module C { header "C.h" }' >> %t/module.modulemap

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fsyntax-only %s -verify \
// RUN:            -I %t -Rmodule-build

@import A; // expected-remark{{building module 'A' as}} expected-remark {{finished building module 'A'}}
@import B; // expected-remark{{building module 'B' as}} expected-remark {{finished building module 'B'}}
@import A; // no diagnostic
@import B; // no diagnostic

// RUN: echo ' ' >> %t/C.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fsyntax-only %s -I %t \
// RUN:            -Rmodule-build 2>&1 | FileCheck %s

// RUN: echo ' ' >> %t/B.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fsyntax-only %s -I %t \
// RUN:            2>&1 | FileCheck -allow-empty -check-prefix=NO-REMARKS %s

// RUN: echo ' ' >> %t/B.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fsyntax-only %s -I %t \
// RUN:            -Rmodule-build -Rno-everything 2>&1 | \
// RUN:    FileCheck -allow-empty -check-prefix=NO-REMARKS %s

// CHECK-NOT: building module 'A'
// CHECK: building module 'B'
// CHECK: building module 'C'
// CHECK: finished building module 'C'
// CHECK: finished building module 'B'
// NO-REMARKS-NOT: building module 'A'
// NO-REMARKS-NOT: building module 'B'
