// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-name=nested -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E -o %t/no-rewrite.ii
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-name=nested -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E -frewrite-includes -o %t/rewrite.ii

// RUN: FileCheck %s --input-file %t/no-rewrite.ii --check-prefix=CHECK --check-prefix=NO-REWRITE
// RUN: FileCheck %s --input-file %t/rewrite.ii    --check-prefix=CHECK --check-prefix=REWRITE

// Check that we can build a module from the preprocessed output.
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-name=nested -x c++-module-map-cpp-output %t/no-rewrite.ii -emit-module -o %t/no-rewrite.pcm
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-name=nested -x c++-module-map-cpp-output %t/rewrite.ii -emit-module -o %t/rewrite.pcm

// Check the module we built works.
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/no-rewrite.pcm %s -I%t -verify -fno-modules-error-recovery
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/rewrite.pcm %s -I%t -verify -fno-modules-error-recovery -DREWRITE

// == module map
// CHECK: # 1 "{{.*}}module.modulemap"
// CHECK: module nested {
// CHECK:   module a {
// CHECK:     header "a.h"
// CHECK:   }
// CHECK:   module b {
// CHECK:     header "b.h"
// CHECK:   }
// CHECK:   module c {
// CHECK:     header "c.h"
// CHECK:   }
// CHECK: }

// NO-REWRITE-NOT: #include
// REWRITE: #include "a.h"

// CHECK: #pragma clang module begin nested.a
// NO-REWRITE-NOT: #include
// REWRITE: #include "c.h"
// CHECK: #pragma clang module begin nested.c
// CHECK: using T = int;
// CHECK: #pragma clang module end
// CHECK: T a();
// CHECK: #pragma clang module end

// CHECK: #pragma clang module begin nested.b
// CHECK: #pragma clang module import nested.c
// CHECK-NOT: #pragma clang module begin nested.c
// CHECK-NOT: using T = int;
// CHECK-NOT: #pragma clang module end
// CHECK: T b();
// CHECK: #pragma clang module end

// CHECK: #pragma clang module import nested.c

#pragma clang module import nested.b

int n = b();
T c; // expected-error {{must be imported}}
#ifdef REWRITE
// expected-note@rewrite.ii:* {{declar}}
#else
// expected-note@no-rewrite.ii:* {{declar}}
#endif
