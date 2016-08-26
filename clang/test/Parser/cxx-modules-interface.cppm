// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.pcm -verify -DTEST=0
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.pcm -verify -Dmodule=int -DTEST=1
// RUN: not %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -fmodule-file=%t.pcm -o %t.pcm -DTEST=2 2>&1 | FileCheck %s --check-prefix=CHECK-2
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -fmodule-file=%t.pcm -o %t.pcm -verify -Dfoo=bar -DTEST=3

#if TEST == 0
// expected-no-diagnostics
#endif

module foo;
#if TEST == 1
// expected-error@-2 {{expected module declaration at start of module interface}}
#elif TEST == 2
// CHECK-2: error: redefinition of module 'foo'
#endif

int n;
#if TEST == 3
// expected-error@-2 {{redefinition of 'n'}}
// expected-note@-3 {{previous}}
#endif
