// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/shadow/A1 -I %S/Inputs/shadow/A2 -I %S/Inputs/shadow %s -fsyntax-only 2>&1 | FileCheck %s -check-prefix=REDEFINITION
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/shadow/A1/module.modulemap -fmodule-map-file=%S/Inputs/shadow/A2/module.modulemap %S/Inputs/shadow %s -fsyntax-only 2>&1 | FileCheck %s -check-prefix=REDEFINITION
// REDEFINITION: error: redefinition of module 'A'
// REDEFINITION: note: previously defined

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c-header %S/Inputs/shadow/A1/module.modulemap -emit-module -o %t/A.pcm -fmodule-name=A
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/shadow/A1/module.modulemap -fmodule-file=A=%t/A.pcm -I %S/Inputs/shadow %s -verify

#import "A1/A1.h"
#import "A2/A2.h"
@import A;

#import "A2/A.h" // expected-note {{implicitly imported}}
// expected-error@A2/module.modulemap:1 {{import of shadowed module 'A'}}
// expected-note@A1/module.modulemap:1 {{previous definition}}

#if defined(A2_A_h)
#error got the wrong definition of module A
#elif !defined(A1_A_h)
#error missing definition from A1
#endif
