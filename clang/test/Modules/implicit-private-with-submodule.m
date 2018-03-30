// RUN: rm -rf %t
// Build PCH using A, with private submodule A.Private
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-submodule -emit-pch -o %t-A.pch %s -DNO_AT_IMPORT

// RUN: rm -rf %t
// Build PCH using A, with private submodule A.Private, check the fixit
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-submodule -emit-pch -o %t-A.pch %s -fdiagnostics-parseable-fixits -DNO_AT_IMPORT 2>&1 | FileCheck %s

// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-submodule -emit-pch -o %t-A.pch %s -DUSE_AT_IMPORT_PRIV
// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-submodule -emit-pch -o %t-A.pch %s -DUSE_AT_IMPORT_BOTH

// expected-warning@Inputs/implicit-private-with-submodule/A.framework/Modules/module.private.modulemap:1{{private submodule 'A.Private' in private module map, expected top-level module}}
// expected-note@Inputs/implicit-private-with-submodule/A.framework/Modules/module.private.modulemap:1{{rename 'A.Private' to ensure it can be found by name}}
// CHECK: fix-it:"{{.*}}module.private.modulemap":{1:18-1:27}:"A_Private"

#ifndef HEADER
#define HEADER

#ifdef NO_AT_IMPORT
#import "A/aprivate.h"
#endif

#ifdef USE_AT_IMPORT_PRIV
@import A.Private;
#endif

#ifdef USE_AT_IMPORT_BOTH
@import A;
@import A.Private;
#endif

const int *y = &APRIVATE;

#endif
