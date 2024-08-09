// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper::getIncremented -x c++ | FileCheck --check-prefix=CHECK-METHOD %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Outer::Inner::getDecremented -x c++ | FileCheck --check-prefix=CHECK-DEEP-METHOD %s

#include "Methods.h"

// CHECK-METHOD: Dumping IntWrapper::getIncremented:
// CHECK-METHOD-NEXT: CXXMethodDecl {{.+}} getIncremented
// CHECK-METHOD: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-DEEP-METHOD: Dumping Outer::Inner::getDecremented:
// CHECK-DEEP-METHOD-NEXT: CXXMethodDecl {{.+}} getDecremented
// CHECK-DEEP-METHOD: UnavailableAttr {{.+}} <<invalid sloc>> "nope"

// CHECK-METHOD-NOT: this should have no effect
// CHECK-DEEP-METHOD-NOT: this should have no effect
