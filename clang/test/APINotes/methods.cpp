// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper::getIncremented -x c++ | FileCheck --check-prefix=CHECK-METHOD %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper::operator+ -x c++ | FileCheck --check-prefix=CHECK-OPERATOR-PLUS %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper::operator* -x c++ | FileCheck --check-prefix=CHECK-OPERATOR-STAR %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper2::getIncremented -x c++ | FileCheck --check-prefix=CHECK-METHOD-2 %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper3::getIncremented -x c++ | FileCheck --check-prefix=CHECK-METHOD-3 %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Methods -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Outer::Inner::getDecremented -x c++ | FileCheck --check-prefix=CHECK-DEEP-METHOD %s

#include "Methods.h"

// CHECK-METHOD: Dumping IntWrapper::getIncremented:
// CHECK-METHOD-NEXT: CXXMethodDecl {{.+}} getIncremented
// CHECK-METHOD: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-OPERATOR-PLUS: Dumping IntWrapper::operator+:
// CHECK-OPERATOR-PLUS-NEXT: CXXMethodDecl {{.+}} operator+
// CHECK-OPERATOR-PLUS: UnavailableAttr {{.+}} <<invalid sloc>> "oh no, this is an operator"

// CHECK-OPERATOR-STAR: Dumping IntWrapper::operator*:
// CHECK-OPERATOR-STAR-NEXT: CXXMethodDecl {{.+}} operator*
// CHECK-OPERATOR-STAR: UnavailableAttr {{.+}} <<invalid sloc>> "oh no, this is an operator star"

// CHECK-METHOD-2: Dumping IntWrapper2::getIncremented:
// CHECK-METHOD-2-NEXT: CXXMethodDecl {{.+}} getIncremented
// CHECK-METHOD-2: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-METHOD-3: Dumping IntWrapper3::getIncremented:
// CHECK-METHOD-3-NEXT: CXXMethodDecl {{.+}} getIncremented
// CHECK-METHOD-3: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-DEEP-METHOD: Dumping Outer::Inner::getDecremented:
// CHECK-DEEP-METHOD-NEXT: CXXMethodDecl {{.+}} getDecremented
// CHECK-DEEP-METHOD: UnavailableAttr {{.+}} <<invalid sloc>> "nope"

// CHECK-METHOD-NOT: this should have no effect
// CHECK-DEEP-METHOD-NOT: this should have no effect
