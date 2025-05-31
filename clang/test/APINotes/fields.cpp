// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Fields -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Fields -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapper::value -x c++ | FileCheck --check-prefix=CHECK-FIELD %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Fields -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Outer::Inner::value -x c++ | FileCheck --check-prefix=CHECK-DEEP-FIELD %s

#include "Fields.h"

// CHECK-FIELD: Dumping IntWrapper::value:
// CHECK-FIELD-NEXT: FieldDecl {{.+}} value
// CHECK-FIELD: UnavailableAttr {{.+}} <<invalid sloc>> "oh no"

// CHECK-DEEP-FIELD: Dumping Outer::Inner::value:
// CHECK-DEEP-FIELD-NEXT: FieldDecl {{.+}} value
// CHECK-DEEP-FIELD: UnavailableAttr {{.+}} <<invalid sloc>> "oh no 2"

// CHECK-FIELD-NOT: this should have no effect
// CHECK-DEEP-FIELD-NOT: this should have no effect
