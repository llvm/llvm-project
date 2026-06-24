// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapperStruct -x c++ | FileCheck --check-prefix=CHECK-STRUCT %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapperStruct::value -x c++ | FileCheck --check-prefix=CHECK-FIELD %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter IntWrapperStruct::do_something -x c++ | FileCheck --check-prefix=CHECK-METHOD %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter some_operation -x c++ | FileCheck --check-prefix=CHECK-FUNC %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter global_int -x c++ | FileCheck --check-prefix=CHECK-VAR %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/SwiftAttributes -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter int_typedef -x c++ | FileCheck --check-prefix=CHECK-TYPEDEF %s

#include "SwiftAttributes.h"

// CHECK-STRUCT: Dumping IntWrapperStruct:
// CHECK-STRUCT-NEXT: CXXRecordDecl {{.+}} struct IntWrapperStruct
// CHECK-STRUCT: SwiftAttrAttr {{.+}} <<invalid sloc>> "some Swift struct attribute"

// CHECK-FIELD: Dumping IntWrapperStruct::value:
// CHECK-FIELD-NEXT: FieldDecl {{.+}} value
// CHECK-FIELD: SwiftAttrAttr {{.+}} <<invalid sloc>> "some Swift field attribute"

// CHECK-METHOD: Dumping IntWrapperStruct::do_something:
// CHECK-METHOD-NEXT: CXXMethodDecl {{.+}} do_something
// CHECK-METHOD: SwiftAttrAttr {{.+}} <<invalid sloc>> "some Swift struct method attribute"

// CHECK-FUNC: Dumping some_operation:
// CHECK-FUNC-NEXT: FunctionDecl {{.+}} some_operation
// CHECK-FUNC: SwiftAttrAttr {{.+}} <<invalid sloc>> "some Swift function attribute"
// CHECK-FUNC: SwiftAttrAttr {{.+}} <<invalid sloc>> "some other Swift function attribute"

// CHECK-VAR: Dumping global_int:
// CHECK-VAR-NEXT: VarDecl {{.+}} global_int
// CHECK-VAR: SwiftAttrAttr {{.+}} <<invalid sloc>> "some Swift global variable attribute"

// CHECK-TYPEDEF: Dumping int_typedef:
// CHECK-TYPEDEF-NEXT: TypedefDecl {{.+}} int_typedef
// CHECK-TYPEDEF: SwiftAttrAttr {{.+}} <<invalid sloc>> "some Swift typedef attribute"
