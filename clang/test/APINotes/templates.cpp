// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -std=c++17 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Box -x c++ | FileCheck -check-prefix=CHECK-BOX,CHECK-BOX17 %s
// RUN: %clang_cc1 -std=c++17 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter MoveOnly -x c++ | FileCheck -check-prefix=CHECK-MOVEONLY,CHECK-MOVEONLY17 %s

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -std=c++20 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Box -x c++ | FileCheck -check-prefix=CHECK-BOX,CHECK-BOX20 %s
// RUN: %clang_cc1 -std=c++20 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter MoveOnly -x c++ | FileCheck -check-prefix=CHECK-MOVEONLY,CHECK-MOVEONLY20 %s

#include "Templates.h"

// CHECK-BOX: Dumping Box:
// CHECK-BOX17-NEXT: ClassTemplateDecl {{.+}} imported in Templates referenced Box
// CHECK-BOX20-NEXT: ClassTemplateDecl {{.+}} imported in Templates hidden referenced Box
// CHECK-BOX: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_owned"

// Make sure the attributes aren't duplicated.
// CHECK-BOX-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_owned"

// CHECK-MOVEONLY: Dumping MoveOnly:
// CHECK-MOVEONLY17-NEXT: ClassTemplateDecl {{.+}} imported in Templates referenced MoveOnly
// CHECK-MOVEONLY20-NEXT: ClassTemplateDecl {{.+}} imported in Templates hidden referenced MoveOnly
// CHECK-MOVEONLY: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// Make sure the attributes aren't duplicated.
// CHECK-MOVEONLY-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// CHECK-MOVEONLY17: ClassTemplateSpecializationDecl {{.+}} imported in Templates {{.+}} MoveOnly
// CHECK-MOVEONLY20: ClassTemplateSpecializationDecl {{.+}} imported in Templates hidden {{.+}} MoveOnly
// CHECK-MOVEONLY: TemplateArgument type 'int'
// CHECK-MOVEONLY: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// Make sure the attributes aren't duplicated.
// CHECK-MOVEONLY-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// CHECK-MOVEONLY17: ClassTemplateSpecializationDecl {{.+}} imported in Templates {{.+}} MoveOnly
// CHECK-MOVEONLY20: ClassTemplateSpecializationDecl {{.+}} imported in Templates hidden {{.+}} MoveOnly
// CHECK-MOVEONLY: TemplateArgument type 'float'
// CHECK-MOVEONLY: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// Make sure the attributes aren't duplicated.
// CHECK-MOVEONLY-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"
