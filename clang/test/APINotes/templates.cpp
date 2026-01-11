// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter Box -x c++ | FileCheck -check-prefix=CHECK-BOX %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Tmpl -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter MoveOnly -x c++ | FileCheck -check-prefix=CHECK-MOVEONLY %s

#include "Templates.h"

// CHECK-BOX: Dumping Box:
// CHECK-BOX-NEXT: ClassTemplateDecl {{.+}} imported in Templates Box
// CHECK-BOX: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_owned"

// Make sure the attributes aren't duplicated.
// CHECK-BOX-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_owned"

// CHECK-MOVEONLY: Dumping MoveOnly:
// CHECK-MOVEONLY-NEXT: ClassTemplateDecl {{.+}} imported in Templates MoveOnly
// CHECK-MOVEONLY: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// Make sure the attributes aren't duplicated.
// CHECK-MOVEONLY-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// CHECK-MOVEONLY: ClassTemplateSpecializationDecl {{.+}} imported in Templates {{.+}} MoveOnly
// CHECK-MOVEONLY: TemplateArgument type 'int'
// CHECK-MOVEONLY: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// Make sure the attributes aren't duplicated.
// CHECK-MOVEONLY-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// CHECK-MOVEONLY: ClassTemplateSpecializationDecl {{.+}} imported in Templates {{.+}} MoveOnly
// CHECK-MOVEONLY: TemplateArgument type 'float'
// CHECK-MOVEONLY: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// Make sure the attributes aren't duplicated.
// CHECK-MOVEONLY-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"
