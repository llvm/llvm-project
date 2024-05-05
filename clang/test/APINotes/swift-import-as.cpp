// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter ImmortalRefType | FileCheck -check-prefix=CHECK-IMMORTAL %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter RefCountedType | FileCheck -check-prefix=CHECK-REF-COUNTED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter NonCopyableType | FileCheck -check-prefix=CHECK-NON-COPYABLE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter CopyableType | FileCheck -check-prefix=CHECK-COPYABLE %s

#include <SwiftImportAs.h>

// CHECK-IMMORTAL: Dumping ImmortalRefType:
// CHECK-IMMORTAL-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct ImmortalRefType
// CHECK-IMMORTAL: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"

// CHECK-REF-COUNTED: Dumping RefCountedType:
// CHECK-REF-COUNTED-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct RefCountedType
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "retain:RCRetain"
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:RCRelease"

// CHECK-NON-COPYABLE: Dumping NonCopyableType:
// CHECK-NON-COPYABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct NonCopyableType
// CHECK-NON-COPYABLE: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// CHECK-COPYABLE: Dumping CopyableType:
// CHECK-COPYABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct CopyableType
// CHECK-COPYABLE-NOT: SwiftAttrAttr
