// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter ImmortalRefType | FileCheck -check-prefix=CHECK-IMMORTAL %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter RefCountedType | FileCheck -check-prefix=CHECK-REF-COUNTED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter NonCopyableType | FileCheck -check-prefix=CHECK-NON-COPYABLE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter CopyableType | FileCheck -check-prefix=CHECK-COPYABLE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter NonEscapableType | FileCheck -check-prefix=CHECK-NON-ESCAPABLE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter EscapableType | FileCheck -check-prefix=CHECK-ESCAPABLE %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter functionReturningFrt__ | FileCheck -check-prefix=CHECK-FUNCTION-RETURNING-FRT %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter functionReturningFrt_returns_unretained | FileCheck -check-prefix=CHECK-FUNCTION-RETURNING-FRT-UNRETAINED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter functionReturningFrt_returns_retained | FileCheck -check-prefix=CHECK-FUNCTION-RETURNING-FRT-RETAINED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter methodReturningFrt__ | FileCheck -check-prefix=CHECK-METHOD-RETURNING-FRT %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter methodReturningFrt_returns_unretained | FileCheck -check-prefix=CHECK-METHOD-RETURNING-FRT-UNRETAINED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter methodReturningFrt_returns_retained | FileCheck -check-prefix=CHECK-METHOD-RETURNING-FRT-RETAINED %s

#include <SwiftImportAs.h>

// CHECK-IMMORTAL: Dumping ImmortalRefType:
// CHECK-IMMORTAL-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct ImmortalRefType
// CHECK-IMMORTAL: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"

// CHECK-REF-COUNTED: Dumping RefCountedType:
// CHECK-REF-COUNTED-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct RefCountedType
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "retain:RCRetain"
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:RCRelease"
// CHECK-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "conforms_to:MySwiftModule.MySwiftRefCountedProtocol"

// CHECK-NON-COPYABLE: Dumping NonCopyableType:
// CHECK-NON-COPYABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct NonCopyableType
// CHECK-NON-COPYABLE: SwiftAttrAttr {{.+}} <<invalid sloc>> "conforms_to:MySwiftModule.MySwiftNonCopyableProtocol"
// CHECK-NON-COPYABLE: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"

// CHECK-COPYABLE: Dumping CopyableType:
// CHECK-COPYABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct CopyableType
// CHECK-COPYABLE-NOT: SwiftAttrAttr

// CHECK-NON-ESCAPABLE: Dumping NonEscapableType:
// CHECK-NON-ESCAPABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct NonEscapableType
// CHECK-NON-ESCAPABLE: SwiftAttrAttr {{.+}} "~Escapable"

// CHECK-ESCAPABLE: Dumping EscapableType:
// CHECK-ESCAPABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct EscapableType
// CHECK-ESCAPABLE: SwiftAttrAttr {{.+}} "Escapable"

// CHECK-FUNCTION-RETURNING-FRT: Dumping functionReturningFrt__:
// CHECK-FUNCTION-RETURNING-FRT: FunctionDecl {{.+}} imported in SwiftImportAs functionReturningFrt__ 'ImmortalRefType *()'
// CHECK-FUNCTION-RETURNING-FRT-NOT: `-SwiftAttrAttr {{.+}} "returns_unretained"
// CHECK-FUNCTION-RETURNING-FRT-NOT: `-SwiftAttrAttr {{.+}} "returns_retained"

// CHECK-FUNCTION-RETURNING-FRT-UNRETAINED: Dumping functionReturningFrt_returns_unretained:
// CHECK-FUNCTION-RETURNING-FRT-UNRETAINED: FunctionDecl {{.+}} imported in SwiftImportAs functionReturningFrt_returns_unretained 'ImmortalRefType *()'
// CHECK-FUNCTION-RETURNING-FRT-UNRETAINED: `-SwiftAttrAttr {{.+}} "returns_unretained"

// CHECK-FUNCTION-RETURNING-FRT-RETAINED: Dumping functionReturningFrt_returns_retained:
// CHECK-FUNCTION-RETURNING-FRT-RETAINED: FunctionDecl {{.+}} imported in SwiftImportAs functionReturningFrt_returns_retained 'ImmortalRefType *()'
// CHECK-FUNCTION-RETURNING-FRT-RETAINED: `-SwiftAttrAttr {{.+}} "returns_retained"

// CHECK-METHOD-RETURNING-FRT: Dumping ImmortalRefType::methodReturningFrt__:
// CHECK-METHOD-RETURNING-FRT: CXXMethodDecl {{.+}} imported in SwiftImportAs methodReturningFrt__ 'ImmortalRefType *()'
// CHECK-METHOD-RETURNING-FRT-NOT: `-SwiftAttrAttr {{.+}} "returns_unretained"
// CHECK-METHOD-RETURNING-FRT-NOT: `-SwiftAttrAttr {{.+}} "returns_retained"

// CHECK-METHOD-RETURNING-FRT-UNRETAINED: Dumping ImmortalRefType::methodReturningFrt_returns_unretained:
// CHECK-METHOD-RETURNING-FRT-UNRETAINED: CXXMethodDecl {{.+}} imported in SwiftImportAs methodReturningFrt_returns_unretained 'ImmortalRefType *()'
// CHECK-METHOD-RETURNING-FRT-UNRETAINED: `-SwiftAttrAttr {{.+}} "returns_unretained"

// CHECK-METHOD-RETURNING-FRT-RETAINED: Dumping ImmortalRefType::methodReturningFrt_returns_retained:
// CHECK-METHOD-RETURNING-FRT-RETAINED: CXXMethodDecl {{.+}} imported in SwiftImportAs methodReturningFrt_returns_retained 'ImmortalRefType *()'
// CHECK-METHOD-RETURNING-FRT-RETAINED: `-SwiftAttrAttr {{.+}} "returns_retained"
