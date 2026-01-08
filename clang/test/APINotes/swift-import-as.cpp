// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter ImmortalRefType | FileCheck -check-prefix=CHECK-IMMORTAL %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter RefCountedType | FileCheck -check-prefix=CHECK-REF-COUNTED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter RefCountedTypeWithDefaultConvention | FileCheck -check-prefix=CHECK-REF-COUNTED-DEFAULT %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter OpaqueRefCountedType | FileCheck -check-prefix=CHECK-OPAQUE-REF-COUNTED %s
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
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter WrappedOptions | FileCheck -check-prefix=CHECK-WRAPPED-OPTIONS %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter NoncopyableWithDestroyType | FileCheck -check-prefix=CHECK-NONCOPYABLE-WITH-DESTROY %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -x c++ -ast-dump -ast-dump-filter ImportAsUnsafe | FileCheck -check-prefix=CHECK-IMPORT-AS-UNSAFE %s

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

// CHECK-REF-COUNTED-DEFAULT: Dumping RefCountedTypeWithDefaultConvention:
// CHECK-REF-COUNTED-DEFAULT-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct RefCountedTypeWithDefaultConvention
// CHECK-REF-COUNTED-DEFAULT: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"
// CHECK-REF-COUNTED-DEFAULT: SwiftAttrAttr {{.+}} <<invalid sloc>> "retain:retain"
// CHECK-REF-COUNTED-DEFAULT: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:release"
// CHECK-REF-COUNTED-DEFAULT: SwiftAttrAttr {{.+}} <<invalid sloc>> "returned_as_unretained_by_default"

// CHECK-OPAQUE-REF-COUNTED: Dumping OpaqueRefCountedType:
// CHECK-OPAQUE-REF-COUNTED-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs{{.*}}struct OpaqueRefCountedType
// CHECK-OPAQUE-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"
// CHECK-OPAQUE-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "retain:ORCRetain"
// CHECK-OPAQUE-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:ORCRelease"
// CHECK-OPAQUE-REF-COUNTED-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:ORCRelease"

// CHECK-OPAQUE-REF-COUNTED: Dumping OpaqueRefCountedType:
// CHECK-OPAQUE-REF-COUNTED-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs{{.*}}struct OpaqueRefCountedType
// CHECK-OPAQUE-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "import_reference"
// CHECK-OPAQUE-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "retain:ORCRetain"
// CHECK-OPAQUE-REF-COUNTED: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:ORCRelease"

// CHECK-OPAQUE-REF-COUNTED-NOT: SwiftAttrAttr {{.+}} <<invalid sloc>> "release:
// CHECK-NON-COPYABLE: Dumping NonCopyableType:
// CHECK-NON-COPYABLE-NEXT: CXXRecordDecl {{.+}} imported in SwiftImportAs {{.+}} struct NonCopyableType
// CHECK-NON-COPYABLE: SwiftAttrAttr {{.+}} <<invalid sloc>> "~Copyable"
// CHECK-NON-COPYABLE: SwiftAttrAttr {{.+}} <<invalid sloc>> "conforms_to:MySwiftModule.MySwiftNonCopyableProtocol"

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

// CHECK-WRAPPED-OPTIONS: Dumping WrappedOptions
// CHECK-WRAPPED-OPTIONS: TypedefDecl{{.*}}WrappedOptions 'unsigned int'
// CHECK-WRAPPED-OPTIONS: SwiftNewTypeAttr {{.*}} swift_wrapper NK_Struct
// CHECK-WRAPPED-OPTIONS: SwiftAttrAttr {{.*}} "conforms_to:Swift.OptionSet"

// CHECK-NONCOPYABLE-WITH-DESTROY: Dumping NoncopyableWithDestroyType
// CHECK-NONCOPYABLE-WITH-DESTROY: RecordDecl {{.*}}struct NoncopyableWithDestroyType
// CHECK-NONCOPYABLE-WITH-DESTROY: SwiftAttrAttr {{.+}} "destroy:NCDDestroy"
// CHECK-NONCOPYABLE-WITH-DESTROY: SwiftAttrAttr {{.+}} "~Copyable"

// CHECK-IMPORT-AS-UNSAFE: Dumping ImportAsUnsafe:
// CHECK-IMPORT-AS-UNSAFE: FunctionDecl {{.+}} ImportAsUnsafe
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "unsafe"

// CHECK-IMPORT-AS-UNSAFE: Dumping ImportAsUnsafeStruct:
// CHECK-IMPORT-AS-UNSAFE: CXXRecordDecl {{.+}} ImportAsUnsafeStruct
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "unsafe"

// CHECK-IMPORT-AS-UNSAFE: Dumping StructWithUnsafeMethod::ImportAsUnsafeMethod:
// CHECK-IMPORT-AS-UNSAFE: CXXMethodDecl {{.+}} ImportAsUnsafeMethod
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "unsafe"

// CHECK-IMPORT-AS-UNSAFE: Dumping StructWithUnsafeMethod::ImportAsUnsafeMethodActuallySafe:
// CHECK-IMPORT-AS-UNSAFE: CXXMethodDecl {{.+}} ImportAsUnsafeMethodActuallySafe
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "safe"

// CHECK-IMPORT-AS-UNSAFE: Dumping ImportAsUnsafeAlreadyAnnotated:
// CHECK-IMPORT-AS-UNSAFE: FunctionDecl {{.+}} ImportAsUnsafeAlreadyAnnotated
// CHECK-IMPORT-AS-UNSAFE: SwiftVersionedAdditionAttr {{.+}} IsReplacedByActive
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "unsafe"
// CHECK-IMPORT-AS-UNSAFE-EMPTY:

// CHECK-IMPORT-AS-UNSAFE: Dumping ImportAsUnsafeVersioned:
// CHECK-IMPORT-AS-UNSAFE: FunctionDecl {{.+}} ImportAsUnsafeVersioned
// CHECK-IMPORT-AS-UNSAFE: SwiftVersionedAdditionAttr {{.+}} 3.0
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "unsafe"
// CHECK-IMPORT-AS-UNSAFE: SwiftVersionedAdditionAttr {{.+}} 6.0
// CHECK-IMPORT-AS-UNSAFE: SwiftAttrAttr {{.+}} "safe"
