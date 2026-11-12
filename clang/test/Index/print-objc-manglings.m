// RUN: c-index-test -write-pch %t.macho.ast -target i686-apple-darwin %s
// RUN: c-index-test -test-print-manglings %t.macho.ast | FileCheck --check-prefix=MACHO %s

// RUN: c-index-test -write-pch %t.itanium.ast -target i686-pc-linux-gnu %s
// RUN: c-index-test -test-print-manglings %t.itanium.ast | FileCheck --check-prefix=ITANIUM %s

@interface C
@end

// MACHO: ObjCInterfaceDecl=C{{.*}} [mangled=_OBJC_CLASS_$_C] [mangled=_OBJC_METACLASS_$_C]
// ITANIUM: ObjCInterfaceDecl=C{{.*}} [mangled=_OBJC_CLASS_C] [mangled=_OBJC_METACLASS_C]

@implementation C
@end

// MACHO: ObjCImplementationDecl=C{{.*}} (Definition) [mangled=_OBJC_CLASS_$_C] [mangled=_OBJC_METACLASS_$_C]
// ITANIUM: ObjCImplementationDecl=C{{.*}} (Definition) [mangled=_OBJC_CLASS_C] [mangled=_OBJC_METACLASS_C]

