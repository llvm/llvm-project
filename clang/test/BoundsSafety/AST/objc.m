// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-deprecated-declarations -Wno-return-type -Wno-objc-root-class -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -Wno-deprecated-declarations -Wno-return-type -Wno-objc-root-class -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

@protocol CountedByProtocol
 - (void)simpleParam:(int)len :(int * __counted_by(len))p;
 - (void)reverseParam:(int * __counted_by(len))p :(int)len;
 - (void)nestedParam:(int*)len :(int * __counted_by(*len))p;

 - (int * __counted_by(len))simpleRet:(int)len;
 - (int * __counted_by(*len))nestedRet:(int*)len;

 - (void)cParam:(int)len, int * __counted_by(len) p;
 - (void)reverseCParam:(int * __counted_by(len))p, int len;
@end

// CHECK-LABEL: -ObjCProtocolDecl {{.*}} CountedByProtocol
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleParam:: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   | | `-DependerDeclsAttr
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - reverseParam:: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   |   `-DependerDeclsAttr
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedParam:: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// CHECK-NEXT:   | | `-DependerDeclsAttr {{.*}} IsDeref
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(*len)':'int *{{(__single)?}}'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleRet: 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedRet: 'int *{{(__single)?}} __counted_by(*len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int *'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - cParam: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   | | `-DependerDeclsAttr
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   `-ObjCMethodDecl {{.*}} - reverseCParam: 'void'
// CHECK-NEXT:     |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:     `-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:       `-DependerDeclsAttr

@interface CountedByClass <CountedByProtocol>
 - (void)simpleParam:(int)len :(int * __counted_by(len))p;
 - (void)reverseParam:(int * __counted_by(len))p :(int)len;
 - (void)nestedParam:(int*)len :(int * __counted_by(*len))p;

 - (int * __counted_by(len))simpleRet:(int)len;
 - (int * __counted_by(*len))nestedRet:(int*)len;

 - (void)cParam:(int)len, int * __counted_by(len) p;
 - (void)reverseCParam:(int * __counted_by(len))p, int len;
@end

// CHECK-LABEL: -ObjCInterfaceDecl {{.*}} CountedByClass
// CHECK-NEXT:   |-ObjCImplementation {{.*}} 'CountedByClass'
// CHECK-NEXT:   |-ObjCProtocol {{.*}} 'CountedByProtocol'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleParam:: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   | | `-DependerDeclsAttr
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - reverseParam:: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   |   `-DependerDeclsAttr
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedParam:: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// CHECK-NEXT:   | | `-DependerDeclsAttr {{.*}} IsDeref
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(*len)':'int *{{(__single)?}}'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleRet: 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedRet: 'int *{{(__single)?}} __counted_by(*len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int *'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - cParam: 'void'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   | | `-DependerDeclsAttr
// CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   `-ObjCMethodDecl {{.*}} - reverseCParam: 'void'
// CHECK-NEXT:     |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:     `-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:       `-DependerDeclsAttr

@implementation CountedByClass
 - (void)simpleParam:(int)len :(int * __counted_by(len))p {}
 - (void)reverseParam:(int * __counted_by(len))p :(int)len {}
 - (void)nestedParam:(int*)len :(int * __counted_by(*len))p {}

 - (int * __counted_by(len))simpleRet:(int)len {}
 - (int * __counted_by(*len))nestedRet:(int*)len {}

 - (void)cParam:(int)len, int * __counted_by(len) p {}
 - (void)reverseCParam:(int * __counted_by(len))p, int len {}
@end

// CHECK-LABEL: -ObjCImplementationDecl {{.*}} CountedByClass
// CHECK-NEXT:   |-ObjCInterface {{.*}} 'CountedByClass'
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleParam:: 'void'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   | | `-DependerDeclsAttr
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-CompoundStmt
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - reverseParam:: 'void'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   |   `-DependerDeclsAttr
// CHECK-NEXT:   | `-CompoundStmt
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedParam:: 'void'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// CHECK-NEXT:   | | `-DependerDeclsAttr {{.*}} IsDeref
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(*len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-CompoundStmt
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleRet: 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NEXT:   | `-CompoundStmt
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedRet: 'int *{{(__single)?}} __counted_by(*len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// CHECK-NEXT:   | `-CompoundStmt
// CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - cParam: 'void'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:   | | `-DependerDeclsAttr
// CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:   | `-CompoundStmt
// CHECK-NEXT:   `-ObjCMethodDecl {{.*}} - reverseCParam: 'void'
// CHECK-NEXT:     |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// CHECK-NEXT:     |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// CHECK-NEXT:     |-ParmVarDecl {{.*}} p 'int *{{(__single)?}} __counted_by(len)':'int *{{(__single)?}}'
// CHECK-NEXT:     |-ParmVarDecl {{.*}} len 'int'
// CHECK-NOT: IsDeref
// CHECK-NEXT:       `-DependerDeclsAttr
// CHECK-NEXT:     `-CompoundStmt
