// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-deprecated-declarations -Wno-return-type -Wno-objc-root-class -ast-dump %s 2>&1 | FileCheck --check-prefixes=COMMON-CHECK,BOUNDS-CHECK %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -Wno-deprecated-declarations -Wno-return-type -Wno-objc-root-class -ast-dump %s 2>&1 | FileCheck --check-prefixes=COMMON-CHECK,ATTR-CHECK %s

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

// COMMON-CHECK-LABEL: -ObjCProtocolDecl {{.*}} CountedByProtocol
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr
// BOUNDS-CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | `-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - reverseParam:: 'void'
// BOUNDS-CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   |   `-DependerDeclsAttr
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr {{.*}} IsDeref
// BOUNDS-CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *__single __counted_by(*len)':'int *__single'
// ATTR-CHECK-NEXT:     | `-ParmVarDecl {{.*}} p 'int * __counted_by(*len)':'int *'
// BOUNDS-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleRet: 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     |-ObjCMethodDecl {{.*}} - simpleRet: 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// BOUNDS-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedRet: 'int *__single __counted_by(*len)':'int *__single'
// ATTR-CHECK-NEXT:     |-ObjCMethodDecl {{.*}} - nestedRet: 'int * __counted_by(*len)':'int *'
// COMMON-CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int *'
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - cParam: 'void'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr
// BOUNDS-CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | `-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   `-ObjCMethodDecl {{.*}} - reverseCParam: 'void'
// BOUNDS-CHECK-NEXT:     |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:       |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:     `-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:       `-DependerDeclsAttr

@interface CountedByClass <CountedByProtocol>
 - (void)simpleParam:(int)len :(int * __counted_by(len))p;
 - (void)reverseParam:(int * __counted_by(len))p :(int)len;
 - (void)nestedParam:(int*)len :(int * __counted_by(*len))p;

 - (int * __counted_by(len))simpleRet:(int)len;
 - (int * __counted_by(*len))nestedRet:(int*)len;

 - (void)cParam:(int)len, int * __counted_by(len) p;
 - (void)reverseCParam:(int * __counted_by(len))p, int len;
@end

// COMMON-CHECK-LABEL: -ObjCInterfaceDecl {{.*}} CountedByClass
// COMMON-CHECK-NEXT:   |-ObjCImplementation {{.*}} 'CountedByClass'
// COMMON-CHECK-NEXT:   |-ObjCProtocol {{.*}} 'CountedByProtocol'
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr
// BOUNDS-CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | `-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - reverseParam:: 'void'
// BOUNDS-CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   |   `-DependerDeclsAttr
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr {{.*}} IsDeref
// BOUNDS-CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *__single __counted_by(*len)':'int *__single'
// ATTR-CHECK-NEXT:     | `-ParmVarDecl {{.*}} p 'int * __counted_by(*len)':'int *'
// BOUNDS-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleRet: 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     |-ObjCMethodDecl {{.*}} - simpleRet: 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int'
// BOUNDS-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedRet: 'int *__single __counted_by(*len)':'int *__single'
// ATTR-CHECK-NEXT:     |-ObjCMethodDecl {{.*}} - nestedRet: 'int * __counted_by(*len)':'int *'
// COMMON-CHECK-NEXT:   | `-ParmVarDecl {{.*}} len 'int *'
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - cParam: 'void'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr
// BOUNDS-CHECK-NEXT:   | `-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | `-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   `-ObjCMethodDecl {{.*}} - reverseCParam: 'void'
// BOUNDS-CHECK-NEXT:     |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:       |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:     `-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:       `-DependerDeclsAttr

@implementation CountedByClass
 - (void)simpleParam:(int)len :(int * __counted_by(len))p {}
 - (void)reverseParam:(int * __counted_by(len))p :(int)len {}
 - (void)nestedParam:(int*)len :(int * __counted_by(*len))p {}

 - (int * __counted_by(len))simpleRet:(int)len {}
 - (int * __counted_by(*len))nestedRet:(int*)len {}

 - (void)cParam:(int)len, int * __counted_by(len) p {}
 - (void)reverseCParam:(int * __counted_by(len))p, int len {}
@end

// COMMON-CHECK-LABEL: -ObjCImplementationDecl {{.*}} CountedByClass
// COMMON-CHECK-NEXT:   |-ObjCInterface {{.*}} 'CountedByClass'
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr
// BOUNDS-CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | `-CompoundStmt
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - reverseParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// BOUNDS-CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   |   `-DependerDeclsAttr
// COMMON-CHECK-NEXT:   | `-CompoundStmt
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedParam:: 'void'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr {{.*}} IsDeref
// BOUNDS-CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *__single __counted_by(*len)':'int *__single'
// ATTR-CHECK-NEXT:     | |-ParmVarDecl {{.*}} p 'int * __counted_by(*len)':'int *'
// COMMON-CHECK-NEXT:   | `-CompoundStmt
// BOUNDS-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - simpleRet: 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     |-ObjCMethodDecl {{.*}} - simpleRet: 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NEXT:   | `-CompoundStmt
// BOUNDS-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - nestedRet: 'int *__single __counted_by(*len)':'int *__single'
// ATTR-CHECK-NEXT:     |-ObjCMethodDecl {{.*}} - nestedRet: 'int * __counted_by(*len)':'int *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int *'
// COMMON-CHECK-NEXT:   | `-CompoundStmt
// COMMON-CHECK-NEXT:   |-ObjCMethodDecl {{.*}} - cParam: 'void'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:   | |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// COMMON-CHECK-NEXT:   | |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:   | | `-DependerDeclsAttr
// BOUNDS-CHECK-NEXT:   | |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:     | |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:   | `-CompoundStmt
// COMMON-CHECK-NEXT:   `-ObjCMethodDecl {{.*}} - reverseCParam: 'void'
// COMMON-CHECK-NEXT:     |-ImplicitParamDecl {{.*}} self 'CountedByClass *'
// COMMON-CHECK-NEXT:     |-ImplicitParamDecl {{.*}} _cmd 'SEL':'SEL *'
// BOUNDS-CHECK-NEXT:     |-ParmVarDecl {{.*}} p 'int *__single __counted_by(len)':'int *__single'
// ATTR-CHECK-NEXT:       |-ParmVarDecl {{.*}} p 'int * __counted_by(len)':'int *'
// COMMON-CHECK-NEXT:     |-ParmVarDecl {{.*}} len 'int'
// COMMON-CHECK-NOT: IsDeref
// COMMON-CHECK-NEXT:       `-DependerDeclsAttr
// COMMON-CHECK-NEXT:     `-CompoundStmt
