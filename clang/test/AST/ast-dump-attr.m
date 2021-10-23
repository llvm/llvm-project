// Test without serialization:
// RUN: %clang_cc1 -fdouble-square-bracket-attributes -triple x86_64-apple-macosx10.10.0 \
// RUN: -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -fdouble-square-bracket-attributes -triple x86_64-apple-macosx10.10.0 \
// RUN: -emit-pch -o %t %s
// RUN: %clang_cc1 -x objective-c -fdouble-square-bracket-attributes -triple x86_64-apple-macosx10.10.0 \
// RUN: -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

@interface NSObject
@end

[[clang::objc_exception]]
@interface Test1 {
// CHECK: ObjCInterfaceDecl{{.*}} Test1
// CHECK-NEXT: ObjCExceptionAttr{{.*}}
  [[clang::iboutlet]] NSObject *Test2;
// CHECK: ObjCIvarDecl{{.*}} Test2
// CHECK-NEXT: IBOutletAttr
}
@property (readonly) [[clang::objc_returns_inner_pointer]] void *Test3, *Test4;
// CHECK: ObjCPropertyDecl{{.*}} Test3 'void *' readonly
// CHECK-NEXT: ObjCReturnsInnerPointerAttr
// CHECK-NEXT: ObjCPropertyDecl{{.*}} Test4 'void *' readonly
// CHECK-NEXT: ObjCReturnsInnerPointerAttr

@property (readonly) [[clang::iboutlet]] NSObject *Test5;
// CHECK: ObjCPropertyDecl{{.*}} Test5 'NSObject *' readonly
// CHECK-NEXT: IBOutletAttr

// CHECK: ObjCMethodDecl{{.*}} implicit{{.*}} Test3
// CHECK-NEXT: ObjCReturnsInnerPointerAttr
// CHECK: ObjCMethodDecl{{.*}} implicit{{.*}} Test4
// CHECK-NEXT: ObjCReturnsInnerPointerAttr
// CHECK: ObjCMethodDecl{{.*}} implicit{{.*}} Test5
// CHECK-NOT: IBOutletAttr
@end

[[clang::objc_runtime_name("name")]] @protocol Test6;
// CHECK: ObjCProtocolDecl{{.*}} Test6
// CHECK-NEXT: ObjCRuntimeNameAttr{{.*}} "name"

[[clang::objc_protocol_requires_explicit_implementation]]
@protocol Test7
// CHECK: ObjCProtocolDecl{{.*}} Test7
// CHECK-NEXT: ObjCExplicitProtocolImplAttr
@end

@interface Test8
// CHECK: ObjCInterfaceDecl{{.*}} Test8
-(void)Test9 [[clang::ns_consumes_self]];
// CHECK: ObjCMethodDecl{{.*}} Test9 'void'
// CHECK-NEXT: NSConsumesSelfAttr
-(void) [[clang::ns_consumes_self]] Test10: (int)Test11;
// CHECK: ObjCMethodDecl{{.*}} Test10: 'void'
// CHECK-NEXT: |-ParmVarDecl{{.*}} Test11 'int'
// CHECK-NEXT: `-NSConsumesSelfAttr
-(void)Test12: (int *) [[clang::noescape]] Test13  to:(int)Test14 [[clang::ns_consumes_self]];
// CHECK: ObjCMethodDecl{{.*}} Test12:to: 'void'
// CHECK-NEXT: |-ParmVarDecl{{.*}} Test13 'int *'
// CHECK-NEXT: | `-NoEscapeAttr
// CHECK-NEXT: |-ParmVarDecl{{.*}} Test14 'int'
// CHECK-NEXT: `-NSConsumesSelfAttr
@end
