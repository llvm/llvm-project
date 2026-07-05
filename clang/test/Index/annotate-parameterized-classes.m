@protocol NSObject
@end

@interface NSObject
@end

@interface A<T : id, U : NSObject *> : NSObject
@end

@interface A<T : id, U : NSObject *> (Cat1)
@end

typedef A<id<NSObject>, NSObject *> ASpecialization1;

@interface B<T : id, U : NSObject *> : A<T, U>
@end

// RUN: c-index-test -test-annotate-tokens=%s:7:1:9:1 %s -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-INTERFACE-DECL %s
// CHECK-INTERFACE-DECL: Identifier: "T" [7:14 - 7:15] TemplateTypeParameter=T:7:14
// CHECK-INTERFACE-DECL: Identifier: "id" [7:18 - 7:20] TypeRef=id:0:0
// CHECK-INTERFACE-DECL: Identifier: "U" [7:22 - 7:23] TemplateTypeParameter=U:7:22
// CHECK-INTERFACE-DECL: Identifier: "NSObject" [7:26 - 7:34] ObjCClassRef=NSObject:4:12

// RUN: c-index-test -test-annotate-tokens=%s:10:1:12:1 %s -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-CATEGORY-DECL %s
// CHECK-CATEGORY-DECL: Identifier: "T" [10:14 - 10:15] TemplateTypeParameter=T:10:14 
// CHECK-CATEGORY-DECL: Identifier: "id" [10:18 - 10:20] TypeRef=id:0:0
// CHECK-CATEGORY-DECL: Identifier: "U" [10:22 - 10:23] TemplateTypeParameter=U:10:22
// CHECK-CATEGORY-DECL: Identifier: "NSObject" [10:26 - 10:34] ObjCClassRef=NSObject:4:12

// RUN: c-index-test -test-annotate-tokens=%s:13:1:14:1 %s -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-SPECIALIZATION %s
// CHECK-SPECIALIZATION: Identifier: "id" [13:11 - 13:13] TypeRef=id:0:0
// CHECK-SPECIALIZATION: Identifier: "NSObject" [13:14 - 13:22] ObjCProtocolRef=NSObject:1:11
// CHECK-SPECIALIZATION: Identifier: "NSObject" [13:25 - 13:33] ObjCClassRef=NSObject:4:12

// RUN: c-index-test -test-annotate-tokens=%s:15:1:16:1 %s -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-SUPER %s
// CHECK-SUPER: Identifier: "A" [15:40 - 15:41] ObjCSuperClassRef=A:7:12
// CHECK-SUPER: Identifier: "T" [15:42 - 15:43] TypeRef=T:15:14
// CHECK-SUPER: Identifier: "U" [15:45 - 15:46] TypeRef=U:15:22
