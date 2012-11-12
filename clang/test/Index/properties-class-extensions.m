// Test that @properties within class extensions are visited by
// clang_visitChildren only in the class extension, not the original
// @interface (where we have a duplicate declaration - to be removed).
@interface Foo {} @end
@interface Foo (Cat)
@property int a;
@end
@interface Foo ()
@property int b;
- (void) bar;
@end

// Test that 'setter' methods defined by @property in the class extension
// but not the in @interface are only presented in the class extension.
@interface Bar  
@property (readonly) id bar;
@end
@interface Bar () 
@property (readwrite) id bar;
@end

// Another test, this one involving protocols, where the @property should
// not appear in the @interface.
@class Rdar8467189_Bar;
@protocol Rdar8467189_FooProtocol
@property (readonly) Rdar8467189_Bar *Rdar8467189_Bar;
@end
@interface Rdar8467189_Foo <Rdar8467189_FooProtocol>
@end
@interface Rdar8467189_Foo ()
@property (readwrite) Rdar8467189_Bar *Rdar8467189_Bar;
@end

// Test if the @property added in an extension is not reported in the @interface.
@interface Qux
@end
@interface Qux ()
@property (assign, readwrite) id qux;
@end

@implementation Qux
@dynamic qux;
@end

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: properties-class-extensions.m:4:12: ObjCInterfaceDecl=Foo:4:12 Extent=[4:1 - 4:23]
// CHECK-NOT: properties-class-extensions.m:9:15: ObjCInstanceMethodDecl=setB::9:15 Extent=[9:15 - 9:16]
// CHECK-NOT: properties-class-extensions.m:9:15: ParmDecl=b:9:15 (Definition) Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:5:12: ObjCCategoryDecl=Cat:5:12 Extent=[5:1 - 7:5]
// CHECK: properties-class-extensions.m:5:12: ObjCClassRef=Foo:4:12 Extent=[5:12 - 5:15]
// CHECK: properties-class-extensions.m:6:15: ObjCPropertyDecl=a:6:15 Extent=[6:1 - 6:16]
// CHECK: properties-class-extensions.m:6:15: ObjCInstanceMethodDecl=a:6:15 Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:6:15: ObjCInstanceMethodDecl=setA::6:15 Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:6:15: ParmDecl=a:6:15 (Definition) Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:8:12: ObjCCategoryDecl=:8:12 Extent=[8:1 - 11:5]
// CHECK: properties-class-extensions.m:8:12: ObjCClassRef=Foo:4:12 Extent=[8:12 - 8:15]
// CHECK: properties-class-extensions.m:9:15: ObjCPropertyDecl=b:9:15 Extent=[9:1 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ObjCInstanceMethodDecl=b:9:15 Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ObjCInstanceMethodDecl=setB::9:15 Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ParmDecl=b:9:15 (Definition) Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:10:10: ObjCInstanceMethodDecl=bar:10:10 Extent=[10:1 - 10:14]
// CHECK: properties-class-extensions.m:15:12: ObjCInterfaceDecl=Bar:15:12 Extent=[15:1 - 17:5]
// CHECK: properties-class-extensions.m:16:25: ObjCPropertyDecl=bar:16:25 Extent=[16:1 - 16:28]
// CHECK: properties-class-extensions.m:16:22: TypeRef=id:0:0 Extent=[16:22 - 16:24]
// CHECK: properties-class-extensions.m:16:25: ObjCInstanceMethodDecl=bar:16:25 Extent=[16:25 - 16:28]
// CHECK: properties-class-extensions.m:18:12: ObjCCategoryDecl=:18:12 Extent=[18:1 - 20:5]
// CHECK: properties-class-extensions.m:18:12: ObjCClassRef=Bar:15:12 Extent=[18:12 - 18:15]
// CHECK: properties-class-extensions.m:19:26: ObjCPropertyDecl=bar:19:26 Extent=[19:1 - 19:29]
// CHECK: properties-class-extensions.m:19:23: TypeRef=id:0:0 Extent=[19:23 - 19:25]
// CHECK-NOT: properties-class-extensions.m:16:25: ObjCInstanceMethodDecl=bar:16:25 Extent=[16:25 - 16:28]
// CHECK: properties-class-extensions.m:19:26: ObjCInstanceMethodDecl=setBar::19:26 Extent=[19:26 - 19:29]
// CHECK: properties-class-extensions.m:19:26: ParmDecl=bar:19:26 (Definition) Extent=[19:26 - 19:29]
// CHECK: properties-class-extensions.m:24:8: ObjCInterfaceDecl=Rdar8467189_Bar:24:8 Extent=[24:1 - 24:23]
// CHECK: properties-class-extensions.m:24:8: ObjCClassRef=Rdar8467189_Bar:24:8 Extent=[24:8 - 24:23]
// CHECK: properties-class-extensions.m:25:11: ObjCProtocolDecl=Rdar8467189_FooProtocol:25:11 (Definition) Extent=[25:1 - 27:5]
// CHECK: properties-class-extensions.m:26:39: ObjCPropertyDecl=Rdar8467189_Bar:26:39 Extent=[26:1 - 26:54]
// CHECK: properties-class-extensions.m:26:22: ObjCClassRef=Rdar8467189_Bar:24:8 Extent=[26:22 - 26:37]
// CHECK: properties-class-extensions.m:26:39: ObjCInstanceMethodDecl=Rdar8467189_Bar:26:39 Extent=[26:39 - 26:54]
// CHECK: properties-class-extensions.m:28:12: ObjCInterfaceDecl=Rdar8467189_Foo:28:12 Extent=[28:1 - 29:5]
// CHECK: properties-class-extensions.m:28:29: ObjCProtocolRef=Rdar8467189_FooProtocol:25:11 Extent=[28:29 - 28:52]
// CHECK-NOT: properties-class-extensions.m:31:40: ObjCPropertyDecl=Rdar8467189_Bar:31:40 Extent=[31:40 - 31:55]
// CHECK-NOT: properties-class-extensions.m:31:23: ObjCClassRef=Rdar8467189_Bar:24:8 Extent=[31:23 - 31:38]
// CHECK: properties-class-extensions.m:30:12: ObjCCategoryDecl=:30:12 Extent=[30:1 - 32:5]
// CHECK: properties-class-extensions.m:30:12: ObjCClassRef=Rdar8467189_Foo:28:12 Extent=[30:12 - 30:27]
// CHECK: properties-class-extensions.m:31:40: ObjCPropertyDecl=Rdar8467189_Bar:31:40 Extent=[31:1 - 31:55]
// CHECK: properties-class-extensions.m:31:23: ObjCClassRef=Rdar8467189_Bar:24:8 Extent=[31:23 - 31:38]
// CHECK: properties-class-extensions.m:31:40: ObjCInstanceMethodDecl=Rdar8467189_Bar:31:40 [Overrides @26:39] Extent=[31:40 - 31:55]
// CHECK: properties-class-extensions.m:31:40: ObjCInstanceMethodDecl=setRdar8467189_Bar::31:40 Extent=[31:40 - 31:55]
// CHECK: properties-class-extensions.m:31:40: ParmDecl=Rdar8467189_Bar:31:40 (Definition) Extent=[31:40 - 31:55]
// CHECK: properties-class-extensions.m:35:12: ObjCInterfaceDecl=Qux:35:12 Extent=[35:1 - 36:5]
// CHECK: properties-class-extensions.m:37:12: ObjCCategoryDecl=:37:12 Extent=[37:1 - 39:5]
// CHECK: properties-class-extensions.m:37:12: ObjCClassRef=Qux:35:12 Extent=[37:12 - 37:15]
// CHECK: properties-class-extensions.m:38:34: ObjCPropertyDecl=qux:38:34 Extent=[38:1 - 38:37]
// CHECK: properties-class-extensions.m:38:31: TypeRef=id:0:0 Extent=[38:31 - 38:33]
// CHECK: properties-class-extensions.m:38:34: ObjCInstanceMethodDecl=qux:38:34 Extent=[38:34 - 38:37]
// CHECK: properties-class-extensions.m:38:34: ObjCInstanceMethodDecl=setQux::38:34 Extent=[38:34 - 38:37]
// CHECK: properties-class-extensions.m:38:34: ParmDecl=qux:38:34 (Definition) Extent=[38:34 - 38:37]
// CHECK: properties-class-extensions.m:42:10: ObjCDynamicDecl=qux:38:34 (Definition) Extent=[42:1 - 42:13]

