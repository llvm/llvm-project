// Test is line- and column-sensitive. Run lines are below.

@interface rdar9771715
@property (readonly) int foo1;
@property (readwrite) int foo2;
@end

@class Foo;

@interface rdar9535717 {
  __weak Foo *foo;
}
@end

@interface Test1 {
  id _name;
}
@end
@interface Test1 ()
- (id)name;
@end
@interface Test1 ()
@property (copy) id name;
@end
@implementation Test1
@synthesize name = _name;
@end

@interface rdar10902015
@end

@implementation rdar10902015

struct S { int x; };

-(void)mm:(struct S*)s {
  rdar10902015 *i = 0;
  s->x = 0;
  Test1 *test1;
  test1.name = 0;
}
@end

@interface Test2
-(int)implicitProp;
-(void)setImplicitProp:(int)x;
@end

void foo1(Test2 *test2) {
  int x = test2.implicitProp;
  test2.implicitProp = x;
  ++test2.implicitProp;
}

@interface Test3
-(void)setFoo:(int)x withBar:(int)y;
@end

void foo3(Test3 *test3) {
  [test3 setFoo:2 withBar:4];
}

@interface Test4
@end
@interface Test4(Dido)
@end
@implementation Test4(Dido)
@end

@class Forw1, Forw2, Forw3;

@interface Test5 {
  id prop1;
  id prop2;
}
@property (assign) id prop1;
@property (assign) id prop2;
@property (assign) id prop3;
@property (assign) id prop4;
@end

@implementation Test5
@synthesize prop1, prop2;
@dynamic prop3, prop4;

-(id)meth1 {
  return 0;
}
-(id)meth2{
  return 0;
}
@end

@interface Test6
@property (assign) id prop1;
@end

@implementation Test6
@synthesize prop1 = _prop1;
@end

@protocol TestProt
-(void)protMeth1;
@property (retain) id propProp1;

@optional
-(void)protMeth2;
@property (retain) id propProp2;

@required
-(void)protMeth3;
@property (retain) id propProp3;
@end

@interface TestNullability
@property (strong, nonnull) id prop1;
@property (strong, nullable) id prop2;
@end

@implementation TestNullability
- (void)meth {
  TestNullability *o;
  [o.prop1 meth];
  [o.prop2 meth];
  _Nullable id lo1;
  _Nonnull id lo2;
  [lo1 meth];
  [lo2 meth];
}
@end

#define NS_ENUM(_name, _type) enum _name : _type _name; enum _name : _type
typedef NS_ENUM(TestTransparent, int) {
  TestTransparentFirst = 0,
  TestTransparentSecond = 1,
};
typedef enum TestTransparent NotTransparent;

TestTransparent transparentTypedef;
enum TestTransparent transparentUnderlying;
NotTransparent opaqueTypedef;

#define MY_ENUM(_name, _type) enum _name : _type _name##_t; enum _name : _type
typedef MY_ENUM(TokenPaste, int) {
  TokenPasteFirst = 0,
};
TokenPaste_t opaqueTypedef2;

#define MY_TYPE(_name) struct _name _name; struct _name
typedef MY_TYPE(SomeT) { int x; };
SomeT someVar;

#define MY_TYPE2(_name) struct _name *_name; struct _name
typedef MY_TYPE2(SomeT2) { int x; };
SomeT2 someVar2;

#define GEN_DECL(mod_name) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name, generated_declaration)))

GEN_DECL("some_module")
@interface ExtCls
-(void)method;
@end

// RUN: c-index-test -cursor-at=%s:4:28 -cursor-at=%s:5:28 %s | FileCheck -check-prefix=CHECK-PROP %s
// CHECK-PROP: ObjCPropertyDecl=foo1:4:26
// CHECK-PROP: ObjCPropertyDecl=foo2:5:27

// RUN: c-index-test -cursor-at=%s:11:11 %s -target x86_64-apple-macosx10.7.0 | FileCheck -check-prefix=CHECK-WITH-WEAK %s
// CHECK-WITH-WEAK: ObjCClassRef=Foo:8:8

// RUN: c-index-test -cursor-at=%s:20:10 %s | FileCheck -check-prefix=CHECK-METHOD %s
// CHECK-METHOD: 20:7 ObjCInstanceMethodDecl=name:20:7 Extent=[20:1 - 20:12]

// RUN: c-index-test -cursor-at=%s:37:17 %s | FileCheck -check-prefix=CHECK-IN-IMPL %s
// CHECK-IN-IMPL: VarDecl=i:37:17

// RUN: c-index-test -cursor-at=%s:38:6 -cursor-at=%s:40:11 \
// RUN:   -cursor-at=%s:50:20 -cursor-at=%s:51:15 -cursor-at=%s:52:20 %s | FileCheck -check-prefix=CHECK-MEMBERREF %s
// CHECK-MEMBERREF: 38:6 MemberRefExpr=x:34:16 SingleRefName=[38:6 - 38:7] RefName=[38:6 - 38:7] Extent=[38:3 - 38:7]
// CHECK-MEMBERREF: 40:9 MemberRefExpr=name:23:21 Extent=[40:3 - 40:13] Spelling=name ([40:9 - 40:13])
// CHECK-MEMBERREF: 50:17 MemberRefExpr=implicitProp:45:7 Extent=[50:11 - 50:29] Spelling=implicitProp ([50:17 - 50:29])
// CHECK-MEMBERREF: 51:9 MemberRefExpr=setImplicitProp::46:8 Extent=[51:3 - 51:21] Spelling=setImplicitProp: ([51:9 - 51:21])
// CHECK-MEMBERREF: 52:11 MemberRefExpr=setImplicitProp::46:8 Extent=[52:5 - 52:23] Spelling=setImplicitProp: ([52:11 - 52:23])

// RUN: c-index-test -cursor-at=%s:56:24 -cursor-at=%s:60:14 \
// RUN:   -cursor-at=%s:65:20 -cursor-at=%s:67:25 \
// RUN:   -cursor-at=%s:70:10 -cursor-at=%s:70:16 -cursor-at=%s:70:25 \
// RUN:   %s | FileCheck -check-prefix=CHECK-SPELLRANGE %s
// CHECK-SPELLRANGE: 56:8 ObjCInstanceMethodDecl=setFoo:withBar::56:8 Extent=[56:1 - 56:37] Spelling=setFoo:withBar: ([56:8 - 56:14][56:22 - 56:29]) Selector index=1
// CHECK-SPELLRANGE: 60:3 ObjCMessageExpr=setFoo:withBar::56:8 Extent=[60:3 - 60:29] Spelling=setFoo:withBar: ([60:10 - 60:16][60:19 - 60:26]) Selector index=0
// CHECK-SPELLRANGE: 65:12 ObjCCategoryDecl=Dido:65:12 Extent=[65:1 - 66:5] Spelling=Dido ([65:18 - 65:22])
// CHECK-SPELLRANGE: 67:17 ObjCCategoryImplDecl=Dido:67:17 (Definition) Extent=[67:1 - 68:2] Spelling=Dido ([67:23 - 67:27])

// CHECK-SPELLRANGE: 70:8 ObjCClassRef=Forw1:70:8 Extent=[70:8 - 70:13] Spelling=Forw1 ([70:8 - 70:13])
// CHECK-SPELLRANGE: 70:15 ObjCClassRef=Forw2:70:15 Extent=[70:15 - 70:20] Spelling=Forw2 ([70:15 - 70:20])
// CHECK-SPELLRANGE: 70:22 ObjCClassRef=Forw3:70:22 Extent=[70:22 - 70:27] Spelling=Forw3 ([70:22 - 70:27])

// RUN: c-index-test -cursor-at=%s:83:15 -cursor-at=%s:83:21 \
// RUN:              -cursor-at=%s:84:12 -cursor-at=%s:84:20 \
// RUN:              -cursor-at=%s:99:14 -cursor-at=%s:99:23 %s | FileCheck -check-prefix=CHECK-MULTISYNTH %s
// CHECK-MULTISYNTH: 83:13 ObjCSynthesizeDecl=prop1:76:23 (Definition) Extent=[83:1 - 83:18] Spelling=prop1 ([83:13 - 83:18])
// CHECK-MULTISYNTH: 83:20 ObjCSynthesizeDecl=prop2:77:23 (Definition) Extent=[83:1 - 83:25] Spelling=prop2 ([83:20 - 83:25])
// CHECK-MULTISYNTH: 84:10 ObjCDynamicDecl=prop3:78:23 (Definition) Extent=[84:1 - 84:15] Spelling=prop3 ([84:10 - 84:15])
// CHECK-MULTISYNTH: 84:17 ObjCDynamicDecl=prop4:79:23 (Definition) Extent=[84:1 - 84:22] Spelling=prop4 ([84:17 - 84:22])
// CHECK-MULTISYNTH: 99:13 ObjCSynthesizeDecl=prop1:95:23 (Definition) Extent=[99:1 - 99:27] Spelling=prop1 ([99:13 - 99:18])
// CHECK-MULTISYNTH: 99:21 MemberRef=_prop1:99:21 Extent=[99:21 - 99:27] Spelling=_prop1 ([99:21 - 99:27])

// RUN: c-index-test -cursor-at=%s:86:7 -cursor-at=%s:89:7 %s | FileCheck -check-prefix=CHECK-SELECTORLOC %s
// CHECK-SELECTORLOC: 86:6 ObjCInstanceMethodDecl=meth1:86:6 (Definition) Extent=[86:1 - 88:2] Spelling=meth1 ([86:6 - 86:11]) Selector index=0
// CHECK-SELECTORLOC: 89:6 ObjCInstanceMethodDecl=meth2:89:6 (Definition) Extent=[89:1 - 91:2] Spelling=meth2 ([89:6 - 89:11]) Selector index=0

// RUN: c-index-test -cursor-at=%s:103:10 -cursor-at=%s:104:10 \
// RUN:              -cursor-at=%s:107:10 -cursor-at=%s:108:10 \
// RUN:              -cursor-at=%s:111:10 -cursor-at=%s:112:10 %s | FileCheck -check-prefix=CHECK-OBJCOPTIONAL %s
// CHECK-OBJCOPTIONAL: 103:8 ObjCInstanceMethodDecl=protMeth1:103:8 Extent=[103:1 - 103:18]
// CHECK-OBJCOPTIONAL: 104:23 ObjCPropertyDecl=propProp1:104:23 [retain,] Extent=[104:1 - 104:32]
// CHECK-OBJCOPTIONAL: 107:8 ObjCInstanceMethodDecl=protMeth2:107:8 (@optional) Extent=[107:1 - 107:18]
// CHECK-OBJCOPTIONAL: 108:23 ObjCPropertyDecl=propProp2:108:23 (@optional) [retain,] Extent=[108:1 - 108:32]
// CHECK-OBJCOPTIONAL: 111:8 ObjCInstanceMethodDecl=protMeth3:111:8 Extent=[111:1 - 111:18]
// CHECK-OBJCOPTIONAL: 112:23 ObjCPropertyDecl=propProp3:112:23 [retain,] Extent=[112:1 - 112:32]

// RUN: c-index-test -cursor-at=%s:123:12 %s | FileCheck -check-prefix=CHECK-RECEIVER-WITH-NULLABILITY %s
// RUN: c-index-test -cursor-at=%s:124:12 %s | FileCheck -check-prefix=CHECK-RECEIVER-WITH-NULLABILITY %s
// RUN: c-index-test -cursor-at=%s:127:8 %s | FileCheck -check-prefix=CHECK-RECEIVER-WITH-NULLABILITY %s
// RUN: c-index-test -cursor-at=%s:128:8 %s | FileCheck -check-prefix=CHECK-RECEIVER-WITH-NULLABILITY %s
// CHECK-RECEIVER-WITH-NULLABILITY: Receiver-type=ObjCId

// RUN: c-index-test -cursor-at=%s:139:1 -cursor-at=%s:140:6 -cursor-at=%s:141:1 -cursor-at=%s:147:1 -cursor-at=%s:151:1 -cursor-at=%s:155:1 %s | FileCheck -check-prefix=CHECK-TRANSPARENT %s
// CHECK-TRANSPARENT: 139:1 TypeRef=TestTransparent:133:17 (Transparent: enum TestTransparent) Extent=[139:1 - 139:16] Spelling=TestTransparent ([139:1 - 139:16])
// CHECK-TRANSPARENT: 140:6 TypeRef=enum TestTransparent:133:17 Extent=[140:6 - 140:21] Spelling=enum TestTransparent ([140:6 - 140:21])
// CHECK-TRANSPARENT: 141:1 TypeRef=NotTransparent:137:30 Extent=[141:1 - 141:15] Spelling=NotTransparent ([141:1 - 141:15])
// CHECK-TRANSPARENT: 147:1 TypeRef=TokenPaste_t:144:9 Extent=[147:1 - 147:13] Spelling=TokenPaste_t ([147:1 - 147:13])
// CHECK-TRANSPARENT: 151:1 TypeRef=SomeT:150:17 (Transparent: struct SomeT) Extent=[151:1 - 151:6] Spelling=SomeT ([151:1 - 151:6])
// CHECK-TRANSPARENT: 155:1 TypeRef=SomeT2:154:18 Extent=[155:1 - 155:7] Spelling=SomeT2 ([155:1 - 155:7])

// RUN: c-index-test -cursor-at=%s:160:12 -cursor-at=%s:161:8 %s | FileCheck -check-prefix=CHECK-EXTERNAL %s
// CHECK-EXTERNAL: 160:12 ObjCInterfaceDecl=ExtCls:160:12 (external lang: Swift, defined: some_module, gen: 1)
// CHECK-EXTERNAL: 161:8 ObjCInstanceMethodDecl=method:161:8 (external lang: Swift, defined: some_module, gen: 1)
C