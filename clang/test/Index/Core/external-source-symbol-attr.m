// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

#define EXT_DECL(mod_name) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name)))
#define GEN_DECL(mod_name) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name, generated_declaration)))

// This should not be indexed.
GEN_DECL("some_module")
@interface I1
// CHECK-NOT: [[@LINE-1]]:12 |
-(void)method;
// CHECK-NOT: [[@LINE-1]]:8 |
@end

EXT_DECL("some_module")
@interface I2
// CHECK: [[@LINE-1]]:12 | class/ObjC | I2 | c:@M@some_module@objc(cs)I2 | {{.*}} | Decl | rel: 0
-(void)method;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | method | c:@M@some_module@objc(cs)I2(im)method | -[I2 method] | Decl,Dyn,RelChild | rel: 1
@end

void test1(I1 *o) {
// CHECK: [[@LINE-1]]:12 | class/ObjC | I1 | c:@M@some_module@objc(cs)I1 |
  [o method];
  // CHECK: [[@LINE-1]]:6 | instance-method/ObjC | method | c:@M@some_module@objc(cs)I1(im)method |
}

EXT_DECL("some_module")
@protocol ExtProt
// CHECK: [[@LINE-1]]:11 | protocol/ObjC | ExtProt | c:@M@some_module@objc(pl)ExtProt |
@end

@interface I1(cat)
// CHECK: [[@LINE-1]]:15 | extension/ObjC | cat | c:@M@some_module@objc(cy)I1@cat |
-(void)cat_method;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | cat_method | c:@M@some_module@objc(cs)I1(im)cat_method
@end

EXT_DECL("cat_module")
@interface I1(cat2)
// CHECK: [[@LINE-1]]:15 | extension/ObjC | cat2 | c:@M@cat_module-some_module@objc(cy)I1@cat2 |
-(void)cat_method2;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | cat_method2 | c:@M@some_module@objc(cs)I1(im)cat_method2
@end

#define NS_ENUM(_name, _type) enum _name:_type _name; enum _name : _type

#pragma clang attribute push(GEN_DECL("modname"), apply_to=any(enum, objc_interface, objc_category, objc_protocol))

@interface I3
// CHECK-NOT: [[@LINE-1]]:12 |
-(void)meth;
// CHECK-NOT: [[@LINE-1]]:8 |
@end

@interface I3(cat)
// CHECK-NOT: [[@LINE-1]]:12 |
// CHECK-NOT: [[@LINE-2]]:15 |
-(void)meth2;
// CHECK-NOT: [[@LINE-1]]:8 |
@end

@protocol ExtProt2
// CHECK-NOT: [[@LINE-1]]:11 |
-(void)meth;
// CHECK-NOT: [[@LINE-1]]:8 |
@end

typedef NS_ENUM(SomeEnum, int) {
// CHECK-NOT: [[@LINE-1]]:17 |
  SomeEnumFirst = 0,
  // CHECK-NOT: [[@LINE-1]]:3 |
};

#pragma clang attribute pop

void test2(I3 *i3, id<ExtProt2> prot2, SomeEnum some) {
  // CHECK: [[@LINE-1]]:12 | class/ObjC | I3 | c:@M@modname@objc(cs)I3 |
  // CHECK: [[@LINE-2]]:23 | protocol/ObjC | ExtProt2 | c:@M@modname@objc(pl)ExtProt2 |
  // CHECK: [[@LINE-3]]:40 | enum/C | SomeEnum | c:@M@modname@E@SomeEnum |
  [i3 meth];
  // CHECK: [[@LINE-1]]:7 | instance-method/ObjC | meth | c:@M@modname@objc(cs)I3(im)meth |
  [i3 meth2];
  // CHECK: [[@LINE-1]]:7 | instance-method/ObjC | meth2 | c:@M@modname@objc(cs)I3(im)meth2 |
  [prot2 meth];
  // CHECK: [[@LINE-1]]:10 | instance-method/ObjC | meth | c:@M@modname@objc(pl)ExtProt2(im)meth |
  some = SomeEnumFirst;
  // CHECK: [[@LINE-1]]:10 | enumerator/C | SomeEnumFirst | c:@M@modname@E@SomeEnum@SomeEnumFirst |
}
