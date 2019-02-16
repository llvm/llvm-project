// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

#define EXT_DECL(mod_name) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name)))
#define GEN_DECL(mod_name) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name, generated_declaration)))
#define PUSH_GEN_DECL(mod_name) push(GEN_DECL(mod_name), apply_to=any(enum, objc_interface, objc_category, objc_protocol))

// Forward declarations should not affect module namespacing below
@class I1;
@class I2;

// This should not be indexed.
GEN_DECL("some_module")
@interface I1
// CHECK-NOT: [[@LINE-1]]:12 |
-(void)method;
// CHECK-NOT: [[@LINE-1]]:8 |
@end

EXT_DECL("some_module")
@interface I2
// CHECK: [[@LINE-1]]:12 | class/Swift | I2 | c:@M@some_module@objc(cs)I2 | {{.*}} | Decl | rel: 0
-(void)method;
// CHECK: [[@LINE-1]]:8 | instance-method/Swift | method | c:@M@some_module@objc(cs)I2(im)method | -[I2 method] | Decl,Dyn,RelChild | rel: 1
@end

void test1(I1 *o) {
// CHECK: [[@LINE-1]]:12 | class/Swift | I1 | c:@M@some_module@objc(cs)I1 |
  [o method];
  // CHECK: [[@LINE-1]]:6 | instance-method/Swift | method | c:@M@some_module@objc(cs)I1(im)method |
}

EXT_DECL("some_module")
@protocol ExtProt
// CHECK: [[@LINE-1]]:11 | protocol/Swift | ExtProt | c:@M@some_module@objc(pl)ExtProt |
@end

@interface I1(cat)
// CHECK: [[@LINE-1]]:15 | extension/ObjC | cat | c:@M@some_module@objc(cy)I1@cat |
-(void)cat_method;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | cat_method | c:@M@some_module@objc(cs)I1(im)cat_method
@end

EXT_DECL("cat_module")
@interface I1(cat2)
// CHECK: [[@LINE-1]]:15 | extension/Swift | cat2 | c:@CM@cat_module@some_module@objc(cy)I1@cat2 |
-(void)cat_method2;
// CHECK: [[@LINE-1]]:8 | instance-method/Swift | cat_method2 | c:@CM@cat_module@some_module@objc(cs)I1(im)cat_method2
@end

#define NS_ENUM(_name, _type) enum _name:_type _name; enum _name : _type

#pragma clang attribute PUSH_GEN_DECL("modname")

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
  // CHECK: [[@LINE-1]]:12 | class/Swift | I3 | c:@M@modname@objc(cs)I3 |
  // CHECK: [[@LINE-2]]:23 | protocol/Swift | ExtProt2 | c:@M@modname@objc(pl)ExtProt2 |
  // CHECK: [[@LINE-3]]:40 | enum/Swift | SomeEnum | c:@M@modname@E@SomeEnum |
  [i3 meth];
  // CHECK: [[@LINE-1]]:7 | instance-method/Swift | meth | c:@M@modname@objc(cs)I3(im)meth |
  [i3 meth2];
  // CHECK: [[@LINE-1]]:7 | instance-method/Swift | meth2 | c:@CM@modname@objc(cs)I3(im)meth2 |
  [prot2 meth];
  // CHECK: [[@LINE-1]]:10 | instance-method/Swift | meth | c:@M@modname@objc(pl)ExtProt2(im)meth |
  some = SomeEnumFirst;
  // CHECK: [[@LINE-1]]:10 | enumerator/Swift | SomeEnumFirst | c:@M@modname@E@SomeEnum@SomeEnumFirst |
}

#pragma clang attribute PUSH_GEN_DECL("other_mod_for_cat")
@interface I3(cat_other_mod)
-(void)meth_other_mod;
@end
#pragma clang attribute pop

void test3(I3 *i3) {
  [i3 meth_other_mod];
  // CHECK: [[@LINE-1]]:7 | instance-method/Swift | meth_other_mod | c:@CM@other_mod_for_cat@modname@objc(cs)I3(im)meth_other_mod |
}
