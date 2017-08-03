// RUN: rm -rf %t.idx
// RUN: %clang_cc1 %s -index-store-path %t.idx -D USE_EXTERNAL
// RUN: c-index-test core -print-record %t.idx | FileCheck %s
// RUN: %clang_cc1 %s -index-store-path %t.idx
// RUN: find %t.idx/*/records -name "external-source-symbol-hash*" | count 2

#ifdef USE_EXTERNAL
#  define EXT_DECL(mod_name) __attribute__((external_source_symbol(language="Swift", defined_in=mod_name)))
#else
#  define EXT_DECL(mod_name)
#endif

#define NS_ENUM(_name, _type) enum _name:_type _name; enum _name : _type

// Forward declarations should pick up the attribute from later decls
@protocol P1;
// CHECK: [[@LINE-1]]:11 | protocol/Swift | c:@M@some_module@objc(pl)P1 | Ref | rel: 0
@class I2;
// CHECK: [[@LINE-1]]:8 | class/Swift | c:@M@other_module@objc(cs)I2 | Ref | rel: 0
enum E3: int;
// CHECK: [[@LINE-1]]:6 | enum/Swift | c:@M@third_module@E@E3 | Ref | rel: 0

void test(id<P1> first, I2 *second, enum E3 third) {}
// CHECK: [[@LINE-1]]:14 | protocol/Swift | c:@M@some_module@objc(pl)P1 | Ref,RelCont | rel: 1
// CHECK: [[@LINE-2]]:25 | class/Swift | c:@M@other_module@objc(cs)I2 | Ref,RelCont | rel: 1
// CHECK: [[@LINE-3]]:42 | enum/Swift | c:@M@third_module@E@E3 | Ref,RelCont | rel: 1

EXT_DECL("some_module")
@protocol P1
// CHECK: [[@LINE-1]]:11 | protocol/Swift | c:@M@some_module@objc(pl)P1 | Decl | rel: 0
-(void)method;
// CHECK: [[@LINE-1]]:8 | instance-method/Swift | c:@M@some_module@objc(pl)P1(im)method | Decl,Dyn,RelChild | rel: 1
@end

EXT_DECL("other_module")
@interface I2
// CHECK: [[@LINE-1]]:12 | class/Swift | c:@M@other_module@objc(cs)I2 | Decl | rel: 0
-(void)method;
// CHECK: [[@LINE-1]]:8 | instance-method/Swift | c:@M@other_module@objc(cs)I2(im)method | Decl,Dyn,RelChild | rel: 1
@end


typedef NS_ENUM(E3, int) {
// CHECK: [[@LINE-1]]:17 | enum/Swift | c:@M@third_module@E@E3 | Def | rel: 0
  firstCase = 1,
  // CHECK: [[@LINE-1]]:3 | enumerator/Swift | c:@M@third_module@E@E3@firstCase | Def,RelChild | rel: 1
} EXT_DECL("third_module");
