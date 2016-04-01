// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

@interface Base
// CHECK: [[@LINE-1]]:12 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Decl | rel: 0
-(void)meth;
// CHECK: [[@LINE-1]]:1 | instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Decl,Dyn,RelChild | rel: 1
// CHECK-NEXT: RelChild | Base | c:objc(cs)Base
@end

void foo();
// CHECK: [[@LINE+1]]:6 | function/C | goo | c:@F@goo | _goo | Def | rel: 0
void goo(Base *b) {
  // CHECK: [[@LINE+2]]:3 | function/C | foo | c:@F@foo | _foo | Ref,Call,RelCall | rel: 1
  // CHECK-NEXT: RelCall | goo | c:@F@goo
  foo();
  // CHECK: [[@LINE+3]]:6 | instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Ref,Call,Dyn,RelRec,RelCall | rel: 2
  // CHECK-NEXT: RelCall | goo | c:@F@goo
  // CHECK-NEXT: RelRec | Base | c:objc(cs)Base
  [b meth];
}

// CHECK: [[@LINE+1]]:11 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Decl | rel: 0
@protocol Prot1
@end

// CHECK: [[@LINE+3]]:11 | protocol/ObjC | Prot2 | c:objc(pl)Prot2 | <no-cgname> | Decl | rel: 0
// CHECK: [[@LINE+2]]:17 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Ref,RelBase | rel: 1
// CHECK-NEXT: RelBase | Prot2 | c:objc(pl)Prot2
@protocol Prot2<Prot1>
@end

// CHECK: [[@LINE+7]]:12 | class/ObjC | Sub | c:objc(cs)Sub | _OBJC_CLASS_$_Sub | Decl | rel: 0
// CHECK: [[@LINE+6]]:18 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelBase | rel: 1
// CHECK-NEXT: RelBase | Sub | c:objc(cs)Sub
// CHECK: [[@LINE+4]]:23 | protocol/ObjC | Prot2 | c:objc(pl)Prot2 | <no-cgname> | Ref,RelBase | rel: 1
// CHECK-NEXT: RelBase | Sub | c:objc(cs)Sub
// CHECK: [[@LINE+2]]:30 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Ref,RelBase | rel: 1
// CHECK-NEXT: RelBase | Sub | c:objc(cs)Sub
@interface Sub : Base<Prot2, Prot1>
@end

@interface NSArray<ObjectType> : Base
// CHECK-NOT: ObjectType
-(ObjectType)getit;
@end

// CHECK: [[@LINE+1]]:6 | function/C | over_func | c:@F@over_func#I# | __Z9over_funci | Decl | rel: 0
void over_func(int x) __attribute__((overloadable));
// CHECK: [[@LINE+1]]:6 | function/C | over_func | c:@F@over_func#f# | __Z9over_funcf | Decl | rel: 0
void over_func(float x) __attribute__((overloadable));

// CHECK: [[@LINE+1]]:6 | enum/C | MyEnum | c:@E@MyEnum | <no-cgname> | Def | rel: 0
enum MyEnum {
  // CHECK: [[@LINE+2]]:3 | enumerator/C | EnumeratorInNamed | c:@E@MyEnum@EnumeratorInNamed | <no-cgname> | Def,RelChild | rel: 1
  // CHECK-NEXT: RelChild | MyEnum | c:@E@MyEnum
  EnumeratorInNamed
};

// CHECK: [[@LINE+1]]:1 | enum/C | <no-name> | c:@Ea@One | <no-cgname> | Def | rel: 0
enum {
  // CHECK: [[@LINE+2]]:3 | enumerator/C | One | c:@Ea@One@One | <no-cgname> | Def,RelChild | rel: 1
  // CHECK-NEXT: RelChild | <no-name> | c:@Ea@One
  One,
  // CHECK: [[@LINE+2]]:3 | enumerator/C | Two | c:@Ea@One@Two | <no-cgname> | Def,RelChild | rel: 1
  // CHECK-NEXT: RelChild | <no-name> | c:@Ea@One
  Two,
};

// CHECK: [[@LINE+1]]:13 | type-alias/C | jmp_buf | c:index-source.m@T@jmp_buf | <no-cgname> | Def | rel: 0
typedef int jmp_buf[(18)];
// CHECK: [[@LINE+2]]:12 | function/C | setjmp | c:@F@setjmp | _setjmp | Decl | rel: 0
// CHECK: [[@LINE+1]]:19 | type-alias/C | jmp_buf | c:index-source.m@T@jmp_buf | <no-cgname> | Ref | rel: 0
extern int setjmp(jmp_buf);

@class I1;
@interface I1
// CHECK: [[@LINE+1]]:1 | instance-method/ObjC | meth | c:objc(cs)I1(im)meth | -[I1 meth] | Decl,Dyn,RelChild | rel: 1
-(void)meth;
@end

@interface I2
@property (readwrite) id prop;
@end

// CHECK: [[@LINE+2]]:17 | field/ObjC | _prop | c:objc(cs)I2@_prop | <no-cgname> | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
@implementation I2
// CHECK: [[@LINE+5]]:13 | instance-property/ObjC | prop | c:objc(cs)I2(py)prop | <no-cgname> | Ref | rel: 0
// CHECK: [[@LINE+4]]:13 | instance-method/ObjC | prop | c:objc(cs)I2(im)prop | -[I2 prop] | Def,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
// CHECK: [[@LINE+2]]:13 | instance-method/ObjC | setProp: | c:objc(cs)I2(im)setProp: | -[I2 setProp:] | Def,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
@synthesize prop = _prop;
@end

@interface I3
@property (readwrite) id prop;
-(id)prop;
-(void)setProp:(id)p;
@end

@implementation I3
// CHECK: [[@LINE+3]]:13 | instance-property/ObjC | prop | c:objc(cs)I3(py)prop | <no-cgname> | Ref | rel: 0
// CHECK: [[@LINE+2]]:13 | instance-method/ObjC | prop | c:objc(cs)I3(im)prop | -[I3 prop] | Def,RelChild | rel: 1
// CHECK: [[@LINE+1]]:13 | instance-method/ObjC | setProp: | c:objc(cs)I3(im)setProp: | -[I3 setProp:] | Def,RelChild | rel: 1
@synthesize prop = _prop;
@end
