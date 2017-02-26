// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

@interface Base
// CHECK: [[@LINE-1]]:12 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Decl | rel: 0
-(void)meth;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Decl,Dyn,RelChild | rel: 1
// CHECK-NEXT: RelChild | Base | c:objc(cs)Base
+(Base*)class_meth;
// CHECK: [[@LINE-1]]:9 | class-method/ObjC | class_meth | c:objc(cs)Base(cm)class_meth | +[Base class_meth] | Decl,Dyn,RelChild | rel: 1
// CHECK: [[@LINE-2]]:3 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | class_meth | c:objc(cs)Base(cm)class_meth

@end

void foo();
// CHECK: [[@LINE+3]]:6 | function/C | goo | c:@F@goo | _goo | Def | rel: 0
// CHECK: [[@LINE+2]]:10 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | goo | c:@F@goo
void goo(Base *b) {
  // CHECK: [[@LINE+2]]:3 | function/C | foo | c:@F@foo | _foo | Ref,Call,RelCall,RelCont | rel: 1
  // CHECK-NEXT: RelCall,RelCont | goo | c:@F@goo
  foo();
  // CHECK: [[@LINE+3]]:6 | instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 2
  // CHECK-NEXT: RelCall,RelCont | goo | c:@F@goo
  // CHECK-NEXT: RelRec | Base | c:objc(cs)Base
  [b meth];

  // CHECK: [[@LINE+2]]:4 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
  // CHECK-NEXT: RelCont | goo | c:@F@goo
  [Base class_meth];

  // CHECK: [[@LINE+4]]:3 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
  // CHECK-NEXT: RelCont | goo | c:@F@goo
  // CHECK: [[@LINE+2]]:14 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
  // CHECK-NEXT: RelCont | goo | c:@F@goo
  Base *f = (Base *) 2;
}

// CHECK: [[@LINE+1]]:11 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Decl | rel: 0
@protocol Prot1
@end

// CHECK: [[@LINE+3]]:11 | protocol/ObjC | Prot2 | c:objc(pl)Prot2 | <no-cgname> | Decl | rel: 0
// CHECK: [[@LINE+2]]:17 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Prot2 | c:objc(pl)Prot2
@protocol Prot2<Prot1>
@end

// CHECK: [[@LINE+7]]:12 | class/ObjC | Sub | c:objc(cs)Sub | _OBJC_CLASS_$_Sub | Decl | rel: 0
// CHECK: [[@LINE+6]]:18 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Sub | c:objc(cs)Sub
// CHECK: [[@LINE+4]]:23 | protocol/ObjC | Prot2 | c:objc(pl)Prot2 | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Sub | c:objc(cs)Sub
// CHECK: [[@LINE+2]]:30 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | Sub | c:objc(cs)Sub
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
// CHECK: [[@LINE+3]]:12 | function/C | setjmp | c:@F@setjmp | _setjmp | Decl | rel: 0
// CHECK: [[@LINE+2]]:19 | type-alias/C | jmp_buf | c:index-source.m@T@jmp_buf | <no-cgname> | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | setjmp | c:@F@setjmp
extern int setjmp(jmp_buf);

@class I1;
@interface I1
// CHECK: [[@LINE+1]]:8 | instance-method/ObjC | meth | c:objc(cs)I1(im)meth | -[I1 meth] | Decl,Dyn,RelChild | rel: 1
-(void)meth;
@end

@interface I2
@property (readwrite) id prop;

// CHECK: [[@LINE+4]]:63 | instance-property(IB,IBColl)/ObjC | buttons | c:objc(cs)I2(py)buttons | <no-cgname> | Decl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
// CHECK: [[@LINE+2]]:50 | class/ObjC | I1 | c:objc(cs)I1 | _OBJC_CLASS_$_I1 | Ref,RelCont,RelIBType | rel: 1
// CHECK-NEXT: RelCont,RelIBType | buttons | c:objc(cs)I2(py)buttons
@property (nonatomic, strong) IBOutletCollection(I1) NSArray *buttons;
@end

@implementation I2
// CHECK: [[@LINE+9]]:13 | instance-property/ObjC | prop | c:objc(cs)I2(py)prop | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
// CHECK-NEXT: RelAcc | _prop | c:objc(cs)I2@_prop
// CHECK: [[@LINE+6]]:13 | instance-method/acc-get/ObjC | prop | c:objc(cs)I2(im)prop | -[I2 prop] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
// CHECK: [[@LINE+4]]:13 | instance-method/acc-set/ObjC | setProp: | c:objc(cs)I2(im)setProp: | -[I2 setProp:] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
// CHECK: [[@LINE+2]]:20 | field/ObjC | _prop | c:objc(cs)I2@_prop | <no-cgname> | Def,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
@synthesize prop = _prop;

// CHECK: [[@LINE+5]]:12 | instance-method(IB)/ObjC | doAction:foo: | c:objc(cs)I2(im)doAction:foo: | -[I2 doAction:foo:] | Def,Dyn,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | c:objc(cs)I2
// CHECK: [[@LINE+3]]:22 | class/ObjC | I1 | c:objc(cs)I1 | _OBJC_CLASS_$_I1 | Ref,RelCont,RelIBType | rel: 1
// CHECK-NEXT: RelCont,RelIBType | doAction:foo: | c:objc(cs)I2(im)doAction:foo:
// CHECK: [[@LINE+1]]:39 | class/ObjC | I1 | c:objc(cs)I1 | _OBJC_CLASS_$_I1 | Ref,RelCont | rel: 1
-(IBAction)doAction:(I1 *)sender foo:(I1 *)bar {}
@end

@interface I3
@property (readwrite) id prop;
// CHECK: [[@LINE+3]]:6 | instance-method/acc-get/ObjC | prop | c:objc(cs)I3(im)prop | -[I3 prop] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I3 | c:objc(cs)I3
// CHECK-NEXT: RelAcc | prop | c:objc(cs)I3(py)prop
-(id)prop;
// CHECK: [[@LINE+3]]:8 | instance-method/acc-set/ObjC | setProp: | c:objc(cs)I3(im)setProp: | -[I3 setProp:] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I3 | c:objc(cs)I3
// CHECK-NEXT: RelAcc | prop | c:objc(cs)I3(py)prop
-(void)setProp:(id)p;
@end

// CHECK: [[@LINE+1]]:17 | class/ObjC | I3 | c:objc(cs)I3 | <no-cgname> | Def | rel: 0
@implementation I3
// CHECK: [[@LINE+5]]:13 | instance-property/ObjC | prop | c:objc(cs)I3(py)prop | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I3 | c:objc(cs)I3
// CHECK-NEXT: RelAcc | _prop | c:objc(cs)I3@_prop
// CHECK: [[@LINE+2]]:13 | instance-method/acc-get/ObjC | prop | c:objc(cs)I3(im)prop | -[I3 prop] | Def,Impl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:13 | instance-method/acc-set/ObjC | setProp: | c:objc(cs)I3(im)setProp: | -[I3 setProp:] | Def,Impl,RelChild | rel: 1
@synthesize prop = _prop;
@end

// CHECK: [[@LINE+5]]:12 | class/ObjC | I3 | c:objc(cs)I3 | _OBJC_CLASS_$_I3 | Ref,RelExt,RelCont | rel: 1
// CHECK-NEXT: RelExt,RelCont | bar | c:objc(cy)I3@bar
// CHECK: [[@LINE+3]]:15 | extension/ObjC | bar | c:objc(cy)I3@bar | <no-cgname> | Decl | rel: 0
// CHECK: [[@LINE+2]]:21 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | bar | c:objc(cy)I3@bar
@interface I3(bar) <Prot1>
@end

// CHECK: [[@LINE+2]]:17 | class/ObjC | I3 | c:objc(cs)I3 | _OBJC_CLASS_$_I3 | Ref,RelCont | rel: 1
// CHECK: [[@LINE+1]]:20 | extension/ObjC | I3 | c:objc(cy)I3@bar | <no-cgname> | Def | rel: 0
@implementation I3(bar)
@end

// CHECK-NOT: [[@LINE+1]]:12 | extension/ObjC |
@interface NonExistent()
@end

@interface MyGenCls<ObjectType> : Base
@end

@protocol MyEnumerating
@end

// CHECK: [[@LINE+4]]:41 | type-alias/C | MyEnumerator | c:index-source.m@T@MyEnumerator | <no-cgname> | Def | rel: 0
// CHECK: [[@LINE+3]]:26 | protocol/ObjC | MyEnumerating | c:objc(pl)MyEnumerating | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE+2]]:9 | class/ObjC | MyGenCls | c:objc(cs)MyGenCls | _OBJC_CLASS_$_MyGenCls | Ref,RelCont | rel: 1
// CHECK: [[@LINE+1]]:18 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
typedef MyGenCls<Base *><MyEnumerating> MyEnumerator;

// CHECK: [[@LINE+5]]:12 | class/ObjC | PermanentEnumerator | c:objc(cs)PermanentEnumerator | _OBJC_CLASS_$_PermanentEnumerator | Decl | rel: 0
// CHECK: [[@LINE+4]]:34 | class/ObjC | MyGenCls | c:objc(cs)MyGenCls | _OBJC_CLASS_$_MyGenCls | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | PermanentEnumerator | c:objc(cs)PermanentEnumerator
// CHECK: [[@LINE+2]]:34 | protocol/ObjC | MyEnumerating | c:objc(pl)MyEnumerating | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | PermanentEnumerator | c:objc(cs)PermanentEnumerator
@interface PermanentEnumerator : MyEnumerator
@end

@interface I4
@property id foo;
@end

@implementation I4 {
  id _blahfoo; // explicit def
  // CHECK: [[@LINE-1]]:6 | field/ObjC | _blahfoo | c:objc(cs)I4@_blahfoo | <no-cgname> | Def,RelChild | rel: 1
}
@synthesize foo = _blahfoo; // ref of field _blahfoo
// CHECK: [[@LINE-1]]:13 | instance-property/ObjC | foo | c:objc(cs)I4(py)foo | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I4 | c:objc(cs)I4
// CHECK-NEXT: RelAcc | _blahfoo | c:objc(cs)I4@_blahfoo
// CHECK: [[@LINE-4]]:13 | instance-method/acc-get/ObjC | foo | c:objc(cs)I4(im)foo | -[I4 foo] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I4 | c:objc(cs)I4
// CHECK: [[@LINE-6]]:13 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I4(im)setFoo: | -[I4 setFoo:] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I4 | c:objc(cs)I4
// CHECK: [[@LINE-8]]:19 | field/ObjC | _blahfoo | c:objc(cs)I4@_blahfoo | <no-cgname> | Ref | rel: 0

-(void)method {
  _blahfoo = 0;
  // CHECK: [[@LINE-1]]:3 | field/ObjC | _blahfoo | c:objc(cs)I4@_blahfoo | <no-cgname> | Ref,Writ,RelCont | rel: 1
}
@end

@interface I5
@property id foo;
@end

@implementation I5
@synthesize foo = _blahfoo; // explicit def of field _blahfoo
// CHECK: [[@LINE-1]]:13 | instance-property/ObjC | foo | c:objc(cs)I5(py)foo | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I5 | c:objc(cs)I5
// CHECK-NEXT: RelAcc | _blahfoo | c:objc(cs)I5@_blahfoo
// CHECK: [[@LINE-4]]:13 | instance-method/acc-get/ObjC | foo | c:objc(cs)I5(im)foo | -[I5 foo] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I5 | c:objc(cs)I5
// CHECK: [[@LINE-6]]:13 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I5(im)setFoo: | -[I5 setFoo:] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I5 | c:objc(cs)I5
// CHECK: [[@LINE-8]]:19 | field/ObjC | _blahfoo | c:objc(cs)I5@_blahfoo | <no-cgname> | Def,RelChild | rel: 1

-(void)method {
  _blahfoo = 0;
  // CHECK: [[@LINE-1]]:3 | field/ObjC | _blahfoo | c:objc(cs)I5@_blahfoo | <no-cgname> | Ref,Writ,RelCont | rel: 1
}
@end

@interface I6
@property id foo;
@end

@implementation I6
@synthesize foo; // implicit def of field foo
// CHECK: [[@LINE-1]]:13 | instance-property/ObjC | foo | c:objc(cs)I6(py)foo | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I6 | c:objc(cs)I6
// CHECK-NEXT: RelAcc | foo | c:objc(cs)I6@foo
// CHECK: [[@LINE-4]]:13 | instance-method/acc-get/ObjC | foo | c:objc(cs)I6(im)foo | -[I6 foo] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I6 | c:objc(cs)I6
// CHECK: [[@LINE-6]]:13 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I6(im)setFoo: | -[I6 setFoo:] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I6 | c:objc(cs)I6
// CHECK: [[@LINE-8]]:13 | field/ObjC | foo | c:objc(cs)I6@foo | <no-cgname> | Def,Impl,RelChild | rel: 1

-(void)method {
  foo = 0;
  // CHECK: [[@LINE-1]]:3 | field/ObjC | foo | c:objc(cs)I6@foo | <no-cgname> | Ref,Writ,RelCont | rel: 1
}
@end

@interface I7
@property id foo;
@end

@implementation I7 // implicit def of field _foo
// CHECK: [[@LINE-1]]:17 | instance-property/ObjC | foo | c:objc(cs)I7(py)foo | <no-cgname> | Def,Impl,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I7 | c:objc(cs)I7
// CHECK-NEXT: RelAcc | _foo | c:objc(cs)I7@_foo
// CHECK: [[@LINE-4]]:17 | instance-method/acc-get/ObjC | foo | c:objc(cs)I7(im)foo | -[I7 foo] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I7 | c:objc(cs)I7
// CHECK: [[@LINE-6]]:17 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I7(im)setFoo: | -[I7 setFoo:] | Def,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I7 | c:objc(cs)I7
// CHECK: [[@LINE-8]]:17 | field/ObjC | _foo | c:objc(cs)I7@_foo | <no-cgname> | Def,Impl,RelChild | rel: 1

-(void)method {
  _foo = 0;
// CHECK: [[@LINE-1]]:3 | field/ObjC | _foo | c:objc(cs)I7@_foo | <no-cgname> | Ref,Writ,RelCont | rel: 1
}
@end
