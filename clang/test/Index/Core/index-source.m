// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s
// RUN: c-index-test core -print-source-symbols -include-locals -- %s -target x86_64-apple-macosx10.7 | FileCheck -check-prefix=LOCAL %s

@interface Base
// CHECK: [[@LINE-1]]:12 | class/ObjC | Base | [[BASE_USR:.*]] | _OBJC_CLASS_$_Base | Decl | rel: 0
-(void)meth;
// CHECK: [[@LINE-1]]:8 | instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Decl,Dyn,RelChild | rel: 1
// CHECK-NEXT: RelChild | Base | c:objc(cs)Base
+(Base*)class_meth;
// CHECK: [[@LINE-1]]:9 | class-method/ObjC | class_meth | c:objc(cs)Base(cm)class_meth | +[Base class_meth] | Decl,Dyn,RelChild | rel: 1
// CHECK: [[@LINE-2]]:3 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | class_meth | c:objc(cs)Base(cm)class_meth

@end

void foo();
// CHECK: [[@LINE+6]]:6 | function/C | goo | c:@F@goo | _goo | Def | rel: 0
// CHECK: [[@LINE+5]]:10 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | goo | c:@F@goo
// CHECK-NOT: [[@LINE+3]]:16 | param
// LOCAL: [[@LINE+2]]:16 | param(local)/C | b | [[b_USR:c:.*]] | _b | Def,RelChild | rel: 1
// LOCAL-NEXT: RelChild | goo | c:@F@goo
void goo(Base *b) {
  // CHECK-NOT: [[@LINE+6]]:7 | variable
  // LOCAL: [[@LINE+5]]:7 | variable(local)/C | x | [[x_USR:c:.*]] | _x | Def,RelCont | rel: 1
  // LOCAL-NEXT: RelCont | goo | c:@F@goo
  // CHECK-NOT: [[@LINE+3]]:11 | param
  // LOCAL: [[@LINE+2]]:11 | param(local)/C | b | [[b_USR]] | _b | Ref,Read,RelCont | rel: 1
  // LOCAL-NEXT: RelCont | x | [[x_USR]]
  int x = b;
  // CHECK-NOT: [[@LINE+5]]:7 | variable
  // LOCAL: [[@LINE+4]]:7 | variable(local)/C | y | [[y_USR:c:.*]] | _y | Def,RelCont | rel: 1
  // CHECK-NOT: [[@LINE+3]]:11 | variable
  // LOCAL: [[@LINE+2]]:11 | variable(local)/C | x | [[x_USR]] | _x | Ref,Read,RelCont | rel: 1
  // LOCAL-NEXT: RelCont | y | [[y_USR]]
  int y = x;

  // CHECK-NOT: [[@LINE+1]]:10 | struct
  // LOCAL: [[@LINE+1]]:10 | struct(local)/C | Foo | c:{{.*}} | <no-cgname> | Def,RelCont | rel: 1
  struct Foo {
    int i;
  };

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

// CHECK: [[@LINE+1]]:11 | protocol/ObjC | Prot1 | [[PROT1_USR:.*]] | <no-cgname> | Decl | rel: 0
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
// CHECK: [[@LINE-1]]:12 | class/ObjC | I2 | [[I2_USR:.*]] | {{.*}} | Decl | rel: 0

@property (readwrite) id prop;
// CHECK: [[@LINE-1]]:26 | instance-method/acc-get/ObjC | prop | [[I2_prop_getter_USR:.*]] | -[I2 prop] | Decl,Dyn,Impl,RelChild,RelAcc | rel: 2
// CHECK: [[@LINE-2]]:26 | instance-method/acc-set/ObjC | setProp: | [[I2_prop_setter_USR:.*]] | -[I2 setProp:] | Decl,Dyn,Impl,RelChild,RelAcc | rel: 2
// CHECK: [[@LINE-3]]:26 | instance-property/ObjC | prop | [[I2_prop_USR:.*]] | <no-cgname> | Decl,RelChild | rel: 1

@property (readwrite, getter=customGet, setter=customSet:) id unrelated;
// CHECK: [[@LINE-1]]:30 | instance-method/acc-get/ObjC | customGet | {{.*}} | -[I2 customGet] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK: [[@LINE-2]]:48 | instance-method/acc-set/ObjC | customSet: | {{.*}} | -[I2 customSet:] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK: [[@LINE-3]]:63 | instance-property/ObjC | unrelated | {{.*}} | <no-cgname> | Decl,RelChild | rel: 1

-(id)declaredGet;
@property (readwrite, getter=declaredGet) id otherProp;
// CHECK: [[@LINE-1]]:30 | instance-method/acc-get/ObjC | declaredGet | {{.*}} | -[I2 declaredGet] | Ref,RelCont | rel: 1
// CHECK: [[@LINE-3]]:6 | instance-method/acc-get/ObjC | declaredGet | {{.*}} | -[I2 declaredGet] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK: [[@LINE-3]]:46 | instance-method/acc-set/ObjC | setOtherProp: | {{.*}} | -[I2 setOtherProp:] | Decl,Dyn,Impl,RelChild,RelAcc | rel: 2

// CHECK: [[@LINE+4]]:63 | instance-property(IB,IBColl)/ObjC | buttons | [[buttons_USR:.*]] | <no-cgname> | Decl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | [[I2_USR]]
// CHECK: [[@LINE+2]]:50 | class/ObjC | I1 | c:objc(cs)I1 | _OBJC_CLASS_$_I1 | Ref,RelCont,RelIBType | rel: 1
// CHECK-NEXT: RelCont,RelIBType | buttons | [[buttons_USR]]
@property (nonatomic, strong) IBOutletCollection(I1) NSArray *buttons;
@end

@implementation I2
// CHECK: [[@LINE+9]]:13 | instance-property/ObjC | prop | [[I2_prop_USR:.*]] | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I2 | [[I2_USR]]
// CHECK-NEXT: RelAcc | _prop | c:objc(cs)I2@_prop
// CHECK: [[@LINE+6]]:13 | instance-method/acc-get/ObjC | prop | [[I2_prop_getter_USR]] | -[I2 prop] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | [[I2_USR]]
// CHECK: [[@LINE+4]]:13 | instance-method/acc-set/ObjC | setProp: | [[I2_prop_setter_USR]] | -[I2 setProp:] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | [[I2_USR]]
// CHECK: [[@LINE+2]]:20 | field/ObjC | _prop | c:objc(cs)I2@_prop | <no-cgname> | Def,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | [[I2_USR]]
@synthesize prop = _prop;

// CHECK: [[@LINE+11]]:12 | instance-method(IB)/ObjC | doAction:foo: | [[doAction_USR:.*]] | -[I2 doAction:foo:] | Def,Dyn,RelChild | rel: 1
// CHECK-NEXT: RelChild | I2 | [[I2_USR]]
// CHECK: [[@LINE+9]]:22 | class/ObjC | I1 | c:objc(cs)I1 | _OBJC_CLASS_$_I1 | Ref,RelCont,RelIBType | rel: 1
// CHECK-NEXT: RelCont,RelIBType | doAction:foo: | [[doAction_USR]]
// CHECK-NOT: [[@LINE+7]]:27 | param
// LOCAL: [[@LINE+6]]:27 | param(local)/C | sender | c:{{.*}} | _sender | Def,RelChild | rel: 1
// LOCAL-NEXT: RelChild | doAction:foo: | [[doAction_USR:.*]]
// CHECK: [[@LINE+4]]:39 | class/ObjC | I1 | c:objc(cs)I1 | _OBJC_CLASS_$_I1 | Ref,RelCont | rel: 1
// CHECK-NOT: [[@LINE+3]]:44 | param
// LOCAL: [[@LINE+2]]:44 | param(local)/C | bar | c:{{.*}} | _bar | Def,RelChild | rel: 1
// LOCAL-NEXT: RelChild | doAction:foo: | [[doAction_USR]]
-(IBAction)doAction:(I1 *)sender foo:(I1 *)bar {
  [self prop];
  // CHECK: [[@LINE-1]]:9 | instance-method/acc-get/ObjC | prop | [[I2_prop_getter_USR]] | -[I2 prop] | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 2
  // CHECK-NEXT: RelCall,RelCont | doAction:foo: | [[doAction_USR]]
  // CHECK-NEXT: RelRec | I2 | [[I2_USR]]

  [self setProp: bar];
  // CHECK: [[@LINE-1]]:9 | instance-method/acc-set/ObjC | setProp: | [[I2_prop_setter_USR]] | -[I2 setProp:] | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 2
  // CHECK-NEXT: RelCall,RelCont | doAction:foo: | [[doAction_USR]]
  // CHECK-NEXT: RelRec | I2 | [[I2_USR]]

  self.prop;
  // CHECK: [[@LINE-1]]:8 | instance-property/ObjC | prop | [[I2_prop_USR]] | <no-cgname> | Ref,RelCont | rel: 1
  // CHECK-NEXT: RelCont | doAction:foo: | [[doAction_USR]]
  // CHECK: [[@LINE-3]]:8 | instance-method/acc-get/ObjC | prop | [[I2_prop_getter_USR]] | -[I2 prop] | Ref,Call,Dyn,Impl,RelRec,RelCall,RelCont | rel: 2
  // CHECK-NEXT: RelCall,RelCont | doAction:foo: | [[doAction_USR]]
  // CHECK-NEXT: RelRec | I2 | [[I2_USR]]

  self.prop = self.prop;
  // CHECK: [[@LINE-1]]:8 | instance-property/ObjC | prop | [[I2_prop_USR]] | <no-cgname> | Ref,Writ,RelCont | rel: 1
  // CHECK-NEXT: RelCont | doAction:foo: | [[doAction_USR]]
  // CHECK:[[@LINE-3]]:8 | instance-method/acc-set/ObjC | setProp: | [[I2_prop_setter_USR]] | -[I2 setProp:] | Ref,Call,Dyn,Impl,RelRec,RelCall,RelCont | rel: 2
  // CHECK-NEXT: RelCall,RelCont | doAction:foo: | [[doAction_USR]]
  // CHECK-NEXT: RelRec | I2 | [[I2_USR]]
}
@end

@interface I3
@property (readwrite) id prop;
// CHECK: [[@LINE+3]]:6 | instance-method/acc-get/ObjC | prop | c:objc(cs)I3(im)prop | -[I3 prop] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I3 | c:objc(cs)I3
// CHECK-NEXT: RelAcc | prop | c:objc(cs)I3(py)prop
-(id)prop;
// CHECK: [[@LINE+4]]:8 | instance-method/acc-set/ObjC | setProp: | c:objc(cs)I3(im)setProp: | -[I3 setProp:] | Decl,Dyn,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I3 | c:objc(cs)I3
// CHECK-NEXT: RelAcc | prop | c:objc(cs)I3(py)prop
// LOCAL-NOT: [[@LINE+1]]:20 | param
-(void)setProp:(id)p;
@end

// CHECK: [[@LINE+1]]:17 | class/ObjC | I3 | c:objc(cs)I3 | <no-cgname> | Def | rel: 0
@implementation I3
// CHECK: [[@LINE+5]]:13 | instance-property/ObjC | prop | c:objc(cs)I3(py)prop | <no-cgname> | Def,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | I3 | c:objc(cs)I3
// CHECK-NEXT: RelAcc | _prop | c:objc(cs)I3@_prop
// CHECK: [[@LINE+2]]:13 | instance-method/acc-get/ObjC | prop | c:objc(cs)I3(im)prop | -[I3 prop] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK: [[@LINE+1]]:13 | instance-method/acc-set/ObjC | setProp: | c:objc(cs)I3(im)setProp: | -[I3 setProp:] | Def,Dyn,Impl,RelChild | rel: 1
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
// CHECK: [[@LINE+1]]:20 | extension/ObjC | bar | c:objc(cy)I3@bar | <no-cgname> | Def | rel: 0
@implementation I3(bar)
@end

// CHECK-NOT: [[@LINE+1]]:12 | extension/ObjC |
@interface NonExistent()
@end

@interface MyGenCls<ObjectType> : Base
@end

@protocol MyEnumerating
@end

// CHECK: [[@LINE+4]]:41 | type-alias/C | MyEnumerator | [[MyEnumerator_USR:.*]] | <no-cgname> | Def | rel: 0
// CHECK: [[@LINE+3]]:26 | protocol/ObjC | MyEnumerating | c:objc(pl)MyEnumerating | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE+2]]:9 | class/ObjC | MyGenCls | c:objc(cs)MyGenCls | _OBJC_CLASS_$_MyGenCls | Ref,RelCont | rel: 1
// CHECK: [[@LINE+1]]:18 | class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Ref,RelCont | rel: 1
typedef MyGenCls<Base *><MyEnumerating> MyEnumerator;

// CHECK: [[@LINE+7]]:12 | class/ObjC | PermanentEnumerator | [[PermanentEnumerator_USR:.*]] | _OBJC_CLASS_$_PermanentEnumerator | Decl | rel: 0
// CHECK: [[@LINE+6]]:34 | type-alias/C | MyEnumerator | [[MyEnumerator_USR]] | <no-cgname> | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | PermanentEnumerator | [[PermanentEnumerator_USR]]
// CHECK: [[@LINE+4]]:34 | class/ObjC | MyGenCls | c:objc(cs)MyGenCls | _OBJC_CLASS_$_MyGenCls | Ref,Impl,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | PermanentEnumerator | [[PermanentEnumerator_USR]]
// CHECK: [[@LINE+2]]:34 | protocol/ObjC | MyEnumerating | c:objc(pl)MyEnumerating | <no-cgname> | Ref,Impl,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | PermanentEnumerator | [[PermanentEnumerator_USR]]
@interface PermanentEnumerator : MyEnumerator
@end

// CHECK: [[@LINE+2]]:48 | protocol/ObjC | Prot1 | c:objc(pl)Prot1 | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK: [[@LINE+1]]:35 | protocol/ObjC | MyEnumerating | c:objc(pl)MyEnumerating | <no-cgname> | Ref,Impl,RelBase,RelCont | rel: 1
@interface PermanentEnumerator2 : MyEnumerator<Prot1>
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
// CHECK: [[@LINE-4]]:13 | instance-method/acc-get/ObjC | foo | c:objc(cs)I4(im)foo | -[I4 foo] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I4 | c:objc(cs)I4
// CHECK: [[@LINE-6]]:13 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I4(im)setFoo: | -[I4 setFoo:] | Def,Dyn,Impl,RelChild | rel: 1
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
// CHECK: [[@LINE-4]]:13 | instance-method/acc-get/ObjC | foo | c:objc(cs)I5(im)foo | -[I5 foo] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I5 | c:objc(cs)I5
// CHECK: [[@LINE-6]]:13 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I5(im)setFoo: | -[I5 setFoo:] | Def,Dyn,Impl,RelChild | rel: 1
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
// CHECK: [[@LINE-4]]:13 | instance-method/acc-get/ObjC | foo | c:objc(cs)I6(im)foo | -[I6 foo] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I6 | c:objc(cs)I6
// CHECK: [[@LINE-6]]:13 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I6(im)setFoo: | -[I6 setFoo:] | Def,Dyn,Impl,RelChild | rel: 1
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
// CHECK: [[@LINE-4]]:17 | instance-method/acc-get/ObjC | foo | c:objc(cs)I7(im)foo | -[I7 foo] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I7 | c:objc(cs)I7
// CHECK: [[@LINE-6]]:17 | instance-method/acc-set/ObjC | setFoo: | c:objc(cs)I7(im)setFoo: | -[I7 setFoo:] | Def,Dyn,Impl,RelChild | rel: 1
// CHECK-NEXT: RelChild | I7 | c:objc(cs)I7
// CHECK: [[@LINE-8]]:17 | field/ObjC | _foo | c:objc(cs)I7@_foo | <no-cgname> | Def,Impl,RelChild | rel: 1

-(void)method {
  _foo = 0;
// CHECK: [[@LINE-1]]:3 | field/ObjC | _foo | c:objc(cs)I7@_foo | <no-cgname> | Ref,Writ,RelCont | rel: 1
}
@end

#define NS_ENUM(_name, _type) enum _name:_type _name; enum _name : _type

typedef NS_ENUM(AnotherEnum, int) {
// CHECK-NOT: [[@LINE-1]]:17 | type-alias/C | AnotherEnum |
// CHECK-NOT: [[@LINE-2]]:17 | {{.*}} | Ref
// CHECK: [[@LINE-3]]:17 | enum/C | AnotherEnum | [[AnotherEnum_USR:.*]] | {{.*}} | Def | rel: 0
  AnotherEnumFirst = 0,
  AnotherEnumSecond = 1,
  AnotherEnumThird = 2,
};

AnotherEnum anotherT;
// CHECK: [[@LINE-1]]:1 | enum/C | AnotherEnum | [[AnotherEnum_USR]] | {{.*}} | Ref,RelCont | rel: 1
enum AnotherEnum anotherE;
// CHECK: [[@LINE-1]]:6 | enum/C | AnotherEnum | [[AnotherEnum_USR]] | {{.*}} | Ref,RelCont | rel: 1

#define TRANSPARENT(_name) struct _name _name; struct _name
#define OPAQUE(_name) struct _name *_name; struct _name

typedef TRANSPARENT(AStruct) {
  int x;
};

AStruct aStructT;
// CHECK: [[@LINE-1]]:1 | struct/C | AStruct | {{.*}} | {{.*}} | Ref,RelCont | rel: 1
struct AStruct aStructS;
// CHECK: [[@LINE-1]]:8 | struct/C | AStruct | {{.*}} | {{.*}} | Ref,RelCont | rel: 1

typedef OPAQUE(Separate) {
  int x;
};

Separate separateT;
// CHECK: [[@LINE-1]]:1 | type-alias/C | Separate | {{.*}} | {{.*}} | Ref,RelCont | rel: 1
struct Separate separateE;
// CHECK: [[@LINE-1]]:8 | struct/C | Separate | {{.*}} | {{.*}} | Ref,RelCont | rel: 1

@interface ClassReceivers

@property(class) int p1;
// CHECK: [[@LINE-1]]:22 | class-method/acc-get/ObjC | p1 | c:objc(cs)ClassReceivers(cm)p1 | +[ClassReceivers p1] | Decl,Dyn,Impl,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | ClassReceivers | c:objc(cs)ClassReceivers
// CHECK-NEXT: RelAcc | p1 | c:objc(cs)ClassReceivers(cpy)p1
// CHECK: [[@LINE-4]]:22 | class-method/acc-set/ObjC | setP1: | c:objc(cs)ClassReceivers(cm)setP1: | +[ClassReceivers setP1:] | Decl,Dyn,Impl,RelChild,RelAcc | rel: 2
// CHECK-NEXT: RelChild | ClassReceivers | c:objc(cs)ClassReceivers
// CHECK-NEXT: RelAcc | p1 | c:objc(cs)ClassReceivers(cpy)p1
// CHECK: [[@LINE-7]]:22 | instance-property/ObjC | p1 | c:objc(cs)ClassReceivers(cpy)p1 | <no-cgname> | Decl,RelChild | rel: 1
// CHECK-NEXT: RelChild | ClassReceivers | c:objc(cs)ClassReceivers

+ (int)implicit;
+ (void)setImplicit:(int)x;

@end

void classReceivers() {
  ClassReceivers.p1 = 0;
// CHECK: [[@LINE-1]]:3 | class/ObjC | ClassReceivers | c:objc(cs)ClassReceivers | _OBJC_CLASS_$_ClassReceivers | Ref,RelCont | rel: 1
// CHECK: [[@LINE-2]]:18 | instance-property/ObjC | p1 | c:objc(cs)ClassReceivers(cpy)p1 | <no-cgname> | Ref,Writ,RelCont | rel: 1
// CHECK-NEXT: RelCont | classReceivers | c:@F@classReceivers
// CHECK: [[@LINE-4]]:18 | class-method/acc-set/ObjC | setP1: | c:objc(cs)ClassReceivers(cm)setP1: | +[ClassReceivers setP1:] | Ref,Call,Impl,RelCall,RelCont | rel: 1
// CHECK-NEXT: RelCall,RelCont | classReceivers | c:@F@classReceivers
  (void)ClassReceivers.p1;
// CHECK: [[@LINE-1]]:9 | class/ObjC | ClassReceivers | c:objc(cs)ClassReceivers | _OBJC_CLASS_$_ClassReceivers | Ref,RelCont | rel: 1
// CHECK: [[@LINE-2]]:24 | instance-property/ObjC | p1 | c:objc(cs)ClassReceivers(cpy)p1 | <no-cgname> | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | classReceivers | c:@F@classReceivers
// CHECK: [[@LINE-4]]:24 | class-method/acc-get/ObjC | p1 | c:objc(cs)ClassReceivers(cm)p1 | +[ClassReceivers p1] | Ref,Call,Impl,RelCall,RelCont | rel: 1
// CHECK-NEXT: RelCall,RelCont | classReceivers | c:@F@classReceivers

  ClassReceivers.implicit = 0;
// CHECK: [[@LINE-1]]:3 | class/ObjC | ClassReceivers | c:objc(cs)ClassReceivers | _OBJC_CLASS_$_ClassReceivers | Ref,RelCont | rel: 1
  (void)ClassReceivers.implicit;
// CHECK: [[@LINE-1]]:9 | class/ObjC | ClassReceivers | c:objc(cs)ClassReceivers | _OBJC_CLASS_$_ClassReceivers | Ref,RelCont | rel: 1
}

@interface ImplicitProperties

- (int)implicit;
- (void)setImplicit:(int)x;

+ (int)classImplicit;
+ (void)setClassImplicit:(int)y;

@end

void testImplicitProperties(ImplicitProperties *c) {
  c.implicit = 0;
// CHECK: [[@LINE-1]]:5 | instance-method/ObjC | setImplicit: | c:objc(cs)ImplicitProperties(im)setImplicit: | -[ImplicitProperties setImplicit:] | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 2
// CHECK-NEXT: RelCall,RelCont | testImplicitProperties | c:@F@testImplicitProperties
  c.implicit;
// CHECK: [[@LINE-1]]:5 | instance-method/ObjC | implicit | c:objc(cs)ImplicitProperties(im)implicit | -[ImplicitProperties implicit] | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 2
// CHECK-NEXT: RelCall,RelCont | testImplicitProperties | c:@F@testImplicitProperties
  ImplicitProperties.classImplicit = 1;
// CHECK: [[@LINE-1]]:22 | class-method/ObjC | setClassImplicit: | c:objc(cs)ImplicitProperties(cm)setClassImplicit: | +[ImplicitProperties setClassImplicit:] | Ref,Call,RelCall,RelCont | rel: 1
// CHECK-NEXT: RelCall,RelCont | testImplicitProperties | c:@F@testImplicitProperties
  ImplicitProperties.classImplicit;
// CHECK: [[@LINE-1]]:22 | class-method/ObjC | classImplicit | c:objc(cs)ImplicitProperties(cm)classImplicit | +[ImplicitProperties classImplicit] | Ref,Call,RelCall,RelCont | rel: 1
// CHECK-NEXT: RelCall,RelCont | testImplicitProperties | c:@F@testImplicitProperties
}

@interface EmptySelectors

- (int):(int)_; // CHECK: [[@LINE]]:8 | instance-method/ObjC | : | c:objc(cs)EmptySelectors(im): | -[EmptySelectors :]
- (void)test: (int)x :(int)y; // CHECK: [[@LINE]]:9 | instance-method/ObjC | test:: | c:objc(cs)EmptySelectors(im)test:: | -[EmptySelectors test::]
- (void):(int)_ :(int)m:(int)z; // CHECK: [[@LINE]]:9 | instance-method/ObjC | ::: | c:objc(cs)EmptySelectors(im)::: | -[EmptySelectors :::]

@end

@implementation EmptySelectors

- (int):(int)_ { // CHECK: [[@LINE]]:8 | instance-method/ObjC | : | c:objc(cs)EmptySelectors(im): | -[EmptySelectors :]
  [self :2]; // CHECK: [[@LINE]]:9 | instance-method/ObjC | : | c:objc(cs)EmptySelectors(im): | -[EmptySelectors :]
  return 0;
}

- (void)test: (int)x :(int)y { // CHECK: [[@LINE]]:9 | instance-method/ObjC | test:: | c:objc(cs)EmptySelectors(im)test:: | -[EmptySelectors test::]
}

- (void) :(int)_ :(int)m :(int)z { // CHECK: [[@LINE]]:10 | instance-method/ObjC | ::: | c:objc(cs)EmptySelectors(im)::: | -[EmptySelectors :::]
  [self test:0:1]; // CHECK: [[@LINE]]:9 | instance-method/ObjC | test:: | c:objc(cs)EmptySelectors(im)test:: | -[EmptySelectors test::]
  [self: 0: 1: 2]; // CHECK: [[@LINE]]:8 | instance-method/ObjC | ::: | c:objc(cs)EmptySelectors(im)::: | -[EmptySelectors :::]
}

@end

@protocol Prot3 // CHECK: [[@LINE]]:11 | protocol/ObjC | Prot3 | [[PROT3_USR:.*]] | <no-cgname> | Decl |
-(void)meth; // CHECK: [[@LINE]]:8 | instance-method(protocol)/ObjC | meth | [[PROT3_meth_USR:.*]] | -[Prot3 meth] | Decl,Dyn,RelChild |
@end

void test_rec1() {
  id<Prot3, Prot1> o1;
  [o1 meth]; // CHECK: [[@LINE]]:7 | instance-method(protocol)/ObjC | meth | [[PROT3_meth_USR]] | {{.*}} | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 3
    // CHECK-NEXT: RelCall,RelCont | test_rec1 |
    // CHECK-NEXT: RelRec | Prot3 | [[PROT3_USR]]
    // CHECK-NEXT: RelRec | Prot1 | [[PROT1_USR]]
  Base<Prot3, Prot1> *o2;
  [o2 meth]; // CHECK: [[@LINE]]:7 | instance-method/ObjC | meth | {{.*}} | Ref,Call,Dyn,RelRec,RelCall,RelCont | rel: 4
    // CHECK-NEXT: RelCall,RelCont | test_rec1 |
    // CHECK-NEXT: RelRec | Base | [[BASE_USR]]
    // CHECK-NEXT: RelRec | Prot3 | [[PROT3_USR]]
    // CHECK-NEXT: RelRec | Prot1 | [[PROT1_USR]]
}
