// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config ipa=dynamic-bifurcate -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_checkInlined(int);
void clang_analyzer_eval(int);

// Test inlining of ObjC class methods.

typedef signed char BOOL;
#define YES ((BOOL)1)
#define NO ((BOOL)0)
typedef struct objc_class *Class;
typedef struct objc_object {
  Class isa;
} * id;
@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@interface NSObject <NSObject> {}
+ (id)alloc;
+ (Class)class;
+ (Class)superclass;
- (id)init;
- (id)autorelease;
- (id)copy;
- (Class)class;
- (instancetype)self;
- (id)retain;
@end

// Vanila: ObjC class method is called by name.
@interface MyParent : NSObject
+ (int)getInt;
@end
@interface MyClass : MyParent
+ (int)getInt;
@end
@implementation MyClass
+ (int)testClassMethodByName {
    int y = [MyClass getInt];
    return 5/y; // expected-warning {{Division by zero}}
}
+ (int)getInt {
  return 0;
}
@end

// The definition is defined by the parent. Make sure we find it and inline.
@interface MyParentDIP : NSObject
+ (int)getInt;
@end
@interface MyClassDIP : MyParentDIP
@end
@implementation MyClassDIP
+ (int)testClassMethodByName {
    int y = [MyClassDIP getInt];
    return 5/y; // expected-warning {{Division by zero}}
}
@end
@implementation MyParentDIP
+ (int)getInt {
    return 0;
}
@end

// ObjC class method is called by name. Definition is in the category.
@interface AAA : NSObject
@end
@interface AAA (MyCat)
+ (int)getInt;
@end
int foo() {
    int y = [AAA getInt];
    return 5/y; // expected-warning {{Division by zero}}
}
@implementation AAA
@end
@implementation AAA (MyCat)
+ (int)getInt {
    return 0;
}
@end

// ObjC class method is called by name. Definition is in the parent category.
@interface PPP : NSObject
@end
@interface PPP (MyCat)
+ (int)getInt;
@end
@interface CCC : PPP
@end
int foo4() {
    int y = [CCC getInt];
    return 5/y; // expected-warning {{Division by zero}}
}
@implementation PPP
@end
@implementation PPP (MyCat)
+ (int)getInt {
    return 0;
}
@end

// There is no declaration in the class but there is one in the parent. Make 
// sure we pick the definition from the class and not the parent.
@interface MyParentTricky : NSObject
+ (int)getInt;
@end
@interface MyClassTricky : MyParentTricky
@end
@implementation MyParentTricky
+ (int)getInt {
    return 0;
}
@end
@implementation MyClassTricky
+ (int)getInt {
  return 1;
}
+ (int)testClassMethodByName {
    int y = [MyClassTricky getInt];
    return 5/y; // no-warning
}
@end

// ObjC class method is called by unknown class declaration (passed in as a 
// parameter). We should not inline in such case.
@interface MyParentUnknown : NSObject
+ (int)getInt;
@end
@interface MyClassUnknown : MyParentUnknown
+ (int)getInt;
@end
@implementation MyClassUnknown
+ (int)testClassVariableByUnknownVarDecl: (Class)cl  {
  int y = [cl getInt];
  return 3/y; // no-warning
}
+ (int)getInt {
  return 0;
}
@end

// ObjC class method call through a decl with a known type.
// Note, [self class] could be a subclass. Do we still want to inline here?
@interface MyClassKT : NSObject
@end
@interface MyClassKT (MyCatKT)
+ (int)getInt;
@end
@implementation MyClassKT (MyCatKT)
+ (int)getInt {
    return 0;
}
@end
@implementation MyClassKT
- (int)testClassMethodByKnownVarDecl {
  Class currentClass = [self class];
  int y = [currentClass getInt];
  return 5 / y; // expected-warning{{Division by zero}}
}
@end

// Another false negative due to us not reasoning about self, which in this 
// case points to the object of the class in the call site and should be equal 
// to [MyParent class].
@interface MyParentSelf : NSObject
+ (int)testSelf;
@end
@implementation MyParentSelf
+ (int)testSelf {
  if (self == [MyParentSelf class])
      return 0;
    else
      return 1;
}
@end
@interface MyClassSelf : MyParentSelf
@end
@implementation MyClassSelf
+ (int)testClassMethodByKnownVarDecl {
  int y = [MyParentSelf testSelf];
  return 5/y; // expected-warning{{Division by zero}}
}
@end
int foo2() {
  int y = [MyParentSelf testSelf];
  return 5/y; // expected-warning{{Division by zero}}
}

// TODO: We do not inline 'getNum' in the following case, where the value of 
// 'self' in call '[self getNum]' is available and evaualtes to 
// 'SelfUsedInParentChild' if it's called from fooA.
// Self region should get created before we call foo and yje call to super 
// should keep it live. 
@interface SelfUsedInParent : NSObject
+ (int)getNum;
+ (int)foo;
@end
@implementation SelfUsedInParent
+ (int)getNum {return 5;}
+ (int)foo {
  int r = [self getNum];
  clang_analyzer_eval(r == 5); // expected-warning{{TRUE}}
  return r;
}
@end
@interface SelfUsedInParentChild : SelfUsedInParent
+ (int)getNum;
+ (int)fooA;
@end
@implementation SelfUsedInParentChild
+ (int)getNum {return 0;}
+ (int)fooA {
  return [super foo];
}
@end
int checkSelfUsedInparentClassMethod() {
    return 5/[SelfUsedInParentChild fooA];
}


@interface Rdar15037033 : NSObject
@end

void rdar15037033() {
  [Rdar15037033 forwardDeclaredMethod]; // expected-warning {{class method '+forwardDeclaredMethod' not found}}
  [Rdar15037033 forwardDeclaredVariadicMethod:1, 2, 3, 0]; // expected-warning {{class method '+forwardDeclaredVariadicMethod:' not found}}
}

@implementation Rdar15037033

+ (void)forwardDeclaredMethod {
  clang_analyzer_checkInlined(1); // expected-warning{{TRUE}}
}

+ (void)forwardDeclaredVariadicMethod:(int)x, ... {
  clang_analyzer_checkInlined(0); // no-warning
}
@end

@interface SelfClassTestParent : NSObject
-(unsigned)returns10;
+(unsigned)returns20;
+(unsigned)returns30;
@end

@interface SelfClassTest : SelfClassTestParent
- (unsigned)returns10;
+ (unsigned)returns20;
+ (unsigned)returns30;
@end

@implementation SelfClassTestParent
- (unsigned)returns10 {
  return 100;
}
+ (unsigned)returns20 {
  return 100;
}
+ (unsigned)returns30 {
  return 100;
}

- (void)testSelfReassignment {
  // Check that we didn't hardcode type for self.
  self = [[[SelfClassTest class] alloc] init];
  Class actuallyChildClass = [self class];
  unsigned result = [actuallyChildClass returns30];
  clang_analyzer_eval(result == 30); // expected-warning{{TRUE}}
}
@end

@implementation SelfClassTest
- (unsigned)returns10 {
  return 10;
}
+ (unsigned)returns20 {
  return 20;
}
+ (unsigned)returns30 {
  return 30;
}
+ (BOOL)isClass {
  return YES;
}
- (BOOL)isClass {
  return NO;
}
+ (SelfClassTest *)create {
  return [[self alloc] init];
}
+ (void)classMethod {
  unsigned result1 = [self returns20];
  clang_analyzer_eval(result1 == 20); // expected-warning{{TRUE}}

  unsigned result2 = [[self class] returns30];
  clang_analyzer_eval(result2 == 30); // expected-warning{{TRUE}}

  unsigned result3 = [[super class] returns30];
  clang_analyzer_eval(result3 == 100); // expected-warning{{TRUE}}

  // Check that class info is propagated with data
  Class class41 = [self class];
  Class class42 = class41;
  unsigned result4 = [class42 returns30];
  clang_analyzer_eval(result4 == 30); // expected-warning{{TRUE}}

  Class class51 = [super class];
  Class class52 = class51;
  unsigned result5 = [class52 returns30];
  clang_analyzer_eval(result5 == 100); // expected-warning{{TRUE}}
}
- (void)instanceMethod {
  unsigned result0 = [self returns10];
  clang_analyzer_eval(result0 == 10); // expected-warning{{TRUE}}

  unsigned result2 = [[self class] returns30];
  clang_analyzer_eval(result2 == 30); // expected-warning{{TRUE}}

  unsigned result3 = [[super class] returns30];
  clang_analyzer_eval(result3 == 100); // expected-warning{{TRUE}}

  // Check that class info is propagated with data
  Class class41 = [self class];
  Class class42 = class41;
  unsigned result4 = [class42 returns30];
  clang_analyzer_eval(result4 == 30); // expected-warning{{TRUE}}

  Class class51 = [super class];
  Class class52 = class51;
  unsigned result5 = [class52 returns30];
  clang_analyzer_eval(result5 == 100); // expected-warning{{TRUE}}

  // Check that we inline class methods when class object is a receiver
  Class class6 = [self class];
  BOOL calledClassMethod = [class6 isClass];
  clang_analyzer_eval(calledClassMethod == YES); // expected-warning{{TRUE}}

  // Check that class info is propagated through the 'self' method
  Class class71 = [self class];
  Class class72 = [class71 self];
  unsigned result7 = [class72 returns30];
  clang_analyzer_eval(result7 == 30); // expected-warning{{TRUE}}

  // Check that 'class' and 'super' info from direct invocation of the
  // corresponding class methods is propagated with data
  Class class8 = [SelfClassTest class];
  unsigned result8 = [class8 returns30];
  clang_analyzer_eval(result8 == 30); // expected-warning{{TRUE}}

  Class class9 = [SelfClassTest superclass];
  unsigned result9 = [class9 returns30];
  clang_analyzer_eval(result9 == 100); // expected-warning{{TRUE}}

  // Check that we get class from a propagated type
  SelfClassTestParent *selfAsParent10 = [[SelfClassTest alloc] init];
  Class class10 = [selfAsParent10 class];
  unsigned result10 = [class10 returns30];
  clang_analyzer_eval(result10 == 30); // expected-warning{{TRUE}}

  SelfClassTestParent *selfAsParent11 = [[[self class] alloc] init];
  Class class11 = [selfAsParent11 class];
  unsigned result11 = [class11 returns30];
  clang_analyzer_eval(result11 == 30); // expected-warning{{TRUE}}
}
@end

@interface Parent : NSObject
+ (int)a;
+ (int)b;
@end
@interface Child : Parent
@end
@interface Other : NSObject
+(void)run;
@end
int main(int argc, const char * argv[]) {
  @autoreleasepool {
    [Other run];
  }
  return 0;
}
@implementation Other
+(void)run {
  int result = [Child a];
  // TODO: This should return 100.
  clang_analyzer_eval(result == 12); // expected-warning{{TRUE}}
}
@end
@implementation Parent
+ (int)a; {
  return [self b];
}
+ (int)b; {
  return 12;
}
@end
@implementation Child
+ (int)b; {
  return 100;
}
@end
