// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_feature(objc_instancetype)
# error Missing 'instancetype' feature macro.
#endif

@interface Root
+ (instancetype)alloc; // expected-note {{explicitly declared 'instancetype'}}
- (instancetype)init; // expected-note{{overridden method is part of the 'init' method family}}
- (instancetype)self; // expected-note {{explicitly declared 'instancetype'}}
- (Class)class;

@property (assign) Root *selfProp;
- (instancetype)selfProp;
@end

@protocol Proto1
@optional
- (instancetype)methodInProto1;
@end

@protocol Proto2
@optional
- (instancetype)methodInProto2; // expected-note{{overridden method returns an instance of its class type}}
- (instancetype)otherMethodInProto2; // expected-note{{overridden method returns an instance of its class type}}
@end

@interface Subclass1 : Root // expected-note 4 {{receiver is instance of class declared here}}
- (instancetype)initSubclass1;
- (void)methodOnSubclass1;
+ (instancetype)allocSubclass1;
@end

@interface Subclass2 : Root
- (instancetype)initSubclass2;
- (void)methodOnSubclass2;
@end

// Check the basic initialization pattern.
void test_instancetype_alloc_init_simple(void) {
  Root *r1 = [[Root alloc] init];
  Subclass1 *sc1 = [[Subclass1 alloc] init];
}

// Test that message sends to instancetype methods have the right type.
void test_instancetype_narrow_method_search(void) {
  // instancetype on class methods
  Subclass1 *sc1 = [[Subclass1 alloc] initSubclass2]; // expected-warning{{'Subclass1' may not respond to 'initSubclass2'}}
  Subclass2 *sc2 = [[Subclass2 alloc] initSubclass2]; // okay

  // instancetype on instance methods
  [[[Subclass1 alloc] init] methodOnSubclass2]; // expected-warning{{'Subclass1' may not respond to 'methodOnSubclass2'}}
  [[[Subclass2 alloc] init] methodOnSubclass2];
  
  // instancetype on class methods using protocols
  typedef Subclass1<Proto1> SC1Proto1;
  typedef Subclass1<Proto2> SC1Proto2;
  [[SC1Proto1 alloc] methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  [[SC1Proto2 alloc] methodInProto2];

  // instancetype on instance methods
  Subclass1<Proto1> *sc1proto1 = 0;
  [[sc1proto1 self] methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  Subclass1<Proto2> *sc1proto2 = 0;
  [[sc1proto2 self] methodInProto2];

  // Exact type checks
  typeof([[Subclass1 alloc] init]) *ptr1 = (Subclass1 **)0;
  typeof([[Subclass2 alloc] init]) *ptr2 = (Subclass2 **)0;

  // Message sends to Class.
  Subclass1<Proto1> *sc1proto1_2 = [[[sc1proto1 class] alloc] init];

  // Property access
  [sc1proto1.self methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  [sc1proto2.self methodInProto2];
  [Subclass1.alloc initSubclass2]; // expected-warning{{'Subclass1' may not respond to 'initSubclass2'}}
  [Subclass2.alloc initSubclass2];

  [sc1proto1.selfProp methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  [sc1proto2.selfProp methodInProto2];
}

// Test that message sends to super methods have the right type.
@interface Subsubclass1 : Subclass1
- (instancetype)initSubclass1;
+ (instancetype)allocSubclass1;

- (void)onlyInSubsubclass1;
@end

@implementation Subsubclass1
- (instancetype)initSubclass1 {
  // Check based on method search.
  [[super initSubclass1] methodOnSubclass2]; // expected-warning{{'Subsubclass1' may not respond to 'methodOnSubclass2'}}
  [super.initSubclass1 methodOnSubclass2]; // expected-warning{{'Subsubclass1' may not respond to 'methodOnSubclass2'}}

  self = [super init]; // common pattern

  // Exact type check.
  typeof([super initSubclass1]) *ptr1 = (Subsubclass1**)0;

  return self;
}

+ (instancetype)allocSubclass1 {
  // Check based on method search.
  [[super allocSubclass1] methodOnSubclass2]; // expected-warning{{'Subsubclass1' may not respond to 'methodOnSubclass2'}}

  // The ASTs don't model super property accesses well enough to get this right
  [super.allocSubclass1 methodOnSubclass2]; // expected-warning{{'Subsubclass1' may not respond to 'methodOnSubclass2'}}

  // Exact type check.
  typeof([super allocSubclass1]) *ptr1 = (Subsubclass1**)0;
  
  return [super allocSubclass1];
}

- (void)onlyInSubsubclass1 {}
@end

// Check compatibility rules for inheritance of related return types.
@class Subclass4;

@interface Subclass3 <Proto1, Proto2>
- (Subclass3 *)methodInProto1;
- (Subclass4 *)methodInProto2; // expected-warning{{method is expected to return an instance of its class type 'Subclass3', but is declared to return 'Subclass4 *'}}
@end

@interface Subclass4 : Root
+ (Subclass4 *)alloc; // okay
- (Subclass3 *)init; // expected-warning{{method is expected to return an instance of its class type 'Subclass4', but is declared to return 'Subclass3 *'}}
- (id)self; // expected-note{{overridden method is part of the 'self' method family}}
- (instancetype)initOther;
@end

@protocol Proto3 <Proto1, Proto2>
@optional
- (id)methodInProto1;
- (Subclass1 *)methodInProto2;
- (int)otherMethodInProto2; // expected-warning{{protocol method is expected to return an instance of the implementing class, but is declared to return 'int'}}
@end

@implementation Subclass4
+ (id)alloc {
  return self; // expected-warning{{incompatible pointer types returning 'Class' from a function with result type 'Subclass4 *'}}
}

- (Subclass3 *)init { return 0; } // don't complain: we lost the related return type

- (Subclass3 *)self { return 0; } // expected-warning{{method is expected to return an instance of its class type 'Subclass4', but is declared to return 'Subclass3 *'}}

- (Subclass4 *)initOther { return 0; }

@end

// Check that inherited related return types influence the types of
// message sends.
void test_instancetype_inherited(void) {
  [[Subclass4 alloc] initSubclass1]; // expected-warning{{'Subclass4' may not respond to 'initSubclass1'}}
  [[Subclass4 alloc] initOther];
}

// Check that related return types tighten up the semantics of
// Objective-C method implementations.
@implementation Subclass2
- (instancetype)initSubclass2 { // expected-note {{explicitly declared 'instancetype'}}
  Subclass1 *sc1 = [[Subclass1 alloc] init];
  return sc1; // expected-warning{{incompatible pointer types returning 'Subclass1 *' from a function with result type 'Subclass2 *'}}
}
- (void)methodOnSubclass2 {}
- (id)self {
  Subclass1 *sc1 = [[Subclass1 alloc] init];
  return sc1; // expected-warning{{incompatible pointer types returning 'Subclass1 *' from a function with result type 'Subclass2 *'}}
}
@end

@interface MyClass : Root
+ (int)myClassMethod;
@end

@implementation MyClass
+ (int)myClassMethod { return 0; }

- (void)blah {
  int i = [[MyClass self] myClassMethod];
}

@end

// rdar://12493140
@protocol P4
- (instancetype) foo; // expected-note {{current method is explicitly declared 'instancetype' and is expected to return an instance of its class type}}
@end
@interface A4 : Root <P4>
- (instancetype) bar; // expected-note {{current method is explicitly declared 'instancetype' and is expected to return an instance of its class type}}
- (instancetype) baz; // expected-note {{overridden method returns an instance of its class type}} expected-note {{previous definition is here}}
@end
@interface B4 : Root @end

@implementation A4 {
  B4 *_b;
}
- (id) foo {
  return _b; // expected-warning {{incompatible pointer types returning 'B4 *' from a function with result type 'A4 *'}}
}
- (id) bar {
  return _b; // expected-warning {{incompatible pointer types returning 'B4 *' from a function with result type 'A4 *'}}
}

// This is really just to ensure that we don't crash.
// FIXME: only one diagnostic, please
- (float) baz { // expected-warning {{method is expected to return an instance of its class type 'A4', but is declared to return 'float'}} expected-warning {{conflicting return type in implementation}}
  return 0;
}
@end
