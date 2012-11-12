// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp

void *sel_registerName(const char *);

@interface Root
+ (instancetype)alloc;
- (instancetype)init; // expected-note{{overridden method is part of the 'init' method family}}
- (instancetype)self;
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

@interface Subclass1 : Root
- (instancetype)initSubclass1;
- (void)methodOnSubclass1;
+ (instancetype)allocSubclass1;
@end

@interface Subclass2 : Root
- (instancetype)initSubclass2;
- (void)methodOnSubclass2;
@end

// Sanity check: the basic initialization pattern.
void test_instancetype_alloc_init_simple() {
  Root *r1 = [[Root alloc] init];
  Subclass1 *sc1 = [[Subclass1 alloc] init];
}

// Test that message sends to instancetype methods have the right type.
void test_instancetype_narrow_method_search() {
  // instancetype on class methods
  Subclass1 *sc1 = [[Subclass1 alloc] initSubclass2]; // expected-warning{{'Subclass1' may not respond to 'initSubclass2'}}
  Subclass2 *sc2 = [[Subclass2 alloc] initSubclass2]; // okay

  // instancetype on instance methods
  [[[Subclass1 alloc] init] methodOnSubclass2]; // expected-warning{{'Subclass1' may not respond to 'methodOnSubclass2'}}
  [[[Subclass2 alloc] init] methodOnSubclass2];
  
  // instancetype on class methods using protocols
  [[Subclass1<Proto1> alloc] methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  [[Subclass1<Proto2> alloc] methodInProto2];

  // instancetype on instance methods
  Subclass1<Proto1> *sc1proto1 = 0;
  [[sc1proto1 self] methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  Subclass1<Proto2> *sc1proto2 = 0;
  [[sc1proto2 self] methodInProto2];

  // Exact type checks
  // Message sends to Class.
  // FIXME. This is not supported due to missing capability in rewriter and not due to instancetype issues
  //  Subclass1<Proto1> *sc1proto1_2 = [[[sc1proto1 class] alloc] init];

  // Property access
  [sc1proto1.self methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  [sc1proto2.self methodInProto2];

  [sc1proto1.selfProp methodInProto2]; // expected-warning{{method '-methodInProto2' not found (return type defaults to 'id')}}
  [sc1proto2.selfProp methodInProto2];
}
