// RUN: %clang_cc1 %s -Wpotentially-direct-selector -verify
// RUN: %clang_cc1 %s -Wstrict-potentially-direct-selector -verify=expected,strict

#define NS_DIRECT __attribute__((objc_direct))

__attribute__((objc_root_class))
@interface Dummies
-(void)inBase;
-(void)inBaseImpl;
-(void)inBaseCat;
-(void)inBaseCatImpl;
-(void)inDerived;
-(void)inDerivedImpl;
-(void)inDerivedCat;
-(void)inDerivedCatImpl;
+(void)inBaseClass;
+(void)inDerivedClass;
+(void)inDerivedCatClass;
@end

__attribute__((objc_root_class))
@interface Base
-(void)inBase NS_DIRECT; // expected-note + {{direct method}}
+(void)inBaseClass NS_DIRECT;  // expected-note + {{direct method}}
@end

@implementation Base
-(void)inBaseImpl NS_DIRECT { // expected-note + {{direct method}}
}
-(void)inBase {}
+(void)inBaseClass {}
@end

@interface Base (Cat)
-(void)inBaseCat NS_DIRECT; // expected-note + {{direct method}}
@end

@implementation Base (Cat)
-(void)inBaseCatImpl NS_DIRECT { // expected-note + {{direct method}}
}
-(void)inBaseCat {}
@end

@interface Derived : Base
-(void)inDerived NS_DIRECT; // expected-note + {{direct method}}
+(void)inDerivedClass NS_DIRECT;  // expected-note + {{direct method}}
@end

@implementation Derived
-(void)inDerivedImpl NS_DIRECT { // expected-note + {{direct method}}
}
-(void)inDerived {}
+(void)inDerivedClass {}
@end

@interface Derived (Cat)
-(void)inDerivedCat NS_DIRECT; // expected-note + {{direct method}}
+(void)inDerivedCatClass NS_DIRECT; // expected-note + {{direct method}}
@end

@implementation Derived (Cat)
-(void)inDerivedCatImpl NS_DIRECT { // expected-note + {{direct method}}
}
-(void)inDerivedCat {}
+(void)inDerivedCatClass {}

-(void)test1 {
  (void)@selector(inBase); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseImpl); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCat); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCatImpl); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerived); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedImpl); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCat); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatImpl); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedClass); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseClass); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatClass); // expected-warning{{@selector expression formed with potentially direct selector}}
}
@end

void test2() {
  (void)@selector(inBase); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCat); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCatImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerived); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCat); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedClass); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseClass); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatClass); // strict-warning{{@selector expression formed with potentially direct selector}}
}

@interface OnlyBase : Base @end
@implementation OnlyBase
-(void)test3 {
  (void)@selector(inBase); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseImpl); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCat); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCatImpl); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerived); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCat); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedClass); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseClass); // expected-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatClass); // strict-warning{{@selector expression formed with potentially direct selector}}
}
@end

__attribute__((objc_root_class))
@interface Unrelated @end
@implementation Unrelated
-(void)test4 {
  (void)@selector(inBase); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCat); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseCatImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerived); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCat); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatImpl); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedClass); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inBaseClass); // strict-warning{{@selector expression formed with potentially direct selector}}
  (void)@selector(inDerivedCatClass); // strict-warning{{@selector expression formed with potentially direct selector}}
}
@end

@implementation Dummies
-(void)inBase {}
-(void)inBaseImpl {}
-(void)inBaseCat {}
-(void)inBaseCatImpl {}
-(void)inDerived {}
-(void)inDerivedImpl {}
-(void)inDerivedCat {}
-(void)inDerivedCatImpl {}
+(void)inBaseClass {}
+(void)inDerivedClass {}
+(void)inDerivedCatClass {}

-(void)test5 {
  (void)@selector(inBase);
  (void)@selector(inBaseImpl);
  (void)@selector(inBaseCat);
  (void)@selector(inBaseCatImpl);
  (void)@selector(inDerived);
  (void)@selector(inDerivedImpl);
  (void)@selector(inDerivedCat);
  (void)@selector(inDerivedCatImpl);
  (void)@selector(inDerivedClass);
  (void)@selector(inBaseClass);
  (void)@selector(inDerivedCatClass);
}
@end
