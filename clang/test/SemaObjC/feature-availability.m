// RUN: %clang_cc1 -fblocks -ffeature-availability=feature1:ON -ffeature-availability=feature2:OFF -fsyntax-only -verify %s
// RUN: %clang_cc1 -fblocks -fsyntax-only -verify -DUSE_DOMAIN %s

#include <feature-availability.h>

#define AVAIL 0
#define UNAVAIL 1

#ifdef USE_DOMAIN
int pred1(void);
static struct __AvailabilityDomain feature1 __attribute__((availability_domain(feature1))) = {__AVAILABILITY_DOMAIN_ENABLED, 0};
static struct __AvailabilityDomain feature2 __attribute__((availability_domain(feature2))) = {__AVAILABILITY_DOMAIN_DISABLED, 0};
#endif

__attribute__((availability(domain:feature1, AVAIL))) int func1(void);
__attribute__((availability(domain:feature1, UNAVAIL))) void func3(void);

struct __attribute__((availability(domain:feature1, UNAVAIL))) S0 {};
struct __attribute__((availability(domain:feature1, AVAIL))) S1 {};

@interface C0 {
  struct S0 ivar0; // expected-error {{use of 'S0' requires feature 'feature1' to be unavailable}}
  struct S1 ivar1; // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
  struct S1 ivar2 __attribute__((availability(domain:feature1, AVAIL)));
  struct S1 ivar3 __attribute__((availability(domain:feature1, UNAVAIL))); // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
}
@property struct S0 prop0; // expected-error {{use of 'S0' requires feature 'feature1' to be unavailable}}
@property struct S1 prop1; // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
@property struct S1 prop2 __attribute__((availability(domain:feature1, AVAIL)));
@property struct S1 prop3 __attribute__((availability(domain:feature1, UNAVAIL))); // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
-(struct S0)m0; // expected-error {{use of 'S0' requires feature 'feature1' to be unavailable}}
-(struct S1)m1; // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
-(struct S1)m2 __attribute__((availability(domain:feature1, AVAIL)));
-(struct S1)m3 __attribute__((availability(domain:feature1, UNAVAIL))); // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
@end

__attribute__((availability(domain:feature1, AVAIL))) // expected-note 2 {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
@interface Base0 {
  struct S0 ivar0; // expected-error {{use of 'S0' requires feature 'feature1' to be unavailable}}
  struct S1 ivar1;
  struct S1 ivar2 __attribute__((availability(domain:feature1, AVAIL)));
  struct S1 ivar3 __attribute__((availability(domain:feature1, UNAVAIL))); // expected-error {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
}
@property struct S0 prop0; // expected-error {{use of 'S0' requires feature 'feature1' to be unavailable}}
@property struct S1 prop1;
@property struct S1 prop2 __attribute__((availability(domain:feature1, AVAIL)));
@property struct S1 prop3 __attribute__((availability(domain:feature1, UNAVAIL))); // expected-error {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
@end

__attribute__((availability(domain:feature1, AVAIL), // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
               availability(domain:feature1, UNAVAIL))) // expected-note {{feature attribute __attribute__((availability(domain:feature1, 1))}}
@interface Base1 // expected-error {{cannot add feature availability to this decl}}
@end

@interface NSObject
@end

@interface Base7<T> : NSObject
@end

@interface Derived3 : Base7<Base0 *> // expected-error {{use of 'Base0' requires feature 'feature1' to be available}}
@end

__attribute__((availability(domain:feature1, AVAIL))) // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}} expected-note 2 {{feature attribute __attribute__((availability(domain:feature1, 0)))}}
@interface Derived0 : Base0 {
  struct S1 *ivar4;
}
@property struct S1 *p0;
@property int p1 __attribute__((availability(domain:feature1, AVAIL)));
@property int prop0_derived0 __attribute__((availability(domain:feature1, UNAVAIL))); // expected-error 3 {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}} expected-note 2 {{is incompatible with __attribute__((availability(domain:feature1, 1)))}}
@end

@interface Derived0()
@property struct S1 *p0_Ext;
@end

@implementation Derived0
-(void)m0 {
  func1();
  func3(); // expected-error {{use of 'func3' requires feature 'feature1' to be unavailable}}
}
-(void)m1 {
  self.p1 = 1;
}
@end

@interface Derived0(C0)
@property struct S1 *p0_C0;
@end

__attribute__((availability(domain:feature1, AVAIL))) // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
@interface Derived1 : Base0 {
  struct S1 *ivar4;
}
@property struct S1 *p0;
@end

@interface Derived1()
@property struct S1 *p0_Ext;
@end

@implementation Derived1
-(void)m0 {
  func1();
}
@end

@interface Derived1(C0)
@property struct S1 *p0_C0;
@end

__attribute__((availability(domain:feature1, UNAVAIL))) // expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
@interface Derived1(C1) // expected-error {{cannot merge incompatible feature attribute to this decl}}
@end

@protocol P0
@property struct S1 *p0; // expected-error {{use of 'S1' requires feature 'feature1' to be available}}
@end

__attribute__((availability(domain:feature1, AVAIL)))
@interface Derived2 : Base0
@end

__attribute__((availability(domain:feature1, AVAIL))) // expected-error {{feature attributes cannot be applied to ObjC class implementations}}
@implementation Derived2
@end

__attribute__((availability(domain:feature1, AVAIL)))
@interface Derived2(Cat1)
@end

__attribute__((availability(domain:feature1, AVAIL))) // expected-error {{feature attributes cannot be applied to ObjC category implementations}}
@implementation Derived2(Cat1)
@end

__attribute__((availability(domain:feature1, UNAVAIL)))
@protocol P1
@end

__attribute__((availability(domain:feature1, UNAVAIL)))
@interface Base2 <P1>
@end

__attribute__((availability(domain:feature1, UNAVAIL)))
@interface Base3 <P1>
@end

__attribute__((availability(domain:feature1, AVAIL)))
@interface Base4 <P1> // expected-error {{use of 'P1' requires feature 'feature1' to be unavailable}}
@end

__attribute__((availability(domain:feature1, AVAIL)))
@protocol P2
@end

@interface Base5
@end

__attribute__((availability(domain:feature1, AVAIL)))
@interface Base5(Cat2) <P2>
@end

__attribute__((availability(domain:feature1, AVAIL)))
@interface Base5(Cat3) <P2>
@end

__attribute__((availability(domain:feature1, UNAVAIL)))
@interface Base5(Cat4) <P2> // expected-error {{use of 'P2' requires feature 'feature1' to be available}}
@end

@interface Base6 <P1, P2> // expected-error {{use of 'P1' requires feature 'feature1' to be unavailable}} expected-error {{use of 'P2' requires feature 'feature1' to be available}}
@end
