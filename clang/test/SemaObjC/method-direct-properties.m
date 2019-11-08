// RUN: %clang_cc1 -fsyntax-only -verify -Wselector-type-mismatch %s

@protocol ProtoDirectFail
@property(nonatomic, direct) int protoProperty; // expected-error {{'objc_direct' attribute cannot be applied to properties declared in an Objective-C protocol}}
@end

__attribute__((objc_root_class))
@interface Root
@property(nonatomic, direct) int propertyWithNonDirectGetter; // expected-note {{previous declaration is here}}
- (int)propertyWithNonDirectGetter;
- (int)propertyWithNonDirectGetter2;
- (int)propertyWithNonDirectGetterInParent;
- (int)propertyWithNonDirectGetterInParent2;

@property(nonatomic, readonly, direct) int getDirect_setDynamic;       // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int getDirect_setDirect;        // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int getDirect_setDirectMembers; // expected-note {{previous declaration is here}}

@property(nonatomic, readonly) int getDynamic_setDirect;
@property(nonatomic, readonly) int getDynamic_setDirectMembers;

@property(nonatomic, readonly) int dynamicProperty;
@property(nonatomic, readonly) int synthDynamicProperty;

@property(nonatomic, readonly, direct) int directProperty;      // expected-note {{previous declaration is here}}
@property(nonatomic, readonly, direct) int synthDirectProperty; // expected-note {{previous declaration is here}}
@end

__attribute__((objc_direct_members))
@interface
Root()
@property(nonatomic) int propertyWithNonDirectGetter2; // expected-note {{previous declaration is here}}

@property(nonatomic, readwrite) int getDirect_setDirectMembers;  // expected-note {{previous declaration is here}}
@property(nonatomic, readwrite) int getDynamic_setDirectMembers; // expected-note {{previous declaration is here}}
@end

@interface Root ()
@property(nonatomic, readwrite) int getDirect_setDynamic;
@property(nonatomic, readwrite, direct) int getDirect_setDirect; // expected-note {{previous declaration is here}}

@property(nonatomic, readwrite, direct) int getDynamic_setDirect; // expected-note {{previous declaration is here}}
@end

@interface Sub : Root
@property(nonatomic, direct) int propertyWithNonDirectGetterInParent; // expected-note {{previous declaration is here}}

- (int)propertyWithNonDirectGetter;          // no error: legal override
- (int)propertyWithNonDirectGetter2;         // no error: legal override
- (int)propertyWithNonDirectGetterInParent;  // no error: legal override
- (int)propertyWithNonDirectGetterInParent2; // no error: legal override

@end

__attribute__((objc_direct_members))
@interface Sub ()
@property(nonatomic) int propertyWithNonDirectGetterInParent2; // expected-note {{previous declaration is here}}
@end

// make sure that the `directness` of methods stuck,
// by observing errors trying to override the setter
@interface SubWitness : Sub

- (int)setPropertyWithNonDirectGetter:(int)value;          // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)setPropertyWithNonDirectGetter2:(int)value;         // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)setPropertyWithNonDirectGetterInParent:(int)value;  // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)setPropertyWithNonDirectGetterInParent2:(int)value; // expected-error {{cannot override a method that is declared direct by a superclass}}

- (int)getDirect_setDynamic; // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)setGetDirect_setDynamic:(int)value;
- (int)getDirect_setDirect;                      // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)setGetDirect_setDirect:(int)value;        // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)getDirect_setDirectMembers;               // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)setGetDirect_setDirectMembers:(int)value; // expected-error {{cannot override a method that is declared direct by a superclass}}

- (int)getDynamic_setDirect;
- (int)setGetDynamic_setDirect:(int)value; // expected-error {{cannot override a method that is declared direct by a superclass}}
- (int)getDynamic_setDirectMembers;
- (int)setGetDynamic_setDirectMembers:(int)value; // expected-error {{cannot override a method that is declared direct by a superclass}}
@end

__attribute__((objc_direct_members))
@implementation Root
- (int)propertyWithNonDirectGetter {
  return 42;
}
- (int)propertyWithNonDirectGetter2 {
  return 42;
}
- (int)propertyWithNonDirectGetterInParent {
  return 42;
}
- (int)propertyWithNonDirectGetterInParent2 {
  return 42;
}

- (int)dynamicProperty {
  return 42;
}
- (int)directProperty {
  return 42;
}
@end

@implementation Sub
- (int)propertyWithNonDirectGetter {
  return 42;
}
- (int)propertyWithNonDirectGetter2 {
  return 42;
}

- (int)dynamicProperty {
  return 42;
}
- (int)synthDynamicProperty {
  return 42;
}

- (int)directProperty { // expected-error {{cannot override a method that is declared direct by a superclass}}
  return 42;
}
- (int)synthDirectProperty { // expected-error {{cannot override a method that is declared direct by a superclass}}
  return 42;
}
@end
