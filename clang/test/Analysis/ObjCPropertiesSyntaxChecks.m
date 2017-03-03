// RUN: %clang_analyze_cc1 -w -fblocks -analyzer-checker=osx.ObjCProperty %s -verify

#include "Inputs/system-header-simulator-objc.h"

@interface I : NSObject {
  NSMutableString *_mutableExplicitStr;
  NSMutableString *_trulyMutableStr;
  NSMutableString *_trulyMutableExplicitStr;
}
@property(copy) NSString *str; // no-warning
@property(copy) NSMutableString *mutableStr; // expected-warning{{Property of mutable type 'NSMutableString' has 'copy' attribute; an immutable object will be stored instead}}
@property(copy) NSMutableString *mutableExplicitStr; // expected-warning{{Property of mutable type 'NSMutableString' has 'copy' attribute; an immutable object will be stored instead}}
@property(copy, readonly) NSMutableString *mutableReadonlyStr; // no-warning
@property(copy, readonly) NSMutableString *mutableReadonlyStrOverriddenInChild; // no-warning
@property(copy, readonly) NSMutableString *mutableReadonlyStrOverriddenInCategory; // no-warning
@property(copy) NSMutableString *trulyMutableStr; // no-warning
@property(copy) NSMutableString *trulyMutableExplicitStr; // no-warning
@property(copy) NSMutableString *trulyMutableStrWithSynthesizedStorage; // no-warning
@end

@interface I () {}
@property(copy) NSMutableString *mutableStrInCategory; // expected-warning{{Property of mutable type 'NSMutableString' has 'copy' attribute; an immutable object will be stored instead}}
@property (copy, readwrite) NSMutableString *mutableReadonlyStrOverriddenInCategory; // expected-warning{{Property of mutable type 'NSMutableString' has 'copy' attribute; an immutable object will be stored instead}}
@end

@implementation I
@synthesize mutableExplicitStr = _mutableExplicitStr;
- (NSMutableString *)trulyMutableStr {
  return _trulyMutableStr;
}
- (void)setTrulyMutableStr: (NSMutableString *) S {
  _trulyMutableStr = [S mutableCopy];
}
@dynamic trulyMutableExplicitStr;
- (NSMutableString *)trulyMutableExplicitStr {
  return _trulyMutableExplicitStr;
}
- (void)setTrulyMutableExplicitStr: (NSMutableString *) S {
  _trulyMutableExplicitStr = [S mutableCopy];
}
@synthesize trulyMutableStrWithSynthesizedStorage;
- (NSMutableString *)trulyMutableStrWithSynthesizedStorage {
  return trulyMutableStrWithSynthesizedStorage;
}
- (void)setTrulyMutableStrWithSynthesizedStorage: (NSMutableString *) S {
  trulyMutableStrWithSynthesizedStorage = [S mutableCopy];
}
@end

@interface J : I {}
@property (copy, readwrite) NSMutableString *mutableReadonlyStrOverriddenInChild; // expected-warning{{Property of mutable type 'NSMutableString' has 'copy' attribute; an immutable object will be stored instead}}
@end

@implementation J
@end

// If we do not see the implementation then we do not want to warn,
// because we may miss a user-defined setter that works correctly.
@interface IWithoutImpl : NSObject {}
@property(copy) NSMutableString *mutableStr; // no-warning
@end

@protocol SomeProtocol
// Don't warn on protocol properties because it is possible to
// conform to them correctly; it is only synthesized setters that
// that are definitely incorrect.
@property (copy) NSMutableString *myProp; // no-crash // no-warning
@end
